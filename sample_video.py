import os
import sys
import math
import argparse
import json
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
from PIL import Image, ImageOps
import imageio
import time
import gc

import torch
import numpy as np
from einops import rearrange, repeat
from torchvision.utils import make_grid
import torchvision.transforms as TT

from sgm.util import get_obj_from_str, isheatmap, exists

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

import diffusion_video
from arguments import get_args, process_config_to_args
import torch.nn.functional as F

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input('Please input English text (Ctrl-D quit): ')
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass

def read_from_file(p, rank=0, world_size=1):
    with open(p, 'r') as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
    if args.load is not None:
        load_checkpoint(model, args)
    model.eval()

    if args.input_type == 'cli':
        assert mpu.get_data_parallel_world_size() == 1, 'Only dp = 1 supported in cli mode.'
        data_iter = read_from_cli()
    elif args.input_type == 'txt':
        dp_rank, dp_world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        data_iter = read_from_file(args.input_file, rank=dp_rank, world_size=dp_world_size)
    else:
        raise NotImplementedError

    sample_func = model.sample

    num_samples = [1]
    force_uc_zero_embeddings = []

    vae_compress_size = args.vae_compress_size
    print('VAE_compress_size:', vae_compress_size)
    # if args.image2video:
    #     zero_pad_dict = torch.load('zero_pad_dict.pt', map_location='cpu')

    with torch.no_grad():
        torch.distributed.barrier(group=mpu.get_data_broadcast_group())
        while True:
            stopped = False
            if mpu.get_data_broadcast_rank() == 0:
                try:
                    text, cnt = next(data_iter)
                except StopIteration:
                    text = ''
                    stopped = True

                # text = 'FPS-%d. ' % args.sampling_fps + text

            else:
                text = ''
                cnt = 0

            broadcast_list = [text, cnt, stopped]
            # broadcast
            mp_size = mpu.get_model_parallel_world_size()
            sp_size = mpu.get_sequence_parallel_world_size()

            if mp_size > 1 or sp_size > 1:
                torch.distributed.broadcast_object_list(broadcast_list, src=mpu.get_data_broadcast_src_rank(), group=mpu.get_data_broadcast_group())

            text, cnt, stopped = broadcast_list
            if stopped:
                break

            if mpu.get_data_broadcast_rank() == 0:
                print(cnt, ': ', text)

            images_nums = 0
            if args.s2v_concat:
                infos = text.split('@@')
                image_size = args.sampling_image_size
                concat_subjects = []
                subjects_save = []
                for subject in infos[1:]:
                    images_nums += 1
                    assert os.path.exists(subject), subject
                    ref_img = Image.open(subject).convert('RGB')
                    img_save = np.array(ref_img)
                    subject_image = torch.from_numpy(img_save).unsqueeze(0).permute(0,3,1,2).contiguous()
                    subjects_save.append(subject_image.squeeze(0))
                    if args.new_straetgy:
                        h, w = args.sampling_image_size
                        img_ratio = ref_img.width / ref_img.height
                        target_ratio = w / h
                        
                        if img_ratio > target_ratio:
                            new_width = w
                            new_height = int(new_width / img_ratio)
                        else: 
                            new_height = h
                            new_width = int(new_height * img_ratio)
                        
                        ref_img = ref_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        delta_w = w - ref_img.size[0]
                        delta_h = h - ref_img.size[1]
                        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                        new_img = ImageOps.expand(ref_img, padding, fill=(255, 255, 255)) 
                        subject_image = torch.from_numpy(np.array(new_img)).unsqueeze(0).permute(0,3,1,2).contiguous()
                    subject_image = (subject_image - 127.5) / 127.5
                    

                    subject_image = subject_image.unsqueeze(2).to(torch.bfloat16).to("cuda")
                    subject_image = model.encode_first_stage(subject_image, None, force_encode=True)
                    subject_image = subject_image.permute(0, 2, 1, 3, 4).contiguous() # BCTHW -> BTCHW
                    concat_subjects.append(subject_image) #B c t h w

                concat_subjects = torch.cat(concat_subjects, dim=1) if not args.subject_dynamic else concat_subjects
                
                T = args.sampling_num_frames
                C, H, W = args.latent_channels, args.sampling_image_size[0] // args.vae_compress_size[-1], args.sampling_image_size[1] // args.vae_compress_size[-1]
            
            else:
                image = None
                T = args.sampling_num_frames
                C, H, W = args.latent_channels, args.sampling_image_size[0]//vae_compress_size[1], args.sampling_image_size[1]//vae_compress_size[2]

            # TODO: broadcast image2video
            value_dict = {
                'prompt': text,
                'negative_prompt': args.sample_neg_prompt if args.sample_neg_prompt is not None else '',
                'num_frames': torch.tensor(T).unsqueeze(0)
            }

            save_dir = os.path.join(args.output_dir, str(cnt) + '_' + text.replace(' ', '_').replace('/', '').replace(',','_')[:120])
            os.makedirs(save_dir, exist_ok=True)
            if args.only_save_latents:
                save_path = os.path.join(save_dir, 'latent.pt')
            else:
                save_path = os.path.join(save_dir, 'output.mp4')
            if os.path.exists(save_path):
                continue
            with open(os.path.join(save_dir, 'text.txt'), 'w') as f:
                f.write(text)

            model.conditioner.embedders[0].to('cuda')
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            model.conditioner.embedders[0].cpu()

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(
                        lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                    )
            
            if args.s2v_concat:
                c['concat_subjects'] = concat_subjects.to(model.dtype) if isinstance(concat_subjects, torch.Tensor) else [concat_subjects[i].to(model.dtype) for i in range(len(concat_subjects))]
                uc['concat_subjects'] = concat_subjects.to(model.dtype) if isinstance(concat_subjects, torch.Tensor) else [concat_subjects[i].to(model.dtype) for i in range(len(concat_subjects))]
                # uc['concat_subjects'] = concat_subjects.to(model.dtype) if not args.image_condition_zero else torch.zeros_like(concat_subjects).to(model.dtype)
                
            for index in range(args.batch_size):
                samples_z = sample_func(
                    c,
                    uc = uc,
                    batch_size = 1,
                    shape = (T, C, H, W),
                    ofs = torch.tensor([2.0]).to('cuda'),
                    fps = torch.tensor([args.sampling_fps]).to('cuda'),
                )
                if mpu.get_sequence_parallel_rank() == 0:
                    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                    if args.only_save_latents:
                        if mpu.get_model_parallel_rank() == 0:
                            samples_z = 1.0 / model.scale_factor * samples_z
                            torch.save(samples_z, save_path)
                    else:
                        if args.subject2video:
                            samples_x = model.decode_first_stage(samples_z[:,:,images_nums:]).to(torch.float32)
                            samples_x = samples_x.permute(0, 2, 3, 4, 1).squeeze(0).contiguous() # BCTHW -> THWC
                            samples = (torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)*255.0).cpu().numpy().astype(np.uint8)
                            if mpu.get_model_parallel_rank() == 0:
                                save_subject_image_path(subjects_save, save_dir)
                                with imageio.get_writer(save_path, fps=args.sampling_fps) as writer:
                                    for frame in samples:
                                        writer.append_data(frame)
                        else:
                            samples_x = model.decode_first_stage(samples_z).to(torch.float32)
                            samples_x = samples_x.permute(0, 2, 3, 4, 1).squeeze(0).contiguous() # BCTHW -> THWC
                            samples = (torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)*255.0).cpu().numpy().astype(np.uint8)
                            if mpu.get_model_parallel_rank() == 0:
                                save_subject_image_path(subjects_save, save_dir)
                                with imageio.get_writer(save_path, fps=args.sampling_fps) as writer:
                                    for frame in samples:
                                        writer.append_data(frame)

                
            gc.collect()
            torch.cuda.empty_cache()

def save_subject_image_path(subjects, path):
    for i, x in enumerate(subjects):
        x = np.array(x.squeeze(0).permute(1,2,0).cpu().numpy().astype('uint8'))
        filepath = os.path.join(path, f'subject_{i}.png')
        image = Image.fromarray(x)
        image.save(filepath)

if __name__ == '__main__':
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    if "sigma_sampler_config" in args.model_config.loss_fn_config.params.keys() and hasattr(args.model_config.loss_fn_config.params.sigma_sampler_config.params, "uniform_sampling"):
        args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    if args.model_type == "dit":
        Engine = diffusion_video.SATVideoDiffusionEngine

    sampling_main(args, model_cls=Engine)
