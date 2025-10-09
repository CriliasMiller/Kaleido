import math
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union, Optional
from omegaconf import ListConfig, OmegaConf
from copy import deepcopy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from sgm.modules.diffusionmodules.loss import guidance_scale_embedding
from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
    SeededNoise
)
from sgm.modules.diffusionmodules.util import (
    timestep_embedding
)
from sat import mpu
from sat.helpers import print_rank0
from sat.training.model_io import load_checkpoint
from sat.mpu.operation import mp_split_model_rank0, mp_split_model_receive, mp_merge_model_rank0, mp_merge_model_send
from sat.arguments import update_args_with_file, overwrite_args_by_dict, set_random_seed
from sat.mpu.initialize import get_node_rank, get_model_parallel_rank, destroy_model_parallel, initialize_model_parallel
from sat.model.base_model import get_model
import gc
from sat.arguments import reset_random_seed
import random

class SATVideoDiffusionEngine(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        model_config = args.model_config
        # model args preprocess
        log_keys = model_config.get('log_keys', None)
        input_key = model_config.get('input_key', 'mp4')
        network_config = model_config.get('network_config', None)
        network_wrapper = model_config.get('network_wrapper', None)
        denoiser_config = model_config.get('denoiser_config', None)
        sampler_config = model_config.get('sampler_config', None)
        conditioner_config = model_config.get('conditioner_config', None)
        i2v_clip_config = model_config.get('i2v_clip_config', None)
        first_stage_config = model_config.get('first_stage_config', None)
        loss_fn_config = model_config.get('loss_fn_config', None)
        scale_factor = model_config.get('scale_factor', 1.0)
        latent_input = model_config.get('latent_input', False)
        disable_first_stage_autocast = model_config.get('disable_first_stage_autocast', False)
        no_cond_log = model_config.get('disable_first_stage_autocast', False)
        untrainable_prefixs = model_config.get('untrainable_prefixs', ['first_stage_model', 'conditioner'])
        compile_model = model_config.get('compile_model', False)
        en_and_decode_n_samples_a_time = model_config.get('en_and_decode_n_samples_a_time', None)
        lora_train = model_config.get('lora_train', False)
        self.use_pd = model_config.get('use_pd', False) # progressive distillation
        self.use_i2v_clip = model_config.get('use_i2v_clip', False) # inspired from wanx-i2v
        self.i2v_encode_video = model_config.get('i2v_encode_video', False) # inspired from wanx-i2v

        self.log_keys = log_keys
        self.input_key = input_key
        self.untrainable_prefixs = untrainable_prefixs
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lora_train = lora_train
        self.noised_image_input = model_config.get('noised_image_input', False)
        self.subjects_input = model_config.get('subjects_input', False)
        self.noised_image_all_concat = model_config.get('noised_image_all_concat', False)
        self.image_cond_dropout = model_config.get('image_cond_dropout', 0.0)
        self.subject_cond_dropout = model_config.get('subject_cond_dropout', 0.0)
        # self.final_concat = model_config.get('final_concat', False)
        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str

        # if args.fsdp2:
        #     dtype = torch.float32
        #     dtype_str = "fp32"
        ## fsdp2 model dtype should be fp32
        network_config['params']['dtype'] = dtype_str
        network_config['params']['use_i2v_clip'] = self.use_i2v_clip
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=dtype
        )
        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        if self.use_i2v_clip:
            self.i2v_clip = instantiate_from_config(i2v_clip_config) if i2v_clip_config is not None else None

        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.latent_input = latent_input
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

        if self.use_pd and (args.mode == 'pretrain' or args.mode == 'finetune'):
            from sat.training.model_io import load_checkpoint
            import copy
            print("############# load teacher model")
            self.teacher_model = copy.deepcopy(self.model)
            load_checkpoint(self.teacher_model, args, prefix='model.')
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

    def disable_untrainable_params(self):
        total_trainable = 0
        if self.lora_train:
            for n, p in self.named_parameters():
                if p.requires_grad == False:
                    continue
                if 'lora_layer' not in n:
                    p.lr_scale = 0
                else:
                    total_trainable += p.numel()
        else:
            for n, p in self.named_parameters():
                if p.requires_grad == False:
                    continue
                flag = False
                for prefix in self.untrainable_prefixs:
                    if n.startswith(prefix) or prefix == "all":
                        flag = True
                        break

                lora_prefix = ['matrix_A', 'matrix_B']
                for prefix in lora_prefix:
                    if prefix in n:
                        flag = False
                        break

                if flag:
                    p.requires_grad_(False)
                else:
                    total_trainable += p.numel()

        print_rank0("***** Total trainable parameters: " + str(total_trainable) + " *****")

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    def _init_first_stage(self, config):
        model = instantiate_from_config(config)
        if not 'wan_vae' in config['target']:
            model = model.eval()
        model.train = disabled_train
        if not 'wan_vae' in config['target']:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.model.parameters():
                param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key].to(self.dtype)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        for n in range(n_rounds):
            z_now = z[n * n_samples : (n + 1) * n_samples]
            recons = self.first_stage_model.decode(z_now) # b c t h w
            all_out.append(recons)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, batch, force_encode=False):
        if not force_encode and self.latent_input:
            return x * self.scale_factor # already encoded # bcthw

        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []

        for n in range(n_rounds):
            x_now = x[n * n_samples: (n + 1) * n_samples]
            latents = self.first_stage_model.encode(x_now) # b c t h w
            all_out.append(latents)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z # b c t h w
        torch.distributed.broadcast(z, src=mpu.get_data_broadcast_src_rank(), group=mpu.get_data_broadcast_group())
        return z

    def forward(self, x, batch):
        if self.use_pd:
            loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch, self.teacher_model, self.sampler)
        else:
            loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"diffusion loss": loss_mean}
        return loss_mean, loss_dict

    def add_noise_to_first_frame(self, image):
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image = image + image_noise
        return image

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch) # btchw for mp4, bcthw for latent
        dynamic_subject = True if isinstance(batch['subject_images'], list) else False
        # x = x.view(-1, *x.shape[2:])
        if self.noised_image_input:
            if self.i2v_encode_video:
                assert not self.latent_input, 'latent_input should be False when i2v_encode_video is True'
                ori_image = x[:, 0:1]
                image = self.add_noise_to_first_frame(ori_image).to(torch.bfloat16)
                image = torch.concat([image, torch.zeros_like(x[:, 1:])], dim=1)
                image = rearrange(image, 'b t c h w -> b c t h w').contiguous()
                image = self.encode_first_stage(image, batch, force_encode=True)
                image = image.permute(0, 2, 1, 3, 4).contiguous() # BCTHW -> BTCHW
                for idx in range(image.shape[0]):
                    if random.random() < self.image_cond_dropout:
                        image[idx] = torch.zeros_like(image[idx])
                batch["concat_images"] = image
            else:
                if self.latent_input:
                    image, x = torch.split(x, [1, x.shape[2]-1], dim=2) # bcthw
                else:
                    ori_image = x[:, 0:1]
                    image = self.add_noise_to_first_frame(ori_image)
                    image = rearrange(image, 'b t c h w -> b c t h w').contiguous()
                image = self.encode_first_stage(image, batch)

        x = rearrange(x, 'b t c h w -> b c t h w').contiguous()
        x = self.encode_first_stage(x, batch)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # b t c h w

        if 'subject_images' in batch and batch['subject_images'] is not None:
            ref_images = batch['subject_images'] # b t c h w
            refs = []
            ref_subjects = ref_images[0].unsqueeze(0).unsqueeze(0) if not dynamic_subject else ref_images
            for ref in ref_subjects:
                ref = ref.to(self.device)
                # ref = self.add_noise_to_first_frame(ref)
                ref = rearrange(ref, 'b t c h w -> b c t h w').contiguous()
                ref = self.encode_first_stage(ref, batch)
                ref = ref.permute(0, 2, 1, 3, 4).contiguous() # b t c h w

                # if random.random() < self.subject_cond_dropout:
                #     ref = torch.zeros_like(ref)
                refs.append(ref.to(self.dtype))
            batch['concat_subjects'] = torch.cat(refs, dim=1) if not dynamic_subject else refs
        # print(x.shape, batch['concat_subjects'].shape)
        if self.noised_image_input:
            if not self.i2v_encode_video:
                image = image.permute(0, 2, 1, 3, 4).contiguous() # bcthw -> btchw
                if self.noised_image_all_concat:
                    image = image.repeat(1, x.shape[1], 1, 1, 1) #TODO: 改回去
                else:
                    image = torch.concat([image, torch.zeros_like(x[:, 1:])], dim=1)
                for idx in range(image.shape[0]):
                    if random.random() < self.image_cond_dropout:
                        image[idx] = torch.zeros_like(image[idx])
                batch["concat_images"] = image
        
        if self.use_i2v_clip:
            assert not self.latent_input
            image_clip_features = self.i2v_clip.visual(ori_image.permute(0, 2, 1, 3, 4)) # btchw -> bcthw
            batch["image_clip_features"] = image_clip_features

        # x = x.view(b, t, *x.shape[1:])
        # batch["global_step"] = self.global_step

        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            uc: Union[Dict, None] = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            prefix = None,
            concat_images = None,
            ofs = None,
            fps = None,
            **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)
        #debug !!!!!!!
        # breakpoint()

        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)
        if hasattr(self.loss_fn, "block_scale") and self.loss_fn.block_scale is not None:
            randn = self.loss_fn.get_blk_noise(randn)

        if prefix is not None:
            randn = torch.cat([prefix, randn[:, prefix.shape[1]:]], dim=1)

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        sp_size = mpu.get_sequence_parallel_world_size()
        if mp_size > 1 or sp_size > 1:
            torch.distributed.broadcast(randn, src=mpu.get_data_broadcast_src_rank(), group=mpu.get_data_broadcast_group())

        chunk_dim = None
        if sp_size > 1:
            sp_rank = mpu.get_sequence_parallel_rank()
            h, w = randn.shape[-2:]
            assert (h / sp_size) % 2 == 0 or (w / sp_size) % 2 == 0
            if (h / sp_size) % 2 == 0:
                chunk_dim = 3
            else:
                chunk_dim = 4
            randn = torch.chunk(randn, sp_size, dim=chunk_dim)[sp_rank]
            if "concat" in cond.keys():
                uc['concat'] = torch.chunk(uc['concat'], sp_size, dim=chunk_dim)[sp_rank]
                cond['concat'] = torch.chunk(cond['concat'], sp_size, dim=chunk_dim)[sp_rank]
            if 'concat_subjects' in cond.keys():
                cond['concat_subjects'] = torch.chunk(cond['concat_subjects'], sp_size, dim=chunk_dim)[sp_rank]
                uc['concat_subjects'] = torch.chunk(uc['concat_subjects'], sp_size, dim=chunk_dim)[sp_rank]

        if self.use_pd == True:
            scale = 1.0
            scale_emb = timestep_embedding(randn.new_ones([batch_size]) * self.sampler.guider.scale, self.model.diffusion_model.cfg_embed_dim).to(self.dtype)
        else:
            scale = None
            scale_emb = None

        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model, input, sigma, c, concat_images=concat_images, chunk_dim=chunk_dim, **addtional_model_inputs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc, scale=scale, scale_emb=scale_emb, ofs=ofs, fps=fps)
        samples = samples.to(self.dtype)
        if sp_size > 1:
            sp_rank = mpu.get_sequence_parallel_rank()
            gather_list = [torch.zeros_like(samples) for _ in range(sp_size)] if sp_rank == 0 else None
            torch.distributed.gather(samples, dst=mpu.get_sequence_parallel_src_rank(), gather_list=gather_list, group=mpu.get_sequence_parallel_group())
            if sp_rank == 0:
                samples = torch.concat(gather_list, dim=chunk_dim)

        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                    (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_video(
            self,
            batch: Dict,
            N: int = 8,
            sample: bool = True,
            ucg_keys: List[str] = None,
            only_log_video_latents = False,
            **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        batch_uc = {}
        batch_uc['txt'] = np.repeat([''],repeats=len(batch['txt'])).reshape(len(batch['txt'])).tolist()
        for key in batch.keys():
            if key not in batch_uc:
                if isinstance(batch[key], torch.Tensor):
                    batch_uc[key] = torch.clone(batch[key])
                else:
                    batch_uc[key] = batch[key]
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=[],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N] # b c t h w
        if not self.latent_input:
            log["inputs"] = x.to(torch.float32)

        # x = x.view(-1, *x.shape[2:])
        x = rearrange(x, 'b t c h w -> b c t h w').contiguous()
        z = self.encode_first_stage(x, batch)
        if not only_log_video_latents:
            log["reconstructions"] = self.decode_first_stage(z).to(torch.float32)
            log["reconstructions"] = log["reconstructions"].permute(0, 2, 1, 3, 4).contiguous()
        z = z.permute(0, 2, 1, 3, 4).contiguous() # b t c h w

        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample and self.noised_image_input:
            if self.latent_input:
                image, z = torch.split(z, [1, z.shape[1] - 1], dim=1)
            else:
                image = x[:, :, 0:1]
                image = self.add_noise_to_first_frame(image)
                image = self.encode_first_stage(image, batch)
                image = image.permute(0, 2, 1, 3, 4).contiguous()
            if self.noised_image_all_concat:
                image = image.repeat(1, z.shape[1], 1, 1, 1) # TODO: 改回去
            else:
                image = torch.concat([image, torch.zeros_like(z[:, 1:])], dim=1)
            c["concat"] = image
            uc["concat"] = image
            samples = self.sample(
                c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            ) # b t c h w
            b, t = samples.shape[:2]
            # samples = samples.view(-1, *samples.shape[2:])
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            if only_log_video_latents:
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                samples = self.decode_first_stage(samples).to(torch.float32)
                # samples = samples.view(b, t, *samples.shape[1:])
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                log["samples"] = samples
        elif sample:
            if self.subjects_input:
                ref_images = batch['subject_images'] # b t c h w
                dynamic_subject = True if isinstance(ref_images, list) else False
                ref_subjects = ref_images[0].unsqueeze(0).unsqueeze(0) if not dynamic_subject else ref_images
                refs = []
                for ref in ref_subjects:
                    ref = rearrange(ref, 'b t c h w -> b c t h w').contiguous()
                    ref = self.encode_first_stage(ref, batch)
                    ref = ref.permute(0, 2, 1, 3, 4).contiguous() # b t c h w
                    refs.append(ref)

                c['concat_subjects'] = torch.cat(refs, dim=1) if isinstance(refs, torch.Tensor) else refs
                uc['concat_subjects'] = torch.cat(refs, dim=1) if isinstance(refs, torch.Tensor) else refs

            samples = self.sample(
                c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
            ) # b t c h w
            b, t = samples.shape[:2]
            # samples = samples.view(-1, *samples.shape[2:])
            samples = samples.permute(0, 2, 1, 3, 4).contiguous() # b c t h w
            if only_log_video_latents:
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                samples = self.decode_first_stage(samples).to(torch.float32)
                # samples = samples.view(b, t, *samples.shape[1:])
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                log["samples"] = samples
        return log
    

    @classmethod
    def from_pretrained_base(cls, args=None, *, prefix='', build_only=False, overwrite_args={}, **kwargs):
        '''Load a pretrained checkpoint of the current model.
            Args:
                name: The identifier of the pretrained model.
                args: NameSpace. will add the loaded args into it. None will create a new model-only one with defaults.
                path: the parent folder of existing `name` model. Default: SAT_HOME.
                url: the url of the model. Default: SAT_URL.
                prefix: the prefix of the checkpoint. Default: ''.
            Returns:
                model: the loaded model.
                args: the loaded args.
        '''

        # create a new args if not provided
        if args is None:
            args = cls.get_args()
        args = overwrite_args_by_dict(args, overwrite_args=overwrite_args)
        model = get_model(args, cls, **kwargs)
        if not build_only:
            load_checkpoint(model, args, prefix=prefix)
        return model, deepcopy(args)
    
    @classmethod
    def from_pretrained(cls, args=None, *, prefix='', build_only=False, use_node_group=True, overwrite_args={}, **kwargs):
        if build_only or 'model_parallel_size' not in overwrite_args:
            return cls.from_pretrained_base(args=args, prefix=prefix, build_only=build_only, overwrite_args=overwrite_args, **kwargs)
        else:
            new_model_parallel_size = overwrite_args['model_parallel_size']
            if new_model_parallel_size != 1 or new_model_parallel_size == 1 and args.model_parallel_size == 1:
                model, model_args = cls.from_pretrained_base(args=args, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                local_rank = get_node_rank() if use_node_group else get_model_parallel_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size % new_model_parallel_size == 0, "world size should be a multiplier of new model_parallel_size."
                destroy_model_parallel()
                initialize_model_parallel(1)
                if local_rank == 0:
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args.pop('model_parallel_size')
                    model_full, args_ = cls.from_pretrained_base(args=args, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                    if args_.model_parallel_size != 1:
                        raise Exception("We do not support overwriting model_parallel_size when original model_parallel_size != 1. Try merging the model using `from_pretrained(xxx,overwrite_args={'model_parallel_size':1})` first if you still want to change model_parallel_size!")
                if hasattr(args, 'mode') and args.mode == 'inference': # For multi-node inference, we should prevent rank 0 eagerly printing some info.
                    torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(new_model_parallel_size)
                if local_rank == 0:
                    mp_split_model_rank0(model, model_full, use_node_group=use_node_group)
                    del model_full
                else:
                    mp_split_model_receive(model, use_node_group=use_node_group)
                reset_random_seed(6)
            else:
                overwrite_args.pop('model_parallel_size')
                model, model_args = cls.from_pretrained_base(args=args, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size == model_args.model_parallel_size, "world size should be equal to model_parallel_size."
                destroy_model_parallel()
                initialize_model_parallel(1)
                if rank == 0:
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args['model_parallel_size'] = 1
                    overwrite_args['model_config'] = args.model_config
                    overwrite_args['model_config']['network_config']['params']['transformer_args']['model_parallel_size'] = 1
                    model_full, args_ = cls.from_pretrained_base(args=args, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(model_args.model_parallel_size)
                if rank == 0:
                    mp_merge_model_rank0(model, model_full)
                    model, model_args = model_full, args_
                else:
                    mp_merge_model_send(model)
                    model_args.model_parallel_size = 1
                destroy_model_parallel()
                initialize_model_parallel(1)
            return model, model_args
