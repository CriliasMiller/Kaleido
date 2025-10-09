import argparse
import os
import torch
import json
import random
import numpy as np
import warnings
import omegaconf
from omegaconf import OmegaConf

import logging
from sat.helpers import print_all, print_rank0
from sat import mpu
from sat.arguments import set_random_seed
from sat.arguments import add_training_args, add_evaluation_args, add_data_args, add_fsdp2_config_args
import torch.distributed

def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--base', type=str, nargs='*',
                       help='config for input and saving')
    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='size of the model parallel. only use if you are an expert.')
    group.add_argument('--sequence-parallel-size', type=int, default=1,
                       help='size of the sequence parallel. only use if you are an expert.')
    group.add_argument('--data-parallel-size', type=int, default=None,
                       help='Size of data parallelism (auto-calculated if None)')

    group.add_argument('--force-pretrain', action='store_true')
    group.add_argument('--device', type=int, default=-1)
    group.add_argument('--debug', action='store_true')
    group.add_argument('--log-image', type=bool, default=True)
    group.add_argument('--model-type', type=str, default="dit")
    group.add_argument('--num-multi-query-heads', type=int, default=0,
                       help='use multi-query attention, num of kv groups. 0 means multi-head attention.')


    # group.add_argument('--checkpoint-activations', action='store_true',
    #                    help='checkpoint activation to allow for training '
    #                         'with larger models and sequences. become slow (< 1.5x), save CUDA memory.')
    # Inessential
    # group.add_argument('--checkpoint-num-layers', type=int, default=1,
    #                    help='chunk size (number of layers) for checkpointing. ')
    # group.add_argument('--checkpoint-skip-layers', type=int, default=0,
    #                    help='skip the last N layers for checkpointing.')

    return parser

def add_sampling_config_args(parser):
    """Sampling configurations"""

    group = parser.add_argument_group('sampling', 'Sampling Configurations')
    group.add_argument('--output-dir', type=str, default='samples')
    group.add_argument('--input-dir', type=str, default=None)
    group.add_argument('--input-type', type=str, default='cli')
    group.add_argument('--input-file', type=str, default='input.txt')
    group.add_argument('--sampling-image-size', type=int, default=1024)
    group.add_argument('--final-size', type=int, default=2048)
    group.add_argument('--sdedit', action='store_true')
    group.add_argument('--grid-num-rows', type=int, default=1)
    group.add_argument('--force-inference', action='store_true')
    group.add_argument('--lcm_steps',  type=int, default=None)
    group.add_argument('--sampling-num-frames', type=int, default=32)
    group.add_argument('--sampling-fps', type=int, default=8)
    group.add_argument('--only-save-latents', action='store_true')
    group.add_argument('--only-log-video-latents', action='store_true')
    group.add_argument('--latent-channels', type=int, default=32)
    group.add_argument('--vae-compress-size', type=list, default=None)
    group.add_argument('--image2video', action='store_true')
    group.add_argument('--horizontal-only', action='store_true')
    group.add_argument('--subject2video', action='store_true')
    group.add_argument('--s2v_concat', action='store_true')
    group.add_argument('--image_condition_zero', action='store_true')
    group.add_argument('--subject_dynamic', action='store_true')
    group.add_argument('--new_straetgy', action='store_true')
    group.add_argument('--final_concat', action='store_true')
    group.add_argument('--sample_neg_prompt', type=str, default=None)
    group.add_argument('--load_base_model', action='store_true')
    group.add_argument('--base_model_path', type=str, default=None)
    group.add_argument('--fsdp2_ckpt', type=bool, default=False)

    return parser

def get_args(args_list=None, parser=None):
    """Parse all the args."""
    if parser is None:
        parser = argparse.ArgumentParser(description='sat')
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    parser = add_model_config_args(parser)
    parser = add_sampling_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_data_args(parser)
    parser = add_fsdp2_config_args(parser)

    # Include DeepSpeed configuration arguments
    import deepspeed
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args(args_list)
    args = process_config_to_args(args)

    if not args.train_data:
        print_rank0('No training data specified', level='WARNING')

    assert (args.train_iters is None) or (args.epochs is None), 'only one of train_iters and epochs should be set.'
    if args.train_iters is None and args.epochs is None:
        args.train_iters = 10000 # default 10k iters
        print_rank0('No train_iters (recommended) or epochs specified, use default 10k iters.', level='WARNING')
    
    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    if args.local_rank is None:
        args.local_rank = int(os.getenv("LOCAL_RANK", '0')) # torchrun

    if args.device == -1: # not set manually
        if torch.cuda.device_count() == 0:
            args.device = 'cpu'
        elif args.local_rank is not None:
            args.device = args.local_rank
        else:
            args.device = args.rank % torch.cuda.device_count()

    # local rank should be consistent with device in DeepSpeed
    if args.local_rank != args.device and args.mode != 'inference':
        raise ValueError(
            'LOCAL_RANK (default 0) and args.device inconsistent. '
            'This can only happens in inference mode. '
            'Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. '
            )
    
    # args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print_rank0('using world size: {}'.format(args.world_size))
    # if args.vocab_size > 0:
    #     _adjust_vocab_size(args)
    
    if args.train_data_weights is not None:
        assert len(args.train_data_weights) == len(args.train_data)

    # Calculate data parallel size if not specified
    if args.data_parallel_size is None:
        args.data_parallel_size = args.world_size // (args.model_parallel_size * args.sequence_parallel_size)
    
    if args.mode != 'inference': # training with deepspeed
        if args.fsdp2:
            args.deepspeed = False
            # Setup FSDP2 configuration
            args.fsdp2_config = {
                'max_grad_norm': args.fsdp2_optimizer_max_grad_norm,
                'mixed_precision': args.fsdp2_mixed_precision,
                'param_dtype': args.fsdp2_param_dtype,
                'reduce_dtype': args.fsdp2_reduce_dtype,
                'auto_wrap': args.fsdp2_auto_wrap,
                'reshard_after_forward': args.fsdp2_reshard_after_forward,
                'offload_params': args.fsdp2_offload_params,
                'min_params_to_wrap': args.fsdp2_min_params_to_wrap,
                'wrap_patterns': args.fsdp2_wrap_patterns,
                'gradient_checkpointing': args.fsdp2_gradient_checkpointing,
                'cpu_offload_pin_memory': args.fsdp2_cpu_offload_pin_memory,
                'sharding_strategy': args.fsdp2_sharding_strategy,
                'backward_prefetch': args.fsdp2_backward_prefetch,
                'forward_prefetch': args.fsdp2_forward_prefetch,
                'sync_module_states': args.fsdp2_sync_module_states,
                'use_orig_params': args.fsdp2_use_orig_params,
                'optimizer': args.fsdp2_optimizer,
                'optimizer_params': {
                    'lr': args.fsdp2_optimizer_lr,
                    'weight_decay': args.fsdp2_optimizer_weight_decay,
                    'eps': args.fsdp2_optimizer_eps,
                    'betas': tuple(args.fsdp2_optimizer_betas)
                }
            }
        else:
            args.deepspeed = True
            if args.deepspeed_config is None: # not specified
                deepspeed_config_path = os.path.join(os.path.dirname(__file__), 'training', f'deepspeed_zero{args.zero_stage}.json')
                with open(deepspeed_config_path) as file:
                    args.deepspeed_config = json.load(file)
                override_deepspeed_config = True
            else:
                override_deepspeed_config = False

    assert not (args.fp16 and args.bf16), 'cannot specify both fp16 and bf16.'
    assert not (args.fsdp2 and args.deepspeed), 'cannot specify both fsdp2 and deepspeed.'
    if args.fsdp2:
        # FSDP2 precision handling
        if args.fsdp2_mixed_precision:
            if not args.fp16 and not args.bf16:
                print_rank0('Automatically set bf16=True for FSDP2 mixed precision.')
                args.bf16 = True
    elif args.zero_stage > 0 and not args.fp16 and not args.bf16:
        print_rank0('Automatically set fp16=True to use ZeRO.')     
        args.fp16 = True
        args.bf16 = False
        
    if args.deepspeed:
        if args.checkpoint_activations:
            args.deepspeed_activation_checkpointing = True
        else:
            args.deepspeed_activation_checkpointing = False
        if args.deepspeed_config is not None:
            deepspeed_config = args.deepspeed_config
            # with open(args.deepspeed_config) as file:
            #     deepspeed_config = json.load(file)
            
        if override_deepspeed_config: # not specify deepspeed_config, use args
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            elif args.bf16:
                deepspeed_config["bf16"]["enabled"] = True
                deepspeed_config["fp16"]["enabled"] = False
            else:
                deepspeed_config["fp16"]["enabled"] = False
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            optimizer_params_config["lr"] = args.lr
            optimizer_params_config["weight_decay"] = args.weight_decay
        else: # override args with values in deepspeed_config
            if args.rank == 0:
                print_rank0('Will override arguments with manually specified deepspeed_config!')
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                args.fp16 = False
            if "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
                args.bf16 = True
            else:
                args.bf16 = False
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            else:
                args.gradient_accumulation_steps = None
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                args.lr = optimizer_params_config.get("lr", args.lr)
                args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
        args.deepspeed_config = deepspeed_config
    elif args.fsdp2:
        # FSDP2 doesn't need separate config file, handled via args
        override_deepspeed_config = False
        deepspeed_config = None

    # if args.sandwich_ln: # removed in v0.3
    #     args.layernorm_order = 'sandwich'

    # initialize distributed and random seed because it always seems to be necessary.
    initialize_distributed(args)
    args.seed = args.seed + mpu.get_data_parallel_rank()
    set_random_seed(args.seed)
    return args

def initialize_distributed(args):
    """Initialize torch.distributed."""
    if torch.distributed.is_initialized():
        if mpu.model_parallel_is_initialized():
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError('model_parallel_size is inconsistent with prior configuration.'
                                 'We currently do not support changing model_parallel_size.')
            return False
        else:
            if args.model_parallel_size > 1:
                warnings.warn('model_parallel_size > 1 but torch.distributed is not initialized via SAT.'
                            'Please carefully make sure the correctness on your own.')
            mpu.initialize_model_parallel(args.model_parallel_size, args.sequence_parallel_size, args.num_multi_query_heads)
        return True
    # the automatic assignment of devices has been moved to arguments.py
    if args.device == 'cpu':
        pass
    else:
        torch.cuda.set_device(f"cuda:{args.device}")
    # Call the init process
    init_method = 'tcp://'
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost')
    
    if args.world_size == 1:
        from sat.helpers import get_free_port
        default_master_port = str(get_free_port())
    else:
        default_master_port = '6000'
    args.master_port = os.getenv('MASTER_PORT', default_master_port)
    init_method += args.master_ip + ':' + args.master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method, device_id=torch.device(f"cuda:{args.local_rank}"))

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size, args.sequence_parallel_size, args.num_multi_query_heads)

    # Set vae context parallel group equal to model parallel group
    from sgm.util import set_context_parallel_group, initialize_context_parallel
    '''
    if args.model_parallel_size <= 2:
        set_context_parallel_group(args.model_parallel_size, mpu.get_model_parallel_group())
    else:
        initialize_context_parallel(2)
    '''
    # use tp group for vae cp
    #set_context_parallel_group(args.model_parallel_size, mpu.get_model_parallel_group())
    initialize_context_parallel(1)

    # mpu.initialize_model_parallel(1)
    # Optional DeepSpeed Activation Checkpointing Features
    if args.deepspeed: 
        import deepspeed
        deepspeed.init_distributed(
            dist_backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank, init_method=init_method)
        # # It seems that it has no negative influence to configure it even without using checkpointing.  
        # deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    else:
        # in model-only mode, we don't want to init deepspeed, but we still need to init the rng tracker for model_parallel, just because we save the seed by default when dropout. 
        try:
            import deepspeed
            from deepspeed.runtime.activation_checkpointing.checkpointing import _CUDA_RNG_STATE_TRACKER, _MODEL_PARALLEL_RNG_TRACKER_NAME
            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1) # default seed 1
        except Exception as e:
            from sat.helpers import print_rank0
            print_rank0(str(e), level="DEBUG")

    return True

def process_config_to_args(args):
    """Fetch args from only --base"""

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    config = OmegaConf.merge(*configs)
    
    args_config = config.pop("args", OmegaConf.create())
    for key in args_config:
        if isinstance(args_config[key], omegaconf.DictConfig) or isinstance(args_config[key], omegaconf.ListConfig):
            arg = OmegaConf.to_object(args_config[key])
        else:
            arg = args_config[key]
        if hasattr(args, key):
            setattr(args, key, arg)

    if "model" in config:
        model_config = config.pop("model", OmegaConf.create())
        args.model_config = model_config
    if "deepspeed" in config:
        deepspeed_config = config.pop("deepspeed", OmegaConf.create())
        args.deepspeed_config = OmegaConf.to_object(deepspeed_config)
    if "data" in config:
        data_config = config.pop("data", OmegaConf.create())
        args.data_config = data_config
    
    if "fsdp2_config" in args and args.fsdp2:
        fsdp2_cfg = args.fsdp2_config
        if "max_grad_norm" in fsdp2_cfg:
            args.fsdp2_optimizer_max_grad_norm = fsdp2_cfg["max_grad_norm"]
        if "mixed_precision" in fsdp2_cfg:
            args.fsdp2_mixed_precision = fsdp2_cfg["mixed_precision"]
        if "param_dtype" in fsdp2_cfg:
            args.fsdp2_param_dtype = fsdp2_cfg["param_dtype"]
        if "reduce_dtype" in fsdp2_cfg:
            args.fsdp2_reduce_dtype = fsdp2_cfg["reduce_dtype"]
        if "auto_wrap" in fsdp2_cfg:
            args.fsdp2_auto_wrap = fsdp2_cfg["auto_wrap"]
        if "wrap_patterns" in fsdp2_cfg:
            args.fsdp2_wrap_patterns = fsdp2_cfg["wrap_patterns"]
        if "min_params_to_wrap" in fsdp2_cfg:
            args.fsdp2_min_params_to_wrap = fsdp2_cfg["min_params_to_wrap"]
        if "reshard_after_forward" in fsdp2_cfg:
            args.fsdp2_reshard_after_forward = fsdp2_cfg["reshard_after_forward"]
        if "offload_params" in fsdp2_cfg:
            args.fsdp2_offload_params = fsdp2_cfg["offload_params"]
        if "sharding_strategy" in fsdp2_cfg:
            args.fsdp2_sharding_strategy = fsdp2_cfg["sharding_strategy"]
        if "optimizer" in fsdp2_cfg:
            args.fsdp2_optimizer = fsdp2_cfg["optimizer"]
        if "optimizer_params" in fsdp2_cfg:
            opt = fsdp2_cfg["optimizer_params"]
            if "lr" in opt: args.fsdp2_optimizer_lr = opt["lr"]
            if "weight_decay" in opt: args.fsdp2_optimizer_weight_decay = opt["weight_decay"]
            if "eps" in opt: args.fsdp2_optimizer_eps = opt["eps"]
            if "betas" in opt: args.fsdp2_optimizer_betas = opt["betas"]
            if "state_lr" in opt: args.lr = opt["state_lr"]

    return args