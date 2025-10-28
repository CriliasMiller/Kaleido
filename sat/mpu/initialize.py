# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Model and data parallel groups."""

import torch

from .utils import ensure_divisibility
from sat.helpers import print_rank0


# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_DATA_BROADCAST_GROUP = None

_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

_SEQUENCE_DATA_PARALLEL_GROUP = None
_SEQUENCE_DATA_PARALLEL_WORLD_SIZE = None
_SEQUENCE_DATA_PARALLEL_RANK = None

_SEQUENCE_PARALLEL_GQA_GROUP = None
_SEQUENCE_PARALLEL_GQA_WORLD_SIZE = None
_SEQUENCE_PARALLEL_GQA_INDEX = None

# Node group that current rank belongs to.
_NODE_GROUP = None

# FSDP2 device mesh for multi-dimensional parallelism
_FSDP2_DEVICE_MESH = None
_FSDP2_ENABLED = False

import os

def get_gqa_group(cur_rank, num_multi_query_heads, group_ranks):
    group_size = len(group_ranks)
    gqa_len = group_size // num_multi_query_heads

    rank_indx = group_ranks.index(cur_rank)
    if (rank_indx + 1) % gqa_len == 0:
        gqa_indx = (rank_indx + 1) // gqa_len
    else:
        gqa_indx = (rank_indx + 1) // gqa_len + 1
    start = (gqa_indx-1) * gqa_len
    end = gqa_indx * gqa_len
    return start, end, gqa_indx - 1

def initialize_model_parallel(model_parallel_size_, sequence_parallel_size_: int = 1, num_multi_query_heads_: int = 0):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    print_rank0('> initializing model parallel with size {}, sequence parallel with size {}'.format(
        model_parallel_size_, sequence_parallel_size_))

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = min(model_parallel_size_, world_size)
    sequence_parallel_size = max(sequence_parallel_size_, 1)
    assert world_size >= model_parallel_size * sequence_parallel_size, f"{world_size=}, {model_parallel_size=}, {sequence_parallel_size=}"
    ensure_divisibility(world_size, model_parallel_size * sequence_parallel_size)
    data_parallel_size = world_size // (model_parallel_size * sequence_parallel_size)
    num_sequence_parallel_groups = world_size // sequence_parallel_size
    num_sequence_data_parallel_groups: int = world_size // sequence_parallel_size // data_parallel_size

    rank = torch.distributed.get_rank()

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    for i in range(model_parallel_size * sequence_parallel_size):
        ranks = range(i, world_size, model_parallel_size * sequence_parallel_size)
        group = torch.distributed.new_group(ranks, group_desc="data_parallel_group")
        if i == (rank % (model_parallel_size * sequence_parallel_size)):
            _DATA_PARALLEL_GROUP = group

    global _SEQUENCE_PARALLEL_GROUP
    assert _SEQUENCE_PARALLEL_GROUP is None, \
        'sequence parallel group is already initialized'

    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    global _DATA_BROADCAST_GROUP
    assert _DATA_BROADCAST_GROUP is None, \
        'data broadcast group is already initialized'


    for i in range(world_size // model_parallel_size):
        mp_ranks = range(i * model_parallel_size, (i+1) * model_parallel_size)
        mp_group = torch.distributed.new_group(mp_ranks, group_desc='model_parallel_group')
        if rank in mp_ranks:
            _MODEL_PARALLEL_GROUP = mp_group

    for dp in range(data_parallel_size):
        for mp in range(model_parallel_size):
            rank_0 = dp * sequence_parallel_size * model_parallel_size + mp
            ranks = range(rank_0, (dp + 1) * sequence_parallel_size * model_parallel_size, model_parallel_size)

            group = torch.distributed.new_group(ranks, group_desc="sequence_parallel_group")
            if rank in ranks:
                _SEQUENCE_PARALLEL_GROUP = group

        data_broadcast_ranks = range(dp * sequence_parallel_size * model_parallel_size, (dp + 1) * sequence_parallel_size * model_parallel_size)
        db_group = torch.distributed.new_group(data_broadcast_ranks, group_desc='data_broadcast_group')
        if rank in data_broadcast_ranks:
            _DATA_BROADCAST_GROUP = db_group

    # In backward, dp_group and seq_group params grads need to be allreduced
    global _SEQUENCE_DATA_PARALLEL_GROUP
    sequence_data_parallel_size: int = sequence_parallel_size * data_parallel_size
    if sequence_parallel_size > 1:
        for i in range(num_sequence_data_parallel_groups): # mp only. no further parallism like pp supported.
            ranks = range(i, world_size, model_parallel_size)
            group = torch.distributed.new_group(ranks, group_desc='sequence_data_parallel_group')
            if rank in ranks:
                _SEQUENCE_DATA_PARALLEL_GROUP = group
    else:
        _SEQUENCE_DATA_PARALLEL_GROUP = _DATA_PARALLEL_GROUP

    guess_local_world_size = world_size if world_size < 8 else 8
    local_world_size = os.environ.get('LOCAL_WORLD_SIZE', None)
    if local_world_size is None:
        local_world_size = guess_local_world_size
        print_rank0(f"You didn't pass in LOCAL_WORLD_SIZE environment variable. We use the guessed LOCAL_WORLD_SIZE={guess_local_world_size}. If this is wrong, please pass the LOCAL_WORLD_SIZE manually.")
    local_world_size = int(local_world_size)
    # Build the node groups.
    global _NODE_GROUP
    assert _NODE_GROUP is None, \
        'node group is already initialized'
    for i in range(world_size // local_world_size):
        ranks = range(i * local_world_size,
                    (i + 1) * local_world_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank // local_world_size):
            _NODE_GROUP = group


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True

def sequence_parallel_is_initialized():
    """Check if sequence and data parallel groups are initialized."""
    if _SEQUENCE_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP

def get_sequence_data_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_DATA_PARALLEL_GROUP is not None, \
        'sequence data parallel group is not initialized'
    return _SEQUENCE_DATA_PARALLEL_GROUP

def get_sequence_data_parallel_rank():
    return torch.distributed.get_rank(group=get_sequence_data_parallel_group())

def get_sequence_data_parallel_src_rank():
    return torch.distributed.get_process_group_ranks(get_sequence_data_parallel_group())[0]

def get_sequence_data_parallel_world_size():
    return torch.distributed.get_world_size(group=get_sequence_data_parallel_group())

def get_node_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _NODE_GROUP is not None, \
        'node group is not initialized, please pass LOCAL_WORLD_SIZE environment variable.'
    return _NODE_GROUP


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def get_node_world_size():
    """Return world size for the node group."""
    return torch.distributed.get_world_size(group=get_node_group())


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())

def get_sequence_parallel_rank():
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_sequence_parallel_group())

def get_node_rank():
    """Return my rank for the node group."""
    return torch.distributed.get_rank(group=get_node_group())

def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    return torch.distributed.get_process_group_ranks(get_model_parallel_group())[0]

def get_sequence_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    return torch.distributed.get_process_group_ranks(get_sequence_parallel_group())[0]


def get_node_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the node group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_node_world_size()
    return (global_rank // local_world_size) * local_world_size

def get_data_broadcast_rank():
    return torch.distributed.get_rank(group=get_data_broadcast_group())

def get_data_broadcast_src_rank():
    return torch.distributed.get_process_group_ranks(get_data_broadcast_group())[0]

def get_data_broadcast_group():
    assert _DATA_BROADCAST_GROUP is not None, \
        'data broadcast group is not initialized'
    return _DATA_BROADCAST_GROUP

def get_data_broadcast_world_size():
    return torch.distributed.get_world_size(group=get_data_broadcast_group())

def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GROUP is not None, \
        'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())

def get_sequence_parallel_gqa_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GQA_GROUP is not None, \
        'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GQA_GROUP

def get_sequence_parallel_gqa_index():
    """Get the sequence parallel gqa index the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GQA_INDEX is not None, \
        'sequence parallel gqa index is not initialized'
    return _SEQUENCE_PARALLEL_GQA_INDEX

def get_sequence_parallel_gqa_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_GQA_WORLD_SIZE
    if _SEQUENCE_PARALLEL_GQA_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_GQA_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_gqa_group())

def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _NODE_GROUP
    _NODE_GROUP = None

    ### fsdp2 setting
    global _FSDP2_DEVICE_MESH
    _FSDP2_DEVICE_MESH = None
    global _FSDP2_ENABLED
    _FSDP2_ENABLED = False

def set_fsdp2_device_mesh(device_mesh):
    """Set the FSDP2 device mesh for multi-dimensional parallelism."""
    global _FSDP2_DEVICE_MESH
    global _FSDP2_ENABLED
    _FSDP2_DEVICE_MESH = device_mesh
    _FSDP2_ENABLED = True


def get_fsdp2_device_mesh():
    """Get the FSDP2 device mesh."""
    assert _FSDP2_DEVICE_MESH is not None, \
        'FSDP2 device mesh is not initialized'
    return _FSDP2_DEVICE_MESH


def is_fsdp2_enabled():
    """Check if FSDP2 is enabled."""
    return _FSDP2_ENABLED


def get_data_parallel_group_for_fsdp2():
    """Get data parallel group for FSDP2.
    
    When using FSDP2 with multi-dimensional parallelism,
    the data parallel group might be different from the
    traditional DP group.
    """
    if _FSDP2_ENABLED:
        # FSDP2 handles its own data parallelism through device mesh
        # Return the traditional DP group for compatibility
        return _DATA_PARALLEL_GROUP
    else:
        return get_data_parallel_group()
