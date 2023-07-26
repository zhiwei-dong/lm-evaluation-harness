import os
import torch
import transformers


def get_max_memory_per_gpu_dict(dtype, model_name, model=None):
    """ try to generate the memory map based on what we know about the model and the available hardware """

    # figure out the memory map - the minimum per gpu required to load the model
    n_gpus = torch.cuda.device_count()

    if model_name == "bigscience/bloom" and n_gpus == 8 and torch.cuda.get_device_properties(0).total_memory > 79*2**30:
        # hand crafted optimized memory map for 8x80 setup over BLOOM
        # this works with bs=40
        if dtype != torch.int8:
            max_memory_per_gpu = {0: '0GIB', 1: '51GIB', 2: '51GIB', 3: '51GIB', 4: '51GIB', 5: '51GIB', 6: '51GIB', 7: '51GIB'}
        else:
            max_memory_per_gpu = {0: '0GIB', 1: '26GIB', 2: '26GIB', 3: '26GIB', 4: '26GIB', 5: '26GIB', 6: '26GIB', 7: '26GIB'}
        print("Max memory per gpu:", max_memory_per_gpu)
        return max_memory_per_gpu

    try:
        if model:
        # model_params calculation, as we don't have a model yet to do:
            model_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        else:
            config = transformers.AutoConfig.from_pretrained(model_name)
            h = config.hidden_size
            l = config.num_hidden_layers
            v = config.vocab_size
            # from https://github.com/bigscience-workshop/bigscience/tree/6917a3b5fefcf439d3485ca184b4d9f6ab605150/math#model-sizing
            model_params = l*(12*h**2 + 13*h) + v*h + 4*h
    except:
        print(f"The model {model_name} has a broken config file. Please notify the owner")
        raise

    if dtype == torch.int8:
        bytes = 1
    else:
        bytes = torch.finfo(dtype).bits / 8
    param_memory_total_in_bytes = model_params * bytes
    # if "opt-30b" in model_name or "opt-66b" in model_name:
    if "opt-30b" in model_name:
        scale = 1.10
    elif "opt-66b" in model_name:
        scale = 1.15
    else:
        scale = 1.05
    # add 5% since weight sizes aren't the same and some GPU may need more memory
    param_memory_per_gpu_in_bytes = int(param_memory_total_in_bytes / n_gpus * scale)
    print(f"Estimating {param_memory_per_gpu_in_bytes/2**30:0.2f}GB per gpu for weights")

    # check the real available memory
    # load cuda kernels first and only measure the real free memory after loading (shorter by ~2GB)
    torch.ones(1).cuda()
    max_memory_per_gpu_in_bytes = torch.cuda.mem_get_info(0)[0]
    if max_memory_per_gpu_in_bytes < param_memory_per_gpu_in_bytes:
        raise ValueError(f"Unable to generate the memory map automatically as the needed estimated memory per gpu ({param_memory_per_gpu_in_bytes/2**30:0.2f}GB) is bigger than the available per gpu memory ({max_memory_per_gpu_in_bytes/2**30:0.2f}GB)")

    max_memory_per_gpu = {i: param_memory_per_gpu_in_bytes for i in range(torch.cuda.device_count())}
    print("Max memory per gpu:", max_memory_per_gpu)
    return max_memory_per_gpu
