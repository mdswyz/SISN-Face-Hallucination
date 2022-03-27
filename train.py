import json
import importlib
import torch
from option import get_option
from solver import Solver
import torch.distributed as dist
import os
import torch.multiprocessing as mp

def main(rank, opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)

    dist.init_process_group("nccl", rank=rank, world_size=opt.num_gpu)
    torch.manual_seed(opt.seed)

    module = importlib.import_module("model.{}".format(opt.model.lower()))

    if not opt.test_only:
        print("rank = {}".format(rank))
        print(json.dumps(vars(opt), indent=4))

    solver = Solver(module, opt)
    if opt.test_only:
        print("Evaluate {} (loaded from {})".format(opt.model, opt.pretrain))
        psnr = solver.evaluate()
        print("{:.2f}".format(psnr))
    else:
        solver.fit()

if __name__ == "__main__":
    opt = get_option()
    
    mp.spawn(
        main,
        args=(opt),
        nprocs=opt.num_gpu
    )
