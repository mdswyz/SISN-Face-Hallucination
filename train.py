import json
import importlib
import torch
from option import get_option
from solver import Solver
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    opt = get_option()
    torch.manual_seed(opt.seed)

    module = importlib.import_module("model.{}".format(opt.model.lower()))

    if not opt.test_only:
        print(json.dumps(vars(opt), indent=4))

    solver = Solver(module, opt)
    if opt.test_only:
        print("Evaluate {} (loaded from {})".format(opt.model, opt.pretrain))
        psnr = solver.evaluate()
        print("{:.2f}".format(psnr))
    else:
        solver.fit()

if __name__ == "__main__":
    main()
