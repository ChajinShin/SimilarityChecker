import torch
import os
import yaml
from solver import Solver


def main():
    # fetch option
    with open('./config.yml', 'r', encoding='utf8') as f:
        opt =yaml.load(f, Loader=yaml.Loader)

    torch.manual_seed(opt['seed'])
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if opt['use_cuda'] else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_idx']

    Solver(opt, dev)


if __name__ == "__main__":
    main()
