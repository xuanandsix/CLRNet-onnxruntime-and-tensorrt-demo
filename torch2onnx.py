import os
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.datasets import build_dataloader
from clrnet.models.registry import build_net

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.load_from
    net = build_net(cfg)
    net = net.cpu()

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(args.load_from, map_location='cpu')['net']

    for k, v in state_dict.items():
        namekey = k[7:]    # 去掉module前缀
        new_state_dict[namekey] = v
        
    net.load_state_dict(new_state_dict)
    
    dummy_input = torch.randn(1, 3, 320,800, device='cpu')
    torch.onnx.export(net, dummy_input, 'tusimple_r18.onnx',
                    export_params=True, opset_version=11, do_constant_folding=True,
                    input_names = ['input'])
   
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')

    parser.add_argument('--load_from',
                    default=None,
                    help='the checkpoint file to load from')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()

# python torch2onnx.py configs/clrnet/clr_resnet18_tusimple.py  --load_from tusimple_r18.pth

