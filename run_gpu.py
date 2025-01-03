import torch 
import time
import os
import argparse
import shutil
import sys
 
def parse_args():
    parser = argparse.ArgumentParser(description='Matrix multiplication')
    parser.add_argument('--gpus', help='gpu amount', default=1, type=int)
    parser.add_argument('--size', help='matrix size', default=12000, type=int)
    parser.add_argument('--interval', help='sleep interval', default=0.01, type=float)
    args = parser.parse_args()
    return args
 
 
def matrix_multiplication(args):
 
    a_list, b_list, result = [], [], []    
    size = (args.size, args.size)
    
    for i in range(args.gpus):
        a_list.append(torch.rand(size, device=i))
        b_list.append(torch.rand(size, device=i))
        result.append(torch.rand(size, device=i))
 
    while True:
        for i in range(args.gpus):
            result[i] = a_list[i] * b_list[i]
        time.sleep(args.interval)
 
if __name__ == "__main__":
    # usage: python run_gpu.py --size 20000 --gpus 4 --interval 0.01
    args = parse_args()
    matrix_multiplication(args)