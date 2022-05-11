import argparse
import numpy as np
import torch 
from torch import nn

def process(img, args):
  B, H_in, W_in, C_in = int(args[0]), int(args[1]), int(args[2]), int(args[3])
  return np.array(img).reshape(B, C_in, H_in, W_in)


def process_filter(f, args):
  C, H, W, N = int(args[0]), int(args[1]), int(args[2]), int(args[3])
  f_out = np.array(f).reshape((C, N, H, W))
  return f_out


def read_input(in_file):
  input_numbers, input_file = None, None
  with open(in_file) as f:
    input_numbers = f.readline().split(" ")
    input_numbers = [float(i.replace("\n","")) for i in input_numbers]
    f.readline()
    input_image =  f.readline().split(" ")
    input_image = input_image[:-1]
    input_image = [float(i.replace("\n","")) for i in input_image]
    input_image = torch.tensor(process(input_image, input_numbers))
    f.readline()
    filter_mat = f.readline().split(" ")
    filter_mat =filter_mat[:-1]
    filter_mat = [float(i.replace("\n","")) for i in filter_mat]
    #print(filter_mat)
    #print(input_numbers[3:7])
    filter_mat = torch.tensor(process_filter(filter_mat, input_numbers[3:7]))
    f.readline()
    filter_1d = f.readline().split(" ")
    filter_1d = filter_1d[:-1]
    filter_1d = [float(i.replace("\n","")) for i in filter_1d]
    #print(filter_1d)
    filter_1d = torch.tensor(process_filter(filter_1d, [input_numbers[3], 1,1,input_numbers[9]]))
    f.readline()
    
    output_image = f.readline().split(" ")
    output_image = output_image[:-1]
    output_image = [float(i.replace("\n","")) for i in output_image]
    args = [input_numbers[0], input_numbers[7], input_numbers[8], input_numbers[9]]
    output_image = torch.tensor(process(output_image, args))

  return input_numbers, input_image, filter_mat, filter_1d, output_image

def test(fname, eps = 1e-5):
  args, X, F2d, F1d, Y = read_input(fname)
  print(args)
  print(X.shape)
  print(F2d.shape)
  print(F1d.shape)
  print(Y.shape)
  conv = nn.Conv2d(in_channels=int(args[3]), out_channels=int(args[3]), kernel_size=int(args[4]), stride=int(args[10]), groups=int(args[3]), bias=False)
  conv.weight = torch.nn.Parameter(F2d)
  point_conv = nn.Conv2d(in_channels=int(args[3]), out_channels=int(args[9]), kernel_size=1, bias=False)
  point_conv.weight = torch.nn.Parameter(F1d)
  depthwise_separable_conv = torch.nn.Sequential(conv, point_conv)
  out = depthwise_separable_conv(X.double())
  assert (out - Y).abs().max().item() <= eps, "FAILED CORRECTNESS"
  print("Passed correctness!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Correctess')
    parser.add_argument('--name', help='test file name')
    args = parser.parse_args()
    test(args.name)