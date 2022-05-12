from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='dws_gpu_cpp',
    ext_modules=[cpp_extension.CUDAExtension('dws_gpu_cpp', ['dws_gpu_handler.cpp','dws_gpu.cu'],)],
    cmdclass={'build_ext': cpp_extension.BuildExtension})