from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='dws_cpp',
      ext_modules=[cpp_extension.CppExtension('dws_cpp', ['dws.cpp'],\
         extra_compile_args=['-Wall', '-march=knl', '-ffast-math', '-fopenmp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})