# python setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os

# os.environ["CC"] = "g++" 
# os.environ["CXX"] = "g++"

setup(
    name = "mkl_local",
    ext_modules = cythonize('mkl_local.pyx'),
    include_dirs = [numpy.get_include()]
)