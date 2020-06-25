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
    name = "interfacer",
    ext_modules = cythonize('*.pyx'),
    include_dirs = [numpy.get_include()]
)