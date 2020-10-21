#!/usr/bin/env python
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "bofinfast.interfacer",
        sources=["bofinfast/interfacer.pyx", "bofinfast/utilities.cpp"],
        include_dirs=[np.get_include(), '.', 'bofinfast'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language='c++')
]

REQUIRES = ["cython", "numpy", "scipy", "qutip"]

setup(
    name="bofinfast",
    version="0.0.1",
    description="C++ version for Bosonic and Fermionic hierarchical-equations-of-motion library for QuTiP",
    long_description=open("README.md").read(),
    url="",
    author="Neill Lambert, Tarun Raheja, Shahnawaz Ahmed, Alexander Pitchford",
    author_email="nwlambert@gmail.com",
    packages=find_packages(include=["bofinfast", "bofinfast.*"]),
    install_requires=REQUIRES,
    ext_modules = cythonize(extensions)
)