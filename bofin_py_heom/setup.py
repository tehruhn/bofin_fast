from setuptools import setup, find_packages

REQUIRES = ["Cython", "qutip", "numpy", "scipy"]

setup(
    name="bofinpy",
    version="0.0.1",
    description="Python version for Bosonic and Fermionic hierarchical-equations-of-motion library for QuTiP",
    long_description=open("../README.md").read(),
    url="",
    author="Neill Lambert, Tarun Raheja, Shahnawaz Ahmed, Alexander Pitchford",
    author_email="nwlambert@gmail.com",
    packages=find_packages(include=['heom', 'heom.*']),
    requires=REQUIRES,
    test_suite='nose.collector',
    tests_require=['unittest','nose']
)