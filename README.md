# BoFiN : A Bosonic and Fermionic hierarchical-equations-of-motion library for QuTiP with applications in light-harvesting, quantum control, and single-molecule electronics

Neill Lambert, Tarun Raheja, Shahnawaz Ahmed, Alexander Pitchford, Franco Nori 

## Abstract

The “hierarchical equations of motion” (HEOM) method is a powerful numerical approach to solve the dynamics and steady-state of a quantum system coupled to a non-Markovian and non-perturbative environment. Originally developed in the context of physical chemistry, it has also been extended and applied to problems in solid-state physics, optics, single-molecule electronics,and biological physics. Here we present a numerical library in Python, integrated with the powerful QuTiP platform, which implements the HEOM for both Bosonic and Fermionic environments. Wedemonstrate it’s utility with a series of examples.  For the Bosonic case, we present examples for fitting arbitrary spectral densities, modelling a Fenna-Matthews-Olsen photosynthetic complex,and simulating dynamical decoupling of a spin from it’s environment.  For the Fermionic case, we present an integrable single-impurity example, used as a benchmark of the code, and a morecomplex example of an impurity strongly coupled to a single vibronic mode, with applications in single-molecule electronics.

## Repository layout

The repository contains two packaged versions of the HEOM solver : 

- **BoFiNPy** : Pure Python version of the HEOM solver. Has a `BosonicHEOMSolver` and `FermionicHEOMSolver`.
- **BoFiNCPP** : Hybrid C++ - Python version, with backend for RHS construction of the HEOM solver written in C++. Otherwise completely identical in user interface and functionality to the pure Python version.

It should be noted that the C++ version dramatically speeds up RHS construction, with respect to the Python version. (TODO SHOW PERFORMANCE GRAPH)

## Install dependencies

From the main repository folder, `qutip_heom`, run the following command to install dependencies :
```
pip3 install -r requirements.txt
```
## Usage instructions for Python version (BoFiN-Py)

From `qutip_heom`, navigate to the `bofin_py_heom` folder using `cd bofin_py_heom/`. From here, run the following commands :
```
pip3 install -e .
```
This installs the pure Python version of the HEOM solvers. These solvers can be imported as :
```
from bofinpy.heom import BosonicHEOMSolver, FermionicHEOMSolver
```
## Usage instructions for C++ version (BoFiN-CPP)

From inside `qutip_heom`, navigate to the `bofin_cpp_heom` folder using `cd bofin_cpp_heom/`. From here, run the following commands :
```
python3 setup.py build_ext --inplace
pip3 install -e .
```
This installs the hybrid Python - C++ version of the HEOM solvers. These are identical in usage to the Python solvers. These solvers can be imported as :
```
from bofincpp.heom import BosonicHEOMSolver, FermionicHEOMSolver
```

## Usage example

## Documentation

To build the documentation locally:
```
cd docs/
make html
```

The documentation HTML files can be found in `docs/build/html`. The main file is `index.html`.

## Example notebooks

There are several example notebooks illustrating usage of the code, in the `example_notebooks` folder.

## Running tests

To run tests using `Nose` in the `qutip_heom` directory, run :
```
nosetests
```
## Citation

## License

## Resources
