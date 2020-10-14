# A Bosonic and Fermionic hierarchical-equations-of-motion library for QuTiP with applications in light-harvesting, quantum control, and single-molecule electronics

Neill Lambert, Tarun Raheja, Shahnawaz Ahmed, Alexander Pitchford, Franco Nori 

## Abstract

The “hierarchical equations of motion” (HEOM) method is a powerful numerical approach to solve the dynamics and steady-state of a quantum system coupled to a non-Markovian and non-perturbative environment. Originally developed in the context of physical chemistry, it has also been extended and applied to problems in solid-state physics, optics, single-molecule electronics,and biological physics. Here we present a numerical library in Python, integrated with the powerful QuTiP platform, which implements the HEOM for both Bosonic and Fermionic environments. Wedemonstrate it’s utility with a series of examples.  For the Bosonic case, we present examples for fitting arbitrary spectral densities, modelling a Fenna-Matthews-Olsen photosynthetic complex,and simulating dynamical decoupling of a spin from it’s environment.  For the Fermionic case, we present an integrable single-impurity example, used as a benchmark of the code, and a morecomplex example of an impurity strongly coupled to a single vibronic mode, with applications in single-molecule electronics.

## Repository layout

## Usage instructions for Python version

These commands should be run from the main repository folder, `qutip_heom`.
Install dependencies, and then install the package locally :
```
pip3 install -r requirements.txt
pip3 install -e .
```
To run tests using `Nose`, run :
```
nosetests
```
## Usage instructions for C++ version

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

## Citation

## License

## Resources
