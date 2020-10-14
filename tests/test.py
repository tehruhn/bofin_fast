from unittest import TestCase

from qutip import Qobj, sigmaz, sigmax, basis, expect, Options
import numpy as np
from numpy.linalg import eigvalsh 
from scipy.integrate import quad
from heom.pyheom import BosonicHEOMSolver
from heom.pyheom import FermionicHEOMSolver

class TestClass(TestCase):
    def test_bosonic_HEOM(self):
    	assert True
    def test_fermionic_HEOM(self):
    	assert True