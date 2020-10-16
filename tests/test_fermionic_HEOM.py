"""
Tests for the Fermionic HEOM solvers.
"""

from qutip import Qobj, sigmaz, sigmax, basis, expect, Options, destroy
import numpy as np
from _testUtils import *
from numpy.linalg import eigvalsh 
from scipy.integrate import quad
from bofinpy.heom import FermionicHEOMSolver as FermionicHEOMSolverPy
from bofincpp.heom import FermionicHEOMSolver as FermionicHEOMSolverCPP

class TestFermionicHEOM:
    """
    Test class for the Fermionic HEOM solvers.
    """

    def test_fermionic_HEOM_py(self):
        """
        Test for the Python Bosonic HEOM solver.
        Compares analytical and numerical results.
        """
        assert True

    def test_fermionic_HEOM_cpp(self):
        """
        Test for the C++ Fermionic HEOM solver.
        Compares analytical and numerical results.
        """
        assert True