"""
Tests for the Bosonic HEOM solvers.
"""

from qutip import Qobj, sigmaz, sigmax, basis, expect, Options, destroy
import numpy as np
from _testUtils import *
from numpy.linalg import eigvalsh 
from scipy.integrate import quad
from bofinpy.heom import BosonicHEOMSolver as BosonicHEOMSolverPy
from bofincpp.heom import BosonicHEOMSolver as BosonicHEOMSolverCPP

class TestBosonicHEOM:
    """
    Test class for the Bosonic HEOM solvers.
    """

    def test_bosonic_HEOM_py(self):
        """
        Test for the Python Bosonic HEOM solver.
        Compares analytical and numerical results.
        """

        # Defining the Hamiltonian
        eps, Del = 0., 0.     
        Hsys = 0.5*eps*sigmaz() + 0.5*Del*sigmax()

        # System-bath coupling (Drude-Lorentz spectral density)
        Q = sigmaz()
        tlist = np.linspace(0, 50, 50)

        # Bath properties:
        gamma, lam, T, beta, NC, Nk, pref = .5, .1, .5, 2., 6, 3, 1.
        ckAR = [pref*lam*gamma*(cot(gamma/(2*T))) + 0.j]
        ckAR.extend([(pref*4*lam*gamma*T*2*np.pi*k*T / 
            ((2*np.pi*k*T)**2 - gamma**2)) + 0.j for k in range(1, Nk+1)])
        vkAR = [gamma+0.j]
        vkAR.extend([2*np.pi*k*T + 0.j for k in range(1, Nk+1)])
        ckAI = [pref * lam * gamma * (-1.0) + 0.j]
        vkAI = [gamma+0.j]

        # HEOM parameters
        NR, NI = len(ckAR), len(ckAI)
        Q2 = [Q for kk in range(NR+NI)]
        options = Options(nsteps=15000, store_states=True, rtol=1e-14, 
            atol=1e-14)

        # Numerical solution
        HEOMMats = BosonicHEOMSolverPy(Hsys, Q2, ckAR, ckAI, vkAR, vkAI, NC, 
            options=options)

        # Initial state of the system
        psi = (basis(2,0) + basis(2,1))/np.sqrt(2)
        rho0 = psi * psi.dag()
        resultMats = HEOMMats.run(rho0, tlist)

        # Some operators to measure the system
        P12p = basis(2,0)*basis(2,1).dag()
        P12exp = expect(resultMats.states, P12p)

        # Analytical solution
        lmaxmats2 =  Nk
        ck = [pref*lam*gamma*(cot(gamma/(2*T))) + pref*lam*gamma*(-1.0)*1.j]
        ck.extend([(pref*4*lam*gamma*T*2*np.pi*k*T / 
            ((2*np.pi*k*T)**2 - gamma**2)) + 0.j \
            for k in range(1, lmaxmats2 + 1)])
        vk = [-gamma]
        vk.extend([-2*np.pi*k*T + 0.j for k in range(1, lmaxmats2 + 1)])
        PEG_DL2 = 0.5*pure_dephasing_evolution_analytical(tlist, 0, 
            np.asarray(ck), np.asarray(vk))

        # difference between expected analytical evolution and HEOM result
        # should be as small as possible
        diff = PEG_DL2 - P12exp 
        count = len([i for i in diff if abs(i) > 1e-4])
        assert count == 0

    def test_bosonic_HEOM_cpp(self):
        """
        Test for the C++ Bosonic HEOM solver.
        Compares analytical and numerical results.
        """

        # Defining the Hamiltonian
        eps, Del = 0., 0.     
        Hsys = 0.5*eps*sigmaz() + 0.5*Del*sigmax()

        # System-bath coupling (Drude-Lorentz spectral density)
        Q = sigmaz()
        tlist = np.linspace(0, 50, 50)

        # Bath properties:
        gamma, lam, T, beta, NC, Nk, pref = .5, .1, .5, 2., 6, 3, 1.
        ckAR = [pref*lam*gamma*(cot(gamma/(2*T))) + 0.j]
        ckAR.extend([(pref*4*lam*gamma*T*2*np.pi*k*T / 
            ((2*np.pi*k*T)**2 - gamma**2)) + 0.j for k in range(1, Nk+1)])
        vkAR = [gamma+0.j]
        vkAR.extend([2*np.pi*k*T + 0.j for k in range(1, Nk+1)])
        ckAI = [pref * lam * gamma * (-1.0) + 0.j]
        vkAI = [gamma+0.j]

        # HEOM parameters
        NR, NI = len(ckAR), len(ckAI)
        Q2 = [Q for kk in range(NR+NI)]
        options = Options(nsteps=15000, store_states=True, rtol=1e-14, 
            atol=1e-14)

        # Numerical solution
        HEOMMats = BosonicHEOMSolverCPP(Hsys, Q2, ckAR, ckAI, vkAR, vkAI, NC, 
            options=options)

        # Initial state of the system
        psi = (basis(2,0) + basis(2,1))/np.sqrt(2)
        rho0 = psi * psi.dag()
        resultMats = HEOMMats.run(rho0, tlist)

        # Some operators to measure the system
        P12p = basis(2,0)*basis(2,1).dag()
        P12exp = expect(resultMats.states, P12p)

        # Analytical solution
        lmaxmats2 =  Nk
        ck = [pref*lam*gamma*(cot(gamma/(2*T))) + pref*lam*gamma*(-1.0)*1.j]
        ck.extend([(pref*4*lam*gamma*T*2*np.pi*k*T / 
            ((2*np.pi*k*T)**2 - gamma**2)) + 0.j \
            for k in range(1, lmaxmats2 + 1)])
        vk = [-gamma]
        vk.extend([-2*np.pi*k*T + 0.j for k in range(1, lmaxmats2 + 1)])
        PEG_DL2 = 0.5*pure_dephasing_evolution_analytical(tlist, 0, 
            np.asarray(ck), np.asarray(vk))

        # difference between expected analytical evolution and HEOM result
        # should be as small as possible
        diff = PEG_DL2 - P12exp 
        count = len([i for i in diff if abs(i) > 1e-4])
        assert count == 0