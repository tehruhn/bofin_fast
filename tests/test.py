"""
Tests for the HEOM solvers.
"""

from unittest import TestCase
from qutip import Qobj, sigmaz, sigmax, basis, expect, Options, destroy
import numpy as np
from numpy.linalg import eigvalsh
from scipy.integrate import quad
from heom.pyheom import BosonicHEOMSolver
from heom.pyheom import FermionicHEOMSolver


class TestHEOM(TestCase):
    """
    Test class for the HEOM solvers.
    """

    def test_bosonic_HEOM(self):
        """
        Test for the Bosonic HEOM solver.
        Compares analytical and numerical results.
        """

        # Utility functions
        def cot(x):
            return 1.0 / np.tan(x)

        def pure_dephasing_evolution_analytical(tlist, wq, ck, vk):
            evolution = np.array(
                [np.exp(-1j * wq * t - correlation_integral(t, ck, vk)) for t in tlist]
            )
            return evolution

        def correlation_integral(t, ck, vk):
            t1 = np.sum(np.multiply(np.divide(ck, vk ** 2), np.exp(vk * t) - 1))
            t2 = np.sum(
                np.multiply(
                    np.divide(np.conjugate(ck), np.conjugate(vk) ** 2),
                    np.exp(np.conjugate(vk) * t) - 1,
                )
            )
            t3 = np.sum(
                (np.divide(ck, vk) + np.divide(np.conjugate(ck), np.conjugate(vk))) * t
            )
            return 2 * (t1 + t2 - t3)

        # Defining the Hamiltonian
        eps, Del = 0.0, 0.0
        Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()

        # System-bath coupling (Drude-Lorentz spectral density)
        Q = sigmaz()
        tlist = np.linspace(0, 50, 50)

        # Bath properties:
        gamma, lam, T, beta, NC, Nk, pref = 0.5, 0.1, 0.5, 2.0, 6, 3, 1.0
        ckAR = [pref * lam * gamma * (cot(gamma / (2 * T))) + 0.0j]
        ckAR.extend(
            [
                (
                    pref
                    * 4
                    * lam
                    * gamma
                    * T
                    * 2
                    * np.pi
                    * k
                    * T
                    / ((2 * np.pi * k * T) ** 2 - gamma ** 2)
                )
                + 0.0j
                for k in range(1, Nk + 1)
            ]
        )
        vkAR = [gamma + 0.0j]
        vkAR.extend([2 * np.pi * k * T + 0.0j for k in range(1, Nk + 1)])
        ckAI = [pref * lam * gamma * (-1.0) + 0.0j]
        vkAI = [gamma + 0.0j]

        # HEOM parameters
        NR, NI = len(ckAR), len(ckAI)
        Q2 = [Q for kk in range(NR + NI)]
        options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

        # Numerical solution
        HEOMMats = BosonicHEOMSolver(
            Hsys, Q2, ckAR, ckAI, vkAR, vkAI, NC, options=options
        )

        # Initial state of the system
        psi = (basis(2, 0) + basis(2, 1)) / np.sqrt(2)
        rho0 = psi * psi.dag()
        resultMats = HEOMMats.run(rho0, tlist)

        # Some operators to measure the system
        P12p = basis(2, 0) * basis(2, 1).dag()
        P12exp = expect(resultMats.states, P12p)

        # Analytical solution
        lmaxmats2 = Nk
        ck = [
            pref * lam * gamma * (cot(gamma / (2 * T)))
            + pref * lam * gamma * (-1.0) * 1.0j
        ]
        ck.extend(
            [
                (
                    pref
                    * 4
                    * lam
                    * gamma
                    * T
                    * 2
                    * np.pi
                    * k
                    * T
                    / ((2 * np.pi * k * T) ** 2 - gamma ** 2)
                )
                + 0.0j
                for k in range(1, lmaxmats2 + 1)
            ]
        )
        vk = [-gamma]
        vk.extend([-2 * np.pi * k * T + 0.0j for k in range(1, lmaxmats2 + 1)])
        PEG_DL2 = 0.5 * pure_dephasing_evolution_analytical(
            tlist, 0, np.asarray(ck), np.asarray(vk)
        )

        # difference between expected analytical evolution and HEOM result
        # should be as small as possible
        diff = PEG_DL2 - P12exp
        count = len([i for i in diff if abs(i) > 1e-4])
        self.assertEqual(count, 0)

    def test_fermionic_HEOM(self):
        """
        Test for Fermionic HEOM Solver.
        Compares numerical and analytical results.
        """

        from qutip.states import enr_state_dictionaries

        # Utility functions
        def deltafun(j, k):
            ans = 1 if j == k else 0
            return ans

        def get_aux_matrices(full, level, N_baths, Nk, N_cut, shape, dims):
            nstates, state2idx, idx2state = enr_state_dictionaries(
                [2] * (Nk * N_baths), N_cut
            )
            aux_indices = []
            aux_heom_indices = []
            for stateid in state2idx:
                if np.sum(stateid) == level:
                    aux_indices.append(state2idx[stateid])
                    aux_heom_indices.append(stateid)
            full = np.array(full)
            aux = []
            for i in aux_indices:
                qlist = [
                    Qobj(full[k, i, :].reshape(shape, shape).T, dims=dims)
                    for k in range(len(full))
                ]
                aux.append(qlist)
            return aux, aux_heom_indices, idx2state

        def Gamma_L_w(w):
            return Gamma * W ** 2 / ((w - mu_l) ** 2 + W ** 2)

        def Gamma_w(w, mu):
            return Gamma * W ** 2 / ((w - mu) ** 2 + W ** 2)

        def f(x):
            return 1 / (exp(x) + 1.0)

        integrand = lambda w: (
            (2 / (np.pi))
            * Gamma_w(w, mu_l)
            * Gamma_w(w, mu_r)
            * (f(beta * (w - mu_l)) - f(beta * (w - mu_r)))
            / (
                (Gamma_w(w, mu_l) + Gamma_w(w, mu_r)) ** 2
                + 4 * (w - e1 - lamshift(w, mu_l) - lamshift(w, mu_r)) ** 2
            )
        )

        def real_func(x):
            return np.real(integrand(x))

        def imag_func(x):
            return np.imag(integrand(x))

        def lamshift(w, mu):
            return (w - mu) * Gamma_w(w, mu) / (2 * W)

        def CurrFunc():
            a = -2
            b = 2
            real_integral = quad(real_func, a, b)
            imag_integral = quad(imag_func, a, b)
            return real_integral[0] + 1.0j * imag_integral[0]

        def f_approx(x):
            f = 0.5
            for l in range(1, lmax + 1):
                f = f - 2 * kappa[l] * x / (x ** 2 + epsilon[l] ** 2)
            return f

        def f(x):
            kB = 1.0
            return 1 / (np.exp(x) + 1.0)

        def C(tlist, sigma, mu):
            eta_list = []
            gamma_list = []
            eta_0 = 0.5 * Gamma * W * f_approx(1.0j * beta * W)
            gamma_0 = W - sigma * 1.0j * mu
            eta_list.append(eta_0)
            gamma_list.append(gamma_0)
            if lmax > 0:
                for l in range(1, lmax + 1):
                    eta_list.append(
                        -1.0j
                        * (kappa[l] / beta)
                        * Gamma
                        * W ** 2
                        / (-(epsilon[l] ** 2 / beta ** 2) + W ** 2)
                    )
                    gamma_list.append(epsilon[l] / beta - sigma * 1.0j * mu)
            c_tot = []
            for t in tlist:
                c_tot.append(
                    sum(
                        [
                            eta_list[l] * np.exp(-gamma_list[l] * t)
                            for l in range(lmax + 1)
                        ]
                    )
                )
            return c_tot, eta_list, gamma_list

        # Define parameters
        Gamma, W, T, beta = 0.01, 1.0, 0.025851991, 1.0 / 0.025851991
        theta, mu_l, mu_r = 2.0, 1.0, 1.0
        tlist = np.linspace(0, 100, 100)

        # Pade decomposition
        # Pade cut-off
        lmax = 10
        Alpha = np.zeros((2 * lmax, 2 * lmax))
        for j in range(2 * lmax):
            for k in range(2 * lmax):
                Alpha[j][k] = (deltafun(j, k + 1) + deltafun(j, k - 1)) / np.sqrt(
                    (2 * (j + 1) - 1) * (2 * (k + 1) - 1)
                )
        eigvalsA = eigvalsh(Alpha)
        eps = []
        for val in eigvalsA[0:lmax]:
            eps.append(-2 / val)

        AlphaP = np.zeros((2 * lmax - 1, 2 * lmax - 1))
        for j in range(2 * lmax - 1):
            for k in range(2 * lmax - 1):
                AlphaP[j][k] = (deltafun(j, k + 1) + deltafun(j, k - 1)) / np.sqrt(
                    (2 * (j + 1) + 1) * (2 * (k + 1) + 1)
                )
        eigvalsAP = eigvalsh(AlphaP)

        chi = []
        for val in eigvalsAP[0 : lmax - 1]:
            chi.append(-2 / val)

        eta_list = [
            0.5
            * lmax
            * (2 * (lmax + 1) - 1)
            * (
                np.prod([chi[k] ** 2 - eps[j] ** 2 for k in range(lmax - 1)])
                / np.prod(
                    [eps[k] ** 2 - eps[j] ** 2 + deltafun(j, k) for k in range(lmax)]
                )
            )
            for j in range(lmax)
        ]

        kappa = [0] + eta_list
        epsilon = [0] + eps
        cppL, etapL, gampL = C(tlist, 1.0, mu_l)
        cpmL, etamL, gammL = C(tlist, -1.0, mu_l)
        cppR, etapR, gampR = C(tlist, 1.0, mu_r)
        cpmR, etamR, gammR = C(tlist, -1.0, mu_r)

        # HEOM simulation with above params (Pade)
        options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

        # Single fermion.
        d1 = destroy(2)

        # Site energy
        e1 = 1.0
        H0 = e1 * d1.dag() * d1

        # There are two leads, but we seperate the interaction into two terms, labelled with \sigma=\pm
        # such that there are 4 interaction operators (See paper)
        Qops = [d1.dag(), d1, d1.dag(), d1]
        Kk = lmax + 1
        Ncc = 2  # For a single impurity we converge with Ncc = 2

        # Note here that the functionality differs from the bosonic case. Here we send lists of lists, were each sub-list
        # refers to one of the two coupling terms for each bath (the notation here refers to eta|sigma|L/R)

        eta_list = [etapR, etamR, etapL, etamL]
        gamma_list = [gampR, gammR, gampL, gammL]
        Qops = [d1.dag(), d1, d1.dag(), d1]

        resultHEOM1 = FermionicHEOMSolver(
            H0, Qops, eta_list, gamma_list, Ncc, options=options
        )

        rhossHP, fullssP = resultHEOM1.steady_state()

        # One advantage of this simple model is the current is analytically solvable, so we can check convergence of the result

        # Analytical solution
        curr_ana = CurrFunc()
        aux_1_list_list = []
        aux1_indices_list = []
        K = Kk
        shape = H0.shape[0]
        dims = H0.dims
        aux_1_list, aux1_indices, idx2state = get_aux_matrices(
            [fullssP], 1, 4, K, Ncc, shape, dims
        )
        d1 = destroy(2)
        currP = -1.0j * (
            ((sum([(d1 * aux_1_list[gg][0]).tr() for gg in range(Kk, 2 * Kk)])))
            - ((sum([(d1.dag() * aux_1_list[gg][0]).tr() for gg in range(Kk)])))
        )

        # difference between Pade and analytical current should be small
        diff = abs(curr_ana - (-currP))
        self.assertLess(diff, 1e-4)
