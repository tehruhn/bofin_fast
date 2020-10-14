from qutip import Qobj, sigmaz, sigmax, basis, expect, Options
import numpy as np
from scipy.integrate import quad

from heom.pyheom import BosonicHEOMSolver


def cot(x):
    return 1.0 / np.tan(x)

    # Defining the system Hamiltonian


eps = 0.0  # Energy of the 2-level system.
Del = 0.0  # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()

# System-bath coupling (Drude-Lorentz spectral density)
Q = sigmaz()  # coupling operator

tlist = np.linspace(0, 50, 50)

# Bath properties:
gamma = 0.5  # cut off frequency
# gamma = 0.1
lam = 0.1  # coupling strenght
T = 0.5
beta = 1.0 / T
# HEOM parameters

NC = 6  # cut off parameter for the bath


Nk = 3  # number of exponentials in approximation of the the spectral density

pref = 1.0

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


NR = len(ckAR)
NI = len(ckAI)
Q2 = [Q for kk in range(NR + NI)]

options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)
print(Q2)
HEOMMats = BosonicHEOMSolver(Hsys, Q2, ckAR, ckAI, vkAR, vkAI, NC, options=options)

# Initial state of the system.

psi = (basis(2, 0) + basis(2, 1)) / np.sqrt(2)

rho0 = psi * psi.dag()


resultMats = HEOMMats.run(rho0, tlist)

# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
# Calculate expectation values in the bases
P11exp = expect(resultMats.states, P11p)
P22exp = expect(resultMats.states, P22p)
P12exp = expect(resultMats.states, P12p)


def pure_dephasing_evolution_analytical(tlist, wq, ck, vk):
    """
    Computes the propagating function appearing in the pure dephasing model.
        
    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.
    
    wq: float
        The qubit frequency in the Hamiltonian.

    ck: ndarray
        The list of coefficients in the correlation function.
        
    vk: ndarray
        The list of frequencies in the correlation function.
    
    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    evolution = np.array(
        [np.exp(-1j * wq * t - correlation_integral(t, ck, vk)) for t in tlist]
    )
    return evolution


def correlation_integral(t, ck, vk):
    """
    Computes the integral sum function appearing in the pure dephasing model.
    
    If the correlation function is a sum of exponentials then this sum
    is given by:
    
    .. math:
        
        \int_0^{t}d\tau D(\tau) = \sum_k\frac{c_k}{\mu_k^2}e^{\mu_k t}
        + \frac{\bar c_k}{\bar \mu_k^2}e^{\bar \mu_k t}
        - \frac{\bar \mu_k c_k + \mu_k \bar c_k}{\mu_k \bar \mu_k} t
        + \frac{\bar \mu_k^2 c_k + \mu_k^2 \bar c_k}{\mu_k^2 \bar \mu_k^2}
        
    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.
    
    ck: ndarray
        The list of coefficients in the correlation function.
        
    vk: ndarray
        The list of frequencies in the correlation function.
    
    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    t1 = np.sum(np.multiply(np.divide(ck, vk ** 2), np.exp(vk * t) - 1))

    t2 = np.sum(
        np.multiply(
            np.divide(np.conjugate(ck), np.conjugate(vk) ** 2),
            np.exp(np.conjugate(vk) * t) - 1,
        )
    )
    t3 = np.sum((np.divide(ck, vk) + np.divide(np.conjugate(ck), np.conjugate(vk))) * t)

    return 2 * (t1 + t2 - t3)


lmaxmats2 = Nk


ck = [pref * lam * gamma * (cot(gamma / (2 * T))) + pref * lam * gamma * (-1.0) * 1.0j]
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
print("-----------------")
print(PEG_DL2)
print("-----------------")

print("difference between expected analytical evolution and heom result:")
print(
    PEG_DL2 - P12exp
)  # This difference should be very small if we use same number of matsubara terms
