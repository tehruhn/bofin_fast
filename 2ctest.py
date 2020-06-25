from qutip import *
from heom_fmotd_NL import BosonicHEOMSolver
import numpy as np


def cot(x):
    return 1./np.tan(x)
# Defining the system Hamiltonian
eps = .5     # Energy of the 2-level system.
Del = 1.0    # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del* sigmax()# Initial state of the system.
rho0 = basis(2,0) * basis(2,0).dag()  
# System-bath coupling (Drude-Lorentz spectral density)
Q = sigmaz() # coupling operator

tlist = np.linspace(0, 600, 6000)

#Bath properties:
gamma = 2. # cut off frequency
lam = .01 # coupling strenght
T = 1
beta = 1./T

#HEOM parameters
NC = 1 # cut off parameter for the bath




wlist = np.linspace(0, 5, 1000)
pref = 1.

J = [w * 2 * lam * gamma / ((gamma**2 + w**2)) for w in wlist]

Nk = 30 # number of exponentials in approximation of the Matsubara approximation


def _calc_matsubara_params():
        """
        Calculate the Matsubara coefficents and frequencies
        Returns
        -------
        c, nu: both list(float)
        """
        c = []
        nu = []
        lam0 = lam
        gam = gamma
        hbar = 1
        beta = 1.0/T
        N_m =  Nk

        g = 2*np.pi / (beta)
        for k in range(N_m):
            if k == 0:
                nu.append(gam)
                c.append(lam0*gam*
                    (1.0/np.tan(gam*hbar*beta/2.0) - 1j) / hbar)
            else:
                g = 2*np.pi / (beta)
                nu.append(k*g)
                c.append(4*lam0*gam*nu[k] /
                      ((nu[k]**2 - gam**2)*beta*hbar**2))

    
        return c, nu

ctest,nutest=_calc_matsubara_params()



ckAR = [pref * lam * gamma * (cot(gamma / (2 * T))) + 0.j]
ckAR.extend([(pref * 4 * lam * gamma * T *  2 * np.pi * k * T / (( 2 * np.pi * k * T)**2 - gamma**2))+0.j for k in range(1,Nk)])

vkAR = [gamma+0.j]
vkAR.extend([2 * np.pi * k * T + 0.j for k in range(1,Nk)])

ckAI = [pref * lam * gamma * (-1.0) + 0.j]

vkAI = [gamma+0.j]




tlist_corr = np.linspace(0,2,1000)

lmaxmats = 15000

def c(t,anamax):

    c_temp = (pref * lam * gamma * (-1.0j + cot(gamma / (2 * T))) * np.exp(-gamma * t))
    for k in range(1, anamax):
        vk = 2 * np.pi * k * T
        c_temp += ((pref * 4 * lam * gamma * T * vk / (vk**2 - gamma**2))  * np.exp(- vk * t) ) 
        
    
    return c_temp

# Reals parts
corrRana = [np.real(c(t,lmaxmats)) for t in tlist_corr]
# Imaginary parts
corrIana = [np.imag((pref * lam * gamma * (-1.0j + cot(gamma / (2 * T))) * np.exp(-gamma * t))) for t in tlist_corr]


NR = len(ckAR)
NI = len(ckAI)
Q2 = [Q for kk in range(NR+NI)]
# print(Q2)
options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)
import time
start = time.time()
HEOMMats = BosonicHEOMSolver(Hsys, Q2, ckAR, ckAI, vkAR, vkAI, NC, options=options)
end = time.time()
print(end - start)



resultMats = HEOMMats.run(rho0, tlist)
#print(resultMats.states)
