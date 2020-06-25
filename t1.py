from qutip import *
from heom1 import HSolverDL
import matplotlib.pyplot as plt
import numpy as np  

# Defining the system Hamiltonian
eps = 0.5     # Energy of the 2-level system.
Del = 1.0    # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del* sigmax()


# Bath description parameters (for HEOM)
temperature = 1.0/0.95 # in units where Boltzmann factor is 1
Nk = 2 # number of exponentials in approximation of the the spectral density
Ncut = 30 # cut off parameter for the bath

# System-bath coupling (Drude-Lorentz spectral density)
Q = sigmaz() # coupling operator
gam = 0.05 # cut off frequency
lam = 0.05 # coupling strenght

c = []
nu = []
lam0 = lam
hbar = 1
beta = 1.0/(1*temperature)
N_m = Nk

g = 2*np.pi / (beta*hbar)
for k in range(N_m):
    if k == 0:
        nu.append(gam)
        c.append(lam0*gam*
            (1.0/np.tan(gam*hbar*beta/2.0) - 1j) / hbar)
    else:
        nu.append(k*g)
        c.append(4*lam0*gam*nu[k] /
              ((nu[k]**2 - gam**2)*beta*hbar**2))

# print(c)
# print(nu)
# Configure the solver
hsolver = HSolverDL(Hsys, Q, lam, temperature, Ncut, Nk, gam, stats=True, renorm = False)

# Initial state of the system.
rho0 = basis(2,0) * basis(2,0).dag()   
# Times to record state
tlist = np.linspace(0, 40, 600)
# run the solver
result = hsolver.run(rho0, tlist)
print(result.states[10])

# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p=basis(2,0) * basis(2,0).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p=basis(2,0) * basis(2,1).dag()
# Calculate expectation values in the bases
P11exp = expect(result.states, P11p)
P12exp = expect(result.states, P12p)

# Plot the results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
axes.plot(tlist, np.real(P11exp), 'b', linewidth=2, label="P11")
axes.plot(tlist, np.real(P12exp), 'r', linewidth=2, label="P12")
axes.set_xlabel(r't', fontsize=28)
axes.legend(loc=0, fontsize=12)
plt.show()
