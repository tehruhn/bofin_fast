from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from heom_fmotd import BosonicHEOMSolver

###############################################

# defines cks and vks for a spectral density

# generating correlation function from spectral density

Nk = 2 

def cot(x):
    return 1./np.tan(x)

# computing cks and vks
print("Computing cks and vks")
gamma = 0.05
# alpha = 1/(2*np.pi)
alpha = 0.05
lam = alpha
T = 1/0.95

wlist = np.linspace(0, 10, 100)

pref = 1.

J = [w * 2 * lam * gamma / (np.pi*(gamma**2 + w**2)) for w in wlist]

def c(t):
    c_temp =[]
    c_temp.append(pref * lam * gamma * (-1.0j + cot(gamma / (2 * T))) * np.exp(-gamma * t))
    for k in range(1,100):
        vk = 2 * np.pi * k * T
        c_temp.append((pref * 4 * lam * gamma * T * vk / (vk**2 - gamma**2))  * np.exp(- vk * t) ) 
    return c_temp

ckAR = [pref * lam * gamma * (cot(gamma / (2 * T))) + 0.j]
ckAR.extend([(pref * 4 * lam * gamma * T *  2 * np.pi * k * T / (( 2 * np.pi * k * T)**2 - gamma**2))+0.j for k in range(1,Nk)])

vkAR = [gamma+0.j]
vkAR.extend([2 * np.pi * k * T + 0.j for k in range(1,Nk)])

ckAI= [pref * lam * gamma * (-1.0) + 0.j]

vkAI = [gamma+0.j]

print(ckAR)
print(ckAI)

print(vkAR)
print(vkAI)


# Defining the system Hamiltonian
eps = 0.5     # Energy of the 2-level system.
Del = 1.0    # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del* sigmax()

# Times to record state
tlist = np.linspace(0, 40, 600)
# Reals parts
ans1 = [np.real(sum(c(t))) for t in tlist]
# Imaginary parts
ans2 = [np.imag(sum(c(t))) for t in tlist]

Q = sigmaz() # coupling operator

NC = 30
NR = len(ckAR)
NI = len(ckAI)
Q2 = [Q for kk in range(NR+NI)]
# print(Q2)
options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

resultNew = BosonicHEOMSolver(Hsys, Q2, ckAR, ckAI, vkAR, vkAI, NC, options=options)

# Initial state of the system.
rho0 = basis(2,0) * basis(2,0).dag()   

result = resultNew.run(rho0, tlist)
print(result.states[10])

# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p=basis(2,0) * basis(2,0).dag()
P22p=basis(2,1) * basis(2,1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p=basis(2,0) * basis(2,1).dag()
# Calculate expectation values in the bases
P11exp = expect(result.states, P11p)
P22exp = expect(result.states, P22p)
P12exp = expect(result.states, P12p)

# Plot the results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,8))
axes.plot(tlist, np.real(P11exp)+ np.real(P22exp), 'b', linewidth=2, label="P11")
axes.plot(tlist, np.real(P11exp), 'b', linewidth=2, label="P11")
axes.plot(tlist, np.real(P12exp), 'r', linewidth=2, label="P12")
axes.set_xlabel(r't', fontsize=28)
axes.legend(loc=0, fontsize=12)
plt.show()