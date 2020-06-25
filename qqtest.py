from qutip import *
# from heom1 import HSolverDL
from math import sqrt, cos
from qutip import tensor, identity, destroy, sigmax, sigmay, sigmaz, basis, qeye
from heom_fmotd_NL import BosonicHEOMSolver
import numpy as np
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
import matplotlib
import matplotlib.pyplot as plt

gamma = 0.05
w0=1.0


lam =0.2
Gamma = gamma/2.
Om = np.sqrt(w0**2 - Gamma**2)
factor=1./4.
ckAR= [factor*lam**2/(Om),factor*lam**2/(Om)]
#note:  the frequencies here are NEGATIVE from their nomral def
vkAR= [-(1.0j*Om - Gamma),-(-1.0j*Om - Gamma)]

ckAI =[-factor*lam**2*1.0j/(Om),factor*lam**2*1.0j/(Om)]
#ckAI =[factor*lam**2*1.0j/(Om),-factor*lam**2*1.0j/(Om)]

vkAI = [-(-1.0j*Om - Gamma),-(1.0j*Om - Gamma)]



#ckAR=[0.14534+0.316206j, 0.14534-0.316206j,-(0.0587924+0.0207246j), -(0.0587924-0.0207246j)] 
#vkAR=[-2.77201-0.985685j,-2.77201+0.985685j,-2.67694-3.11522j,-2.67694+3.11522j]


# third and fourth terms are changed slightly to make exactly zero at t=0
#ckAI=[-0.00683011-0.0449112j, -(0.00683011-0.0449112j),0.00683011+0.00938383j, 0.00683011-0.00938383j]  
#vkAI=[-2.35315-1.04322j,-2.35315+1.04322j,-2.33632-3.21569j,-2.33632+3.21569j]

#vkAR=[-vk for vk in vkAR]    
#vkAI=[-vk for vk in vkAI]
#ckAR=[1.*ck for ck in ckAR]    
#ckAI=[1.*ck for ck in ckAI]


NR=2
NI=2
# Q = sigmax()
Q = sigmax()
# Q = [sigmax(), sigmax(), sigmax(), sigmax()]
# Q = sigmax()
Del =0.#np.pi/2.    
wq= 1.0     # Energy of the 2-level system.
Hsys = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()
# Hsys = 1

#tlist = np.linspace(0, 7.777777777777778, 36)
tlist = np.linspace(0, 200, 4000)
    
#for amirs Omega= pi data, tlist is different (actually a bit weird)
#tlist=tlistA
initial_state = basis(2,1) * basis(2,1).dag()                # Initial state of the system.


#return_vals = [tensor(qeye(N), kk) for kk in [Q]]            # List for which to calculate expectation value
return_vals = [initial_state, basis(2,0) * basis(2,1).dag()   ]            # List for which to calculate expectation value
eigen_sparse = False
calc_time = True                                             
options = Options(nsteps=15000, store_states=True,rtol=1e-12, atol=1e-12)        # Options for the solver.


#Convergence parameters
#we need very high Nc to get convergence.  might be interesting to understand why
Nc = input("nc: ")
Nc = int(Nc)

# results1Nc7 = a.hsolve(Hsys, Q,ckAR,ckAI,vkAR,vkAI, Nc, NR,NI, tlist, initial_state, options=options,\
    # progress_bar1=TextProgressBar(),progress_bar2=TextProgressBar())

# print("here")
# note that this expects the cks and vks as lists and not numpy arrays
# Hsys = liouvillian(Hsys)
def func(x):
    return 0.1*cos(x)
# Hsys = [Hsys, [sigmax(), func]]
# print(Hsys, Q,ckAR,ckAI,vkAR,vkAI, Nc)
# Hsys = 1
A = BosonicHEOMSolver(Hsys, Q,ckAR,ckAI,vkAR,vkAI, Nc, options=options)

# A = HSolverDL(Hsys, Q, lam, temperature, Ncut, Nk, gam, stats=True, renorm = False)
print("starting solve")
B = A.run(initial_state, tlist)

C, D = A.steady_state(Hsys, initial_state)

fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(tlist,expect(B.states,1-initial_state*initial_state.dag()))
ax1.plot(tlist,[expect(C,1-initial_state*initial_state.dag()) for t in tlist])

plt.show()
