from qutip import *
from heom_fmotd import BosonicHEOMSolver, FermionicHEOMSolver

import heom_fermions_pochen as heom_fermions

from numpy import pi
import numpy as np
from math import exp

from scipy.integrate import quad
import scipy as scipy
import matplotlib.pyplot as plt
#parameters and spectra check

Gamma = 0.01

#W = 0.01
W = 10**4
T = 0.025851991 #in ev
beta = 1./T

mu_l = -1.
mu_r = 1.

def Gamma_L_w(w):
    return Gamma*W**2/((w-mu_l)**2 + W**2)

def Gamma_R_w(w):
    return Gamma*W**2/((w-mu_r)**2 + W**2)


def f(x):
    kB=1.
    return 1/(np.exp(x)+1.)

def f2(x):
    return 0.5
    
lmax = 1
kappa = [0.]
kappa.extend([1. for l in range(1,lmax+1)])
epsilon = [0]
epsilon.extend([(2*l-1)*pi for l in range(1,lmax+1)])

def f_approx(x):
    f = 0.5
    for l in range(1,lmax+1):
        f= f - 2*kappa[l]*x/(x**2+epsilon[l]**2)
    return f


def C(tlist,sigma,mu):
    eta_list = []
    gamma_list  =[]
    
    #l = 0
    eta_0 = 0.5*Gamma*W*f(1.0j*beta*W)
    gamma_0 = W - sigma*1.0j*mu
    eta_list.append(eta_0)
    gamma_list.append(gamma_0)
    if lmax>0:
        for l in range(1,lmax+1):
            eta_list.append(-1.0j*(kappa[l]/beta)*Gamma*W**2/(-(epsilon[l]**2/beta**2)+W**2))
            gamma_list.append(epsilon[l]/beta - sigma*1.0j*mu)
    c_tot = []
    for t in tlist:
        c_tot.append(sum([eta_list[l]*np.exp(-gamma_list[l]*t) for l in range(lmax+1)]))
    return c_tot, eta_list, gamma_list

tlist = np.linspace(0,10,100)

#correlation terms
cppL,etapL,gampL = C(tlist,1.0,mu_l)

cpmL,etamL,gammL = C(tlist,-1.0,mu_l)

#mu_r = -mu_l

cppR,etapR,gampR = C(tlist,1.0,mu_r)

cpmR,etamR,gammR = C(tlist,-1.0,mu_r)

#system hamiltonian

d1 = destroy(2)

e1 = 1. 

H0 = e1*d1.dag()*d1 

Qops = [d1.dag(),d1,d1.dag(),d1]

rho_0 = basis(2,0)*basis(2,0).dag()

Kk=lmax+1
Ncc = 2




eta_list = [etapR,etamR,etapL,etamL]

gamma_list = [gampR,gammR,gampL,gammL]
# print(gamma_list)
print(len(eta_list))
import time
start = time.time()
resultHEOM1=heom_fermions.HSolverFermions(H0, [], Qops,  eta_list, gamma_list,  Ncc, Kk,renorm=False,bnd_cut_approx=False)
end = time.time()
print(12345)
print(end - start)

start = time.time()
out1M,full1M=resultHEOM1.run(rho_0,tlist)
end = time.time()


print(end - start)
# print(out1M)
print(expect(out1M.states[-1],d1.dag()*d1))


fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(tlist,expect(out1M.states,d1.dag()*d1))
#ax1.plot(tlist,[expect(C,1-initial_state*initial_state.dag()) for t in tlist])
plt.show()


#print(end - start)

#start = time.time()

#rhossHM,fullssM=resultHEOM1.ss(rho_0,H0)
#end = time.time()
#print(end - start)

