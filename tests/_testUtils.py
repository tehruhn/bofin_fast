"""
Utility functions for tests
"""

from qutip import Qobj, sigmaz, sigmax, basis, expect, Options, destroy
from qutip.states import enr_state_dictionaries
import numpy as np
from numpy.linalg import eigvalsh 
from scipy.integrate import quad


def cot(x):
    """
    Computes cotangent for array.
    """
    return 1./np.tan(x)

def pure_dephasing_evolution_analytical(tlist, wq, ck, vk):
    """
    Analytically computes pure dephasing evolution.
    """
    evolution = np.array([np.exp(-1j*wq*t - correlation_integral(t, ck, 
        vk)) for t in tlist])
    return evolution

def correlation_integral(t, ck, vk):
    """
    Computes correlation integral.
    """
    t1 = np.sum(np.multiply(np.divide(ck, vk**2), np.exp(vk*t) - 1))
    t2 = np.sum(np.multiply(np.divide(np.conjugate(ck), \
        np.conjugate(vk)**2),np.exp(np.conjugate(vk)*t) - 1))
    t3 = np.sum((np.divide(ck, vk) + np.divide(np.conjugate(ck), 
        np.conjugate(vk)))*t)
    return 2*(t1 + t2 - t3)

def deltafun(n1, n2):
    """
    Returns Dirac delta of two integers.
    """
    delta = 1 if n1 == n2 else 0
    return delta

def get_aux_matrices(full, level, N_baths, Nk, N_cut, shape, dims):
    """
    Computes auxiliary density matrices.
    """
    nstates, state2idx, idx2state = \
        enr_state_dictionaries([2]*(Nk*N_baths) ,N_cut)
    aux_indices = []
    aux_heom_indices = []
    for stateid in state2idx:
        if np.sum(stateid) == level:
            aux_indices.append(state2idx[stateid])
            aux_heom_indices.append(stateid)
    full = np.array(full)
    aux = []
    for i in aux_indices:
        qlist = [Qobj(full[k, i, :].reshape(shape, shape).T,dims=dims) 
            for k in range(len(full))]
        aux.append(qlist)
    return aux, aux_heom_indices, idx2state

def Gamma_L_w(w):
    """
    Computes Gamma_L_w
    """
    return Gamma*W**2/((w - mu_l)**2 + W**2)

def Gamma_w(w, mu):
    """
    Computes Gamma_w.
    """
    return Gamma*W**2/((w - mu)**2 + W**2)