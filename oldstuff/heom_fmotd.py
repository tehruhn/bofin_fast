# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson,
#                      Neill Lambert, Anubhav Vardhan, Alexander Pitchford.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
This module provides exact solvers for a system-bath setup using the
hierarchy equations of motion (HEOM).
"""

# Authors: Neill Lambert, Tarun Raheja
# Contact: nwlambert@gmail.com

import timeit
import numpy as np
from math import sqrt, factorial
import scipy.sparse as sp
import scipy.integrate
from qutip import *
from qutip.superoperator import liouvillian
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result
from numpy import matrix, linalg
from qutip.cy.spmatfuncs import cy_ode_rhs
from cypadcsr import _pad_csr
from interfacer import boson_interfacecpp, fermion_interfacecpp


class BosonicHEOMSolver(object):
    """
    This is superclass for all solvers that use the HEOM method for
    calculating the dynamics evolution. There are many references for this.
    A good introduction, and perhaps closest to the notation used here is:
    DOI:10.1103/PhysRevLett.104.250401
    A more canonical reference, with full derivation is:
    DOI: 10.1103/PhysRevA.41.6676
    The method can compute open system dynamics without using any Markovian
    or rotating wave approximation (RWA) for systems where the bath
    correlations can be approximated to a sum of complex exponentials.
    The method builds a matrix of linked differential equations, which are
    then solved used the same ODE solvers as other qutip solvers (e.g. mesolve)
    Attributes
    ----------
    H_sys : Qobj or list 
        System Hamiltonian
        Or 
        Liouvillian
        Or 
        list of Hamiltonians with time dependence
        
        Format for input (if list):
        [time_independent_part, [H1, time_dep_function1], [H2, time_dep_function2]]

    coup_op : Qobj or list
        Operator describing the coupling between system and bath.
        Could also be a list of operators, which needs to be the same length as ck's and vk's.

    ckAR, ckAI, vkAR, vkAI : lists
        Lists containing coefficients for fitting spectral density correlation

    N_cut : int
        Cutoff parameter for the bath

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    def __init__(self, H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut,
                  options=None):

        self.reset()
        if options is None:
            self.options = Options()
        else:
            self.options = options
        # set other attributes
        self.configure(H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut, options)


    def reset(self):
        """
        Reset any attributes to default values
        """
        self.H_sys = None
        self.coup_op = None
        self.ckAR = []
        self.ckAI = []
        self.vkAR = []
        self.vkAI = []
        self.N_cut = 5
        self.options = None
        self.ode = None

    def configure(self, H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut,
                  options=None):
    
        """
        Configure the solver using the passed parameters
        The parameters are described in the class attributes, unless there
        is some specific behaviour

        Parameters
        ----------
        options : :class:`qutip.solver.Options`
            Generic solver options.
            If set to None the default options will be used
        """

        # some basic type checks

        if type(H_sys) != qutip.qutip.Qobj and type(H_sys) != list:
            raise RuntimeError("Hamiltonian is not specified in correct \
                format. Please specify either a Hamiltonian QObj, or a \
                Liouvillian, or a list of Hamiltonians with time \
                dependence as given in the documentation.")

        if type(H_sys) == list:
            size = len(H_sys)
            for i in range(0, size):
                if(i == 0):
                    if type(H_sys[i]) != qutip.qutip.Qobj:
                        raise RuntimeError("Hamiltonian is not specified in correct " +
                        "format. Please specify either a Hamiltonian QObj, or a " +
                        "Liouvillian, or a list of Hamiltonians with time " +
                        "dependence as given in the documentation.")
                else:
                    if type(H_sys[i][0]) != qutip.qutip.Qobj and type(H_sys[i][1])!= function:
                        raise RuntimeError("Hamiltonian is not specified in correct " +
                        "format. Please specify either a Hamiltonian QObj, or a " +
                        "Liouvillian, or a list of Hamiltonians with time " +
                        "dependence as given in the documentation.")

        if (type(coup_op) != qutip.qutip.Qobj) and (type(coup_op) == list and type(coup_op[0]) != qutip.qutip.Qobj):
            raise RuntimeError("Coupling operator must be a QObj or list of QObjs.")

        if type(coup_op) == list:
            if len(coup_op) != (len(ckAR) + len(ckAI)):
                raise RuntimeError("Expected " + str(len(ckAI) + len(ckAR)) + " coupling operators.")

        if type(ckAR[0]) == list or type(vkAR[0]) == list or type(ckAR[0]) == list or type(ckAI[0]) == list:
            raise RuntimeError("Lists of coefficients should be one dimensional.")

        if len(ckAR) != len(vkAR) or len(ckAI) != len(vkAI):
            raise RuntimeError("Spectral density correlation coefficients not " +
                "specified correctly.")

        # no two vk's should be same in same set

        for i in range(len(vkAR)):
            for j in range(i+1, len(vkAR)):
                if(np.isclose(vkAR[i], vkAR[j], rtol=1e-5, atol=1e-7)):
                    #raise RuntimeError("No two vk's should be same, as it can then be simplified.")
                    warnings.warn("Expected simplified input.")

        # no two vk's should be same in same set
        for i in range(len(vkAI)):
            for j in range(i+1, len(vkAI)):
                if(np.isclose(vkAI[i], vkAI[j], rtol=1e-5, atol=1e-7)):
                    #raise RuntimeError("No two vk's should be same, as it can then be simplified.")
                    warnings.warn("Expected simplified input.")
        self.H_sys = H_sys

        nr = len(ckAR)
        ni = len(ckAI)
        ckAR = list(ckAR)
        ckAI = list(ckAI)
        vkAR = list(vkAR)
        vkAI = list(vkAI)

        # handling the case where gammas might be the same
        # assumes that no 2 of the values in a vector can be the same

        if(type(coup_op) != list):
            coup_op = [coup_op for i in range(nr+ni)]

        common_ck = []
        real_indices = []
        common_vk = []
        img_indices = []
        common_coup_op = []
        #################################################################
        for i in range(len(vkAR)):
            for j in range(len(vkAI)):
                if(np.isclose(vkAR[i], vkAI[j], rtol=1e-5, atol=1e-7) and np.allclose(coup_op[i], coup_op[nr+j], rtol=1e-5, atol=1e-7)):                
                    print("SIMPL DUN")
                    common_ck.append(ckAR[i])
                    common_ck.append(ckAI[j])
                    common_vk.append(vkAR[i])
                    common_vk.append(vkAI[j])
                    real_indices.append(i)
                    img_indices.append(j)
                    common_coup_op.append(coup_op[i])

#this is just a fudge to get it to work for 7site case.  is not general        #EXPERIKMENTY
#
#        for i in range(len(vkAR)):

#                #coup_op[nr+j] is wrong
#                #this would be easier if we put params as list of lists

 #               if(np.isclose(vkAR[i], vkAI[i], rtol=1e-5, atol=1e-7)):# and np.allclose(coup_op[i], coup_op[nr+j], rtol=1e-5, atol=1e-7)):
 #                   print("SIMPL DUN")
 #                   common_ck.append(ckAR[i])
 #                   common_ck.append(ckAI[i])
 #                   common_vk.append(vkAR[i])
 #                   common_vk.append(vkAI[i])
 #                   real_indices.append(i)
 #                   img_indices.append(i)
 #                   common_coup_op.append(coup_op[i*2])
#################################################################################
        # print(sorted(real_indices, reverse=True))
        # print(sorted(img_indices, reverse=True))

        for i in sorted(real_indices, reverse=True):
            ckAR.pop(i)
            vkAR.pop(i)

        for i in sorted(img_indices, reverse=True):
            ckAI.pop(i)
            vkAI.pop(i)
        print(len(common_ck))
        # handling coup_op
        # remove real and imaginary ones

        img_coup_ops = [x+nr for x in img_indices]
        coup_op_indices = real_indices + sorted(img_coup_ops)
        # print(coup_op_indices)
        for i in sorted(coup_op_indices, reverse=True):
            coup_op.pop(i)
            
            
        ####################################################
        coup_op += common_coup_op  #EXPERIKMENTY
        #coup_op = common_coup_op
        ###################################################
        
        
        print(coup_op)
        print(len(coup_op))
        self.coup_op = coup_op
        self.ckAR = ckAR
        self.ckAI = ckAI
        self.vkAR = vkAR
        self.vkAI = vkAI
        self.common_ck = common_ck
        self.common_vk = common_vk
        self.N_cut = int(N_cut)
        if isinstance(options, Options): self.options = options
        # print("MUCH BEFORE")
        # print(ckAR)
        # print(ckAI)
        # print(vkAR)
        # print(vkAI)
        # print(common_ck)
        # print(common_vk)

        # setting variables locally

        options = self.options
        H = self.H_sys
        Q = self.coup_op
        Nc = self. N_cut
        ckAR = self.ckAR
        ckAI = self.ckAI
        vkAR = self.vkAR
        vkAI = self.vkAI
        common_ck = self.common_ck
        common_vk = self.common_vk
        NR = len(ckAR)
        NI = len(ckAI)

        # input cleaning for C++
        # passing list of coupling operators

        if type(Q) != list:
            Q = Q.data.toarray()
        else:
            Q1 = []
            for element in Q:
                element = element.data.toarray()
                Q1.append(element)
            Q = np.vstack(Q1)
            # print(Q.shape)

        ck = np.array(ckAR + ckAI + common_ck)
        vk = np.array(vkAR + vkAI + common_vk)

        # k is a flag variable that checks for type of input
        # k = 1 for hamiltonians
        # k = anything else for liouvillians

        k = 1 
        isListInput = False
        if type(H) is list:
            # treat like list of Hamiltonians
            Hsys = H[0].data.toarray()
            isListInput = True

        else:
            # check if given thing is a Liouvillian or a Hamiltonian
            Hsys = H.data.toarray()
            if H.type == 'oper':
                # Hamiltonian
                k = 1
            else:
                # Liouvillian
                k = 2

        # at the end of this, Hsys is supposed to have the right value
        # and the flag tells what kind of input was given

        # actual call to C++ interfacing code
        # computes the big sparse matrix very fast
        # print("TO PASS")
        # print(Hsys.flatten(),Hsys.shape[0])
        # print(Q.flatten(), Q.shape[0], Q.shape[1])
        # print(ck, ck.shape[0], vk, vk.shape[0])
        # print(Nc, NR, NI, k)

        Lbig2, N_he = boson_interfacecpp(Hsys.flatten(),Hsys.shape[0], 
                                    Q.flatten(), Q.shape[0], Q.shape[1], 
                                    ck, ck.shape[0], vk, vk.shape[0], Nc, NR, NI, k)
        # print("Nhe: ", N_he)
        # r is solver, which will be set based on condition
        r = None
        # specific things to do if list is input
        if isListInput:
            L_list = []
            constant_func = lambda x: 1.0
            # making list of all the big sparse matrices
            unit_helems = sp.identity(N_he, format='csr')
            L_list.append([Lbig2, constant_func])
            # for each time dependent hamiltonian component in list
            for i in range(1, len(H)):
                Ltemp = sp.kron(unit_helems, liouvillian(H[i][0]))
                L_list.append([Ltemp, H[i][1]])
            r = scipy.integrate.ode(_dsuper_list_td)
            r.set_f_params(L_list)
            # print("DID LIST")

        else:
            # setting up ODE solver for integration
            r = scipy.integrate.ode(cy_ode_rhs)
            # print("NO LIST")
            r.set_f_params(Lbig2.data, Lbig2.indices, Lbig2.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                     atol=options.atol, rtol=options.rtol,
                     nsteps=options.nsteps, first_step=options.first_step,
                     min_step=options.min_step,max_step=options.max_step)

        self._ode = r
        self._configured = True
        if k == 1 or isListInput:
            self._sup_dim = Hsys.shape[0] * Hsys.shape[0]
        else:
            self._sup_dim = int(sqrt(Hsys.shape[0])) * int(sqrt(Hsys.shape[0]))
        self._N_he = N_he
    
    def run(self, rho0, tlist):
        """
        Function to solve for an open quantum system using the
        HEOM model.

        Parameters
        ----------
        rho0 : Qobj
            Initial state (density matrix) of the system.

        tlist : list
            Time over which system evolves.

        Returns
        -------
        results : :class:`qutip.solver.Result`
            Object storing all results from the simulation.
        """

        start_run = timeit.default_timer()

        
        sup_dim = self._sup_dim
        
        r = self._ode

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        output.states.append(Qobj(rho0))
        # Using 'F' effectively transposes
        rho0_flat = rho0.full().ravel('F') 
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        r.set_initial_value(rho0_he, tlist[0])

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)

        return output
        
def _dsuper_list_td(t, y, L_list):
    """
    Auxiliary function for the integration.
    Is called at every time step.
    """
    L = L_list[0][0] 
    for n in range(1, len(L_list)):
        L = L + L_list[n][0] * L_list[n][1](t)
    return (L*y)



class FermionicHEOMSolver(object):
    """
    Same as above, but with Fermionic baths.

    Attributes
    ----------
    H_sys : Qobj or list 
        System Hamiltonian
        Or 
        Liouvillian
        Or 
        list of Hamiltonians with time dependence
        
        Format for input (if list):
        [time_independent_part, [H1, time_dep_function1], [H2, time_dep_function2]]

    coup_op : Qobj or list
        Operator describing the coupling between system and bath.
        Could also be a list of operators, which needs to be the same length as ck's and vk's.

    ck, vk : lists
        Lists containing spectral density correlation

    N_cut : int
        Cutoff parameter for the bath

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    # CHANGED
    def __init__(self, H_sys, coup_op, ck, vk, N_cut, options=None):

        self.reset()
        if options is None:
            self.options = Options()
        else:
            self.options = options
        # set other attributes
        self.configure(H_sys, coup_op, ck, vk, N_cut, options)


    # CHANGED
    def reset(self):
        """
        Reset any attributes to default values
        """
        self.H_sys = None
        self.coup_op = None
        self.ck = []
        self.vk = []
        self.N_cut = 10
        self.options = None
        self.ode = None

    def configure(self, H_sys, coup_op, ck, vk, N_cut, options=None):
    
        """
        Configure the solver using the passed parameters
        The parameters are described in the class attributes, unless there
        is some specific behaviour

        Parameters
        ----------
        options : :class:`qutip.solver.Options`
            Generic solver options.
            If set to None the default options will be used
        """

        # some basic type checks
        # CHANGED
        if type(H_sys) != qutip.qutip.Qobj and type(H_sys) != list:
            raise RuntimeError("Hamiltonian is not specified in correct \
                format. Please specify either a Hamiltonian QObj, or a \
                Liouvillian, or a list of Hamiltonians with time \
                dependence as given in the documentation.")

        # CHANGED
        if type(H_sys) == list:
            size = len(H_sys)
            for i in range(0, size):
                if(i == 0):
                    if type(H_sys[i]) != qutip.qutip.Qobj:
                        raise RuntimeError("Hamiltonian is not specified in correct " +
                        "format. Please specify either a Hamiltonian QObj, or a " +
                        "Liouvillian, or a list of Hamiltonians with time " +
                        "dependence as given in the documentation.")
                else:
                    if type(H_sys[i][0]) != qutip.qutip.Qobj and type(H_sys[i][1])!= function:
                        raise RuntimeError("Hamiltonian is not specified in correct" +
                        "format. Please specify either a Hamiltonian QObj, or a " +
                        "Liouvillian, or a list of Hamiltonians with time " +
                        "dependence as given in the documentation.")
        # CHANGED
        if (type(coup_op) == list and type(coup_op[0]) != qutip.qutip.Qobj):
            raise RuntimeError("Coupling operator must be a QObj or list of QObjs.")

        # put checks for cks and vks to be lists of lists and to have same number of elements
        # in corresponding parts

        # TODO

        self.H_sys = H_sys
        self.coup_op = coup_op
        self.ck = ck
        self.vk = vk
        self.N_cut = int(N_cut)
        if isinstance(options, Options): self.options = options

        # setting variables locally

        options = self.options
        H = self.H_sys
        Q = self.coup_op
        ck = self.ck
        vk = self.vk
        Nc = self. N_cut

        # input cleaning for C++

        # passing cks and vks
        len_list = [len(elem) for elem in ck]
        flat_ck = [elem for row in ck for elem in row]
        flat_vk = [elem for row in vk for elem in row]

        # passing list of coupling operators
        if type(Q) != list:
            Q = Q.data.toarray()
        else:
            Q1 = []
            for element in Q:
                element = element.data.toarray()
                Q1.append(element)
            Q = np.vstack(Q1)
            # print(Q.shape)

        # k is a flag variable that checks for type of input
        # k = 1 for hamiltonians
        # k = anything else for liouvillians
        k = 1 
        isListInput = False
        if type(H) is list:
            # treat like list of Hamiltonians
            Hsys = H[0].data.toarray()
            isListInput = True

        else:
            # check if given thing is a Liouvillian or a Hamiltonian
            Hsys = H.data.toarray()
            if H.type == 'oper':
                # Hamiltonian
                k = 1
            else:
                # Liouvillian
                k = 2

        # at the end of this, Hsys is supposed to have the right value
        # and the flag tells what kind of input was given

        # actual call to C++ interfacing code
        # computes the big sparse matrix very fast
        print("reached here")
        Lbig2, N_he = fermion_interfacecpp(Hsys.flatten(),Hsys.shape[0], 
                                    Q.flatten(), Q.shape[0], Q.shape[1], 
                                    np.array(flat_ck), len(flat_ck), np.array(flat_vk), 
                                    len(flat_vk), np.array(len_list, dtype=np.int32), len(len_list),
                                    Nc, k)
        # print("Nhe: ", N_he)
        # print(Lbig2.nonzero()[0].shape)
        # r is solver, which will be set based on condition
        r = None
        # specific things to do if list is input
        if isListInput:
            L_list = []
            constant_func = lambda x: 1.0
            # making list of all the big sparse matrices
            unit_helems = sp.identity(N_he, format='csr')
            L_list.append([Lbig2, constant_func])
            # for each time dependent hamiltonian component in list
            for i in range(1, len(H)):
                Ltemp = sp.kron(unit_helems, liouvillian(H[i][0]))
                L_list.append([Ltemp, H[i][1]])
            r = scipy.integrate.ode(_dsuper_list_td)
            r.set_f_params(L_list)

        else:
            # setting up ODE solver for integration
            r = scipy.integrate.ode(cy_ode_rhs)
            r.set_f_params(Lbig2.data, Lbig2.indices, Lbig2.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                     atol=options.atol, rtol=options.rtol,
                     nsteps=options.nsteps, first_step=options.first_step,
                     min_step=options.min_step,max_step=options.max_step)

        self._ode = r
        self._configured = True
        if k == 1 or isListInput:
            self._sup_dim = Hsys.shape[0] * Hsys.shape[0]
        else:
            self._sup_dim = int(sqrt(Hsys.shape[0])) * int(sqrt(Hsys.shape[0]))
        self._N_he = N_he
    
    def run(self, rho0, tlist):
        """
        Function to solve for an open quantum system using the
        HEOM model.

        Parameters
        ----------
        rho0 : Qobj
            Initial state (density matrix) of the system.

        tlist : list
            Time over which system evolves.

        Returns
        -------
        results : :class:`qutip.solver.Result`
            Object storing all results from the simulation.
        """

        start_run = timeit.default_timer()

        
        sup_dim = self._sup_dim
        
        r = self._ode

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        output.states.append(Qobj(rho0))
        # Using 'F' effectively transposes
        rho0_flat = rho0.full().ravel('F') 
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        r.set_initial_value(rho0_he, tlist[0])

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)

        return output
        