# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson,
#                      Neill Lambert, Alexander Pitchford.
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
from scipy.sparse.linalg import (use_solver, splu, spilu, spsolve, eigs,
                         LinearOperator, gmres, lgmres, bicgstab)
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode
from qutip.superoperator import vec2mat  


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
    H_sys : Qobj or QobjEvo 
        System Hamiltonian
        Or 
        Liouvillian
        Or 
        QobjEvo

    coup_op : Qobj or list
        Operator describing the coupling between system and bath.
        Could also be a list of operators, which needs to be the same length 
        as ck's and vk's.

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

    def process_input(self, H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut,
                  options=None):
        """
        Type-checks provided input
        Merges same gammas
        """

        # Checks for Hamiltonian

        if (type(H_sys) != qutip.qutip.Qobj and 
            type(H_sys) != qutip.qutip.QobjEvo and
            type(H_sys) != list):
            raise RuntimeError("Hamiltonian format is incorrect.")

        if type(H_sys) == list:
            size = len(H_sys)
            for i in range(0, size):
                if(i == 0):
                    if type(H_sys[i]) != qutip.qutip.Qobj:
                        raise RuntimeError("Hamiltonian format is incorrect.")
                else:
                    if (type(H_sys[i][0]) != qutip.qutip.Qobj and 
                        type(H_sys[i][1])!= function):
                        raise RuntimeError("Hamiltonian format is incorrect.")


        # Checks for coupling operator

        if ((type(coup_op) != qutip.qutip.Qobj) and 
           (type(coup_op) == list and type(coup_op[0]) != qutip.qutip.Qobj)):
            raise RuntimeError("Coupling operator must be a QObj or list "+
                " of QObjs.")

        if type(coup_op) == list:
            if len(coup_op) != (len(ckAR) + len(ckAI)):
                raise RuntimeError("Expected " + str(len(ckAI) + len(ckAR)) 
                    + " coupling operators.")

        # Checks for ckAR, ckAI, vkAR, vkAI

        if (type(ckAR) != list or type(vkAR) != list or 
            type(ckAR) != list or type(ckAI) != list):
            raise RuntimeError("Expected list for coefficients.")

        if (type(ckAR[0]) == list or type(vkAR[0]) == list or 
            type(ckAR[0]) == list or type(ckAI[0]) == list):
            raise RuntimeError("Lists of coefficients should be one " +
            "dimensional.")

        if len(ckAR) != len(vkAR) or len(ckAI) != len(vkAI):
            raise RuntimeError("Spectral density correlation coefficients not "+"specified correctly.")

        # Check that no two vk's should be same in same set
        for i in range(len(vkAR)):
            for j in range(i+1, len(vkAR)):
                if(np.isclose(vkAR[i], vkAR[j], rtol=1e-5, atol=1e-7)):
                    warnings.warn("Expected simplified input.")

        for i in range(len(vkAI)):
            for j in range(i+1, len(vkAI)):
                if(np.isclose(vkAI[i], vkAI[j], rtol=1e-5, atol=1e-7)):
                    warnings.warn("Expected simplified input.")
                    
        
        if type(H_sys) == list:
            self.H_sys = QobjEvo(H_sys) 
        else:
            self.H_sys = H_sys

        nr = len(ckAR)
        ni = len(ckAI)
        ckAR = list(ckAR)
        ckAI = list(ckAI)
        vkAR = list(vkAR)
        vkAI = list(vkAI)

        # Check to make list of coupling operators

        if(type(coup_op) != list):
            coup_op = [coup_op for i in range(nr+ni)]

        # Check for handling the case where gammas might be the same

        common_ck = []
        real_indices = []
        common_vk = []
        img_indices = []
        common_coup_op = []
        for i in range(len(vkAR)):
            for j in range(len(vkAI)):
                if(np.isclose(vkAR[i], vkAI[j], rtol=1e-5, atol=1e-7) and 
                   np.allclose(coup_op[i], coup_op[nr+j], rtol=1e-5, atol=1e-7)):
                    common_ck.append(ckAR[i])
                    common_ck.append(ckAI[j])
                    common_vk.append(vkAR[i])
                    common_vk.append(vkAI[j])
                    real_indices.append(i)
                    img_indices.append(j)
                    common_coup_op.append(coup_op[i])

        for i in sorted(real_indices, reverse=True):
            ckAR.pop(i)
            vkAR.pop(i)

        for i in sorted(img_indices, reverse=True):
            ckAI.pop(i)
            vkAI.pop(i)

        # Check to similarly truncate coupling operators

        img_coup_ops = [x+nr for x in img_indices]
        coup_op_indices = real_indices + sorted(img_coup_ops)
        for i in sorted(coup_op_indices, reverse=True):
            coup_op.pop(i)

        coup_op += common_coup_op

        # Assigns to attributes

        self.coup_op = coup_op
        self.ckAR = ckAR
        self.ckAI = ckAI
        self.vkAR = vkAR
        self.vkAI = vkAI
        self.common_ck = common_ck
        self.common_vk = common_vk
        self.N_cut = int(N_cut)
        if isinstance(options, Options): self.options = options


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

        # Type checks the input and truncates exponents if necessary

        self.process_input(H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut,
                  options=None)

        # Sets variables locally for configuring solver

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

        # Input reconfig for passing to C++

        # Passing stacked coupling operators
        Q = np.vstack([coupl_op.data.toarray() for coupl_op in Q])

        # Passing exponents 
        ck = np.array(ckAR + ckAI + common_ck).astype(complex)
        vk = np.array(vkAR + vkAI + common_vk).astype(complex)

        # Passing Hamiltonian

        # (also passed a flag which tells if Hamiltonian
        # is SuperOp or regular Qobj)
        isHamiltonian = True 
        isTimeDep = False

        if type(H) is qutip.qutip.QobjEvo:
            Hsys = H.to_list()[0].data.toarray()
            isTimeDep = True

        else:
            Hsys = H.data.toarray()
            if H.type == 'oper':
                isHamiltonian = True
            else:
                isHamiltonian = False

        # Flag for C++ indicating input is Hamiltonian
        isHam = 1 if isHamiltonian else 2

        # Passing data to C++ interfacer

        RHSmat, nstates = boson_interfacecpp(Hsys.flatten(),Hsys.shape[0], 
                          Q.flatten(), Q.shape[0], Q.shape[1], 
                          ck, ck.shape[0], vk, vk.shape[0], Nc, NR, NI, isHam)

        # Setting up solver

        solver = None

        if isTimeDep:

            solver_params = []
            constant_func = lambda x: 1.0
            h_identity_mat = sp.identity(nstates, format='csr')
            solver_params.append([RHSmat, constant_func])
            
            H_list = H.to_list()
            # Store each time dependent component
            for idx in range(1, len(H_list)):
                temp_mat = sp.kron(h_identity_mat, liouvillian(H_list[idx][0]))
                solver_params.append([temp_mat, H_list[idx][1]])

            solver = scipy.integrate.ode(_dsuper_list_td)
            solver.set_f_params(solver_params)

        else:

            solver = scipy.integrate.ode(cy_ode_rhs)
            solver.set_f_params(RHSmat.data, RHSmat.indices, RHSmat.indptr)

        # Sets options for solver

        solver.set_integrator('zvode', method=options.method, order=options.order,
                     atol=options.atol, rtol=options.rtol,
                     nsteps=options.nsteps, first_step=options.first_step,
                     min_step=options.min_step,max_step=options.max_step)

        # Sets attributes related to solver

        self._ode = solver
        self.RHSmat = RHSmat
        self._configured = True
        if isHamiltonian or isTimeDep:
            self._sup_dim = Hsys.shape[0] * Hsys.shape[0]
        else:
            self._sup_dim = int(sqrt(Hsys.shape[0])) * int(sqrt(Hsys.shape[0]))
        self._N_he = nstates

    def steady_state(self, H, rho0):
        """
        Computes steady state dynamics
        """

        nstates =  self._N_he
        sup_dim = self._sup_dim
        n = int(np.sqrt(sup_dim))
        unit_h_elems = sp.identity(nstates, format='csr')
        L = self.RHSmat# + sp.kron(unit_h_elems, 
                        #liouvillian(H).data)

        b_mat = np.zeros(sup_dim*nstates, dtype=complex)
        b_mat[0] = 1.

        L[0, 0 : n**2*nstates] = 0.
        L = L.tocsc() + \
            sp.csc_matrix((np.ones(n), (np.zeros(n), 
                          [num*(n+1)for num in range(n)])),
                          shape=(n**2*nstates, n**2*nstates))

        # Use superLU solver

        LU = splu(L)
        solution = LU.solve(b_mat)

        data = dense2D_to_fastcsr_fmode(vec2mat(solution[:sup_dim]), n, n)
        data = 0.5*(data + data.H)

        solution = solution.reshape((nstates, H.shape[0]**2))

        return Qobj(data, dims=rho0.dims), solution
    
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

        sup_dim = self._sup_dim
        
        solver = self._ode

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        output.states.append(Qobj(rho0))
        rho0_flat = rho0.full().ravel('F') 
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        solver.set_initial_value(rho0_he, tlist[0])

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                solver.integrate(solver.t + dt[t_idx])
                rho = Qobj(solver.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
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
    H_sys : Qobj or QobjEvo
        System Hamiltonian
        Or 
        Liouvillian
        Or 
        QobjEvo
        
    coup_op : Qobj or list
        Operator describing the coupling between system and bath.
        Could also be a list of operators, which needs to be the 
        same length as ck's and vk's.

    ck, vk : lists
        Lists containing spectral density correlation

    N_cut : int
        Cutoff parameter for the bath

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    def __init__(self, H_sys, coup_op, ck, vk, N_cut, options=None):

        self.reset()
        if options is None:
            self.options = Options()
        else:
            self.options = options
        self.configure(H_sys, coup_op, ck, vk, N_cut, options)

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

    def process_input(self, H_sys, coup_op, ck, vk, N_cut,
                  options=None):
        """
        Type-checks provided input
        Merges same gammas
        """

        # Checks for Hamiltonian

        if (type(H_sys) != qutip.qutip.Qobj and 
            type(H_sys) != qutip.qutip.QobjEvo and
            type(H_sys) != list):
            raise RuntimeError("Hamiltonian format is incorrect.")

        if type(H_sys) == list:
            size = len(H_sys)
            for i in range(0, size):
                if(i == 0):
                    if type(H_sys[i]) != qutip.qutip.Qobj:
                        raise RuntimeError("Hamiltonian format is incorrect.")
                else:
                    if (type(H_sys[i][0]) != qutip.qutip.Qobj and 
                        type(H_sys[i][1])!= function):
                        raise RuntimeError("Hamiltonian format is incorrect.")

        # Checks for cks and vks

        if (type(ck) != list or type(vk) != list or
            type(ck[0]) != list or type(vk[0]) != list):
            raise RuntimeError("Expected list of lists.")

        if len(ck) != len(vk):
            raise RuntimeError("Exponents supplied incorrectly.")

        for idx in range(len(ck)):
            if len(ck[idx]) != len(vk[idx]):
                raise RuntimeError("Exponents supplied incorrectly.")

        # Checks for coupling operator

        if ((type(coup_op) != qutip.qutip.Qobj) and 
           (type(coup_op) == list and type(coup_op[0]) != qutip.qutip.Qobj)):
            raise RuntimeError("Coupling operator must be a QObj or list "+
                " of QObjs.")

        if type(coup_op) == list:
            if (len(coup_op) != len(ck)):
                raise RuntimeError("Expected " + str(len(ck)) 
                    + " coupling operators.")

        # Make list of coupling operators

        if type(coup_op) != list:
            coup_op = [coup_op for elem in range(len(ck))]

        # TODO
        # more checks for coup ops and ck and vk

        if type(H_sys) == list:
            self.H_sys = QobjEvo(H_sys) 
        else:
            self.H_sys = H_sys
        self.coup_op = coup_op
        self.ck = ck
        self.vk = vk
        self.N_cut = int(N_cut)
        if isinstance(options, Options): self.options = options



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

        # Type check input
        self.process_input(H_sys, coup_op, ck, vk, N_cut,
                           options)

        # Setting variables locally

        options = self.options
        H = self.H_sys
        Q = self.coup_op
        ck = self.ck
        vk = self.vk
        Nc = self. N_cut

        # Input reconfig for passing to C++

        # Passing stacked coupling operators
        Q = np.vstack([coupl_op.data.toarray() for coupl_op in Q])

        # Passing Hamiltonian

        # (also passed a flag which tells if Hamiltonian
        # is SuperOp or regular Qobj)
        isHamiltonian = True 
        isTimeDep = False

        if type(H) is qutip.qutip.QobjEvo:
            Hsys = H.to_list()[0].data.toarray()
            isTimeDep = True

        else:
            Hsys = H.data.toarray()
            if H.type == 'oper':
                isHamiltonian = True
            else:
                isHamiltonian = False

        # Flag for C++ indicating input is Hamiltonian
        isHam = 1 if isHamiltonian else 2

        # Passing exponents
        len_list = [len(elem) for elem in ck]
        flat_ck = [elem for row in ck for elem in row]
        flat_vk = [elem for row in vk for elem in row]

        flat_ck = flat_ck.astype(complex)
        flat_vk = flat_vk.astype(complex)

        # Passing data to C++ interfacer

        RHSmat, nstates = fermion_interfacecpp(Hsys.flatten(),Hsys.shape[0], 
                                    Q.flatten(), Q.shape[0], Q.shape[1], 
                                    np.array(flat_ck), len(flat_ck), 
                                    np.array(flat_vk), 
                                    len(flat_vk), np.array(len_list,
                                     dtype=np.int32), len(len_list),
                                    Nc, isHam)
        # Setting up solver

        solver = None

        if isTimeDep:

            solver_params = []
            constant_func = lambda x: 1.0
            h_identity_mat = sp.identity(nstates, format='csr')
            solver_params.append([RHSmat, constant_func])
            H_list = H.to_list()
            # Store each time dependent component
            for idx in range(1, len(H_list)):
                temp_mat = sp.kron(h_identity_mat, liouvillian(H_list[i][0]))
                solver_params.append([temp_mat, H_list[i][1]])

            solver = scipy.integrate.ode(_dsuper_list_td)
            solver.set_f_params(solver_params)

        else:

            solver = scipy.integrate.ode(cy_ode_rhs)
            solver.set_f_params(RHSmat.data, RHSmat.indices, RHSmat.indptr)

        # Sets options for solver

        solver.set_integrator('zvode', method=options.method, order=options.order,
                     atol=options.atol, rtol=options.rtol,
                     nsteps=options.nsteps, first_step=options.first_step,
                     min_step=options.min_step,max_step=options.max_step)

        # Sets attributes related to solver

        self._ode = solver
        self.RHSmat = RHSmat
        self._configured = True
        if isHamiltonian or isTimeDep:
            self._sup_dim = Hsys.shape[0] * Hsys.shape[0]
        else:
            self._sup_dim = int(sqrt(Hsys.shape[0])) * int(sqrt(Hsys.shape[0]))
        self._N_he = nstates
    
    def steady_state(self, H, rho0):
        """
        Computes steady state dynamics
        """

        nstates =  self._N_he
        sup_dim = self._sup_dim
        n = int(np.sqrt(sup_dim))
        unit_h_elems = sp.identity(nstates, format='csr')
        L = self.RHSmat# + sp.kron(unit_h_elems, 
                        #liouvillian(H).data)

        b_mat = np.zeros(sup_dim*nstates, dtype=complex)
        b_mat[0] = 1.

        L[0, 0 : n**2*nstates] = 0.
        L = L.tocsc() + \
            sp.csc_matrix((np.ones(n), (np.zeros(n), 
                          [num*(n+1)for num in range(n)])),
                          shape=(n**2*nstates, n**2*nstates))

        # Use superLU solver

        LU = splu(L)
        solution = LU.solve(b_mat)

        data = dense2D_to_fastcsr_fmode(vec2mat(solution[:sup_dim]), n, n)
        data = 0.5*(data + data.H)

        solution = solution.reshape((nstates, H.shape[0]**2))

        return Qobj(data, dims=rho0.dims), solution

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

        sup_dim = self._sup_dim
        
        solver = self._ode

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        output.states.append(Qobj(rho0))
        rho0_flat = rho0.full().ravel('F') 
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        solver.set_initial_value(rho0_he, tlist[0])

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                solver.integrate(solver.t + dt[t_idx])
                rho = Qobj(solver.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)

        return output
        