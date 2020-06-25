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

# Authors: Neill Lambert, Anubhav Vardhan, Alexander Pitchford
# Contact: nwlambert@gmail.com

import timeit
import numpy as np
#from scipy.misc import factorial
import scipy.sparse as sp
import scipy.integrate
from copy import copy
from qutip import Qobj, qeye
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost,mat2vec, vec2mat
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result, Stats
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar

    
def coth(x):
    """
    Calculates the coth function.
    
    Parameters
    ----------
    x: np.ndarray
        Any numpy array or list like input.
        
    Returns
    -------
    cothx: ndarray
        The coth function applied to the input.
    """
    return 1/np.tanh(x)
    
    
class HEOMSolver(object):
    """
    This is superclass for all solvers that use the HEOM method for
    calculating the dynamics evolution. There are many references for this.
    A good introduction, and perhaps closest to the notation used here is:
    DOI:10.1103/PhysRevLett.104.250401
    A more canonical reference, with full derivation is:
    DOI: 10.1103/PhysRevA.41.6676
    The method can compute open system dynamics without using any Markovian
    or rotating wave approximation (RWA) for systems where the bath
    correlations can be approximated to a sum of complex eponentials.
    The method builds a matrix of linked differential equations, which are
    then solved used the same ODE solvers as other qutip solvers (e.g. mesolve)

    This class should be treated as abstract. Currently the only subclass
    implemented is that for the Drude-Lorentz spectral density. This covers
    the majority of the work that has been done using this model, and there
    are some performance advantages to assuming this model where it is
    appropriate.

    There are opportunities to develop a more general spectral density code.

    Attributes
    ----------
    H_sys : Qobj
        System Hamiltonian

    coup_op : Qobj
        Operator describing the coupling between system and bath.

    coup_strength : float
        Coupling strength.

    temperature : float
        Bath temperature, in units corresponding to planck

    N_cut : int
        Cutoff parameter for the bath

    N_exp : int
        Number of exponential terms used to approximate the bath correlation
        functions

    planck : float
        reduced Planck constant

    boltzmann : float
        Boltzmann's constant

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used

    progress_bar: BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    stats : :class:`qutip.solver.Stats`
        optional container for holding performance statitics
        If None is set, then statistics are not collected
        There may be an overhead in collecting statistics

    exp_coeff : list of complex
        Coefficients for the exponential series terms

    exp_freq : list of complex
        Frequencies for the exponential series terms
    """
    def __init__(self):
        raise NotImplementedError("This is a abstract class only. "
                "Use a subclass, for example HSolverDL")

    def reset(self):
        """
        Reset any attributes to default values
        """
        self.planck = 1.0
        self.boltzmann = 1.0
        self.H_sys_td = None
        self.H_sys = None
        self.coup_op = None
        self.coup_strength = 0.0
        self.temperature = 1.0
        self.N_cut = 10
        self.N_exp = 2
        self.N_he = 0

        self.exp_coeff = None
        self.exp_freq = None

        self.options = None
        self.progress_bar = None
        self.stats = None

        self.ode = None
        self.configured = False

    def configure(self, H_sys,  H_sys_td, coup_op, coup_strength, temperature,
                     N_cut, N_exp, planck=None, boltzmann=None,
                     renorm=None, bnd_cut_approx=None,
                     options=None, progress_bar=None, stats=None):
        """
        Configure the solver using the passed parameters
        The parameters are described in the class attributes, unless there
        is some specific behaviour

        Parameters
        ----------
        options : :class:`qutip.solver.Options`
            Generic solver options.
            If set to None the default options will be used

        progress_bar: BaseProgressBar
            Optional instance of BaseProgressBar, or a subclass thereof, for
            showing the progress of the simulation.
            If set to None, then the default progress bar will be used
            Set to False for no progress bar

        stats: :class:`qutip.solver.Stats`
            Optional instance of solver.Stats, or a subclass thereof, for
            storing performance statistics for the solver
            If set to True, then the default Stats for this class will be used
            Set to False for no stats
        """

        self.H_sys = H_sys
        self.H_sys_td =  H_sys_td
        self.coup_op = coup_op
        self.coup_strength = coup_strength
        self.temperature = temperature
        self.N_cut = N_cut
        self.N_exp = N_exp
        if planck: self.planck = planck
        if boltzmann: self.boltzmann = boltzmann
        if isinstance(options, Options): self.options = options
        if isinstance(progress_bar, BaseProgressBar):
            self.progress_bar = progress_bar
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()
        elif progress_bar == False:
            self.progress_bar = None
        if isinstance(stats, Stats):
            self.stats = stats
        elif stats == True:
            self.stats = self.create_new_stats()
        elif stats == False:
            self.stats = None

    def create_new_stats(self):
        """
        Creates a new stats object suitable for use with this solver
        Note: this solver expects the stats object to have sections
            config
            integrate
        """
        stats = Stats(['config', 'run'])
        stats.header = "Hierarchy Solver Stats"
        return stats

class HSolverDL(HEOMSolver):
    """
    HEOM solver based on the Drude-Lorentz model for spectral density.
    Drude-Lorentz bath the correlation functions can be exactly analytically
    expressed as an infinite sum of exponentials which depend on the
    temperature, these are called the Matsubara terms or Matsubara frequencies

    For practical computation purposes an approximation must be used based
    on a small number of Matsubara terms (typically < 4).

    Attributes
    ----------
    cut_freq : float
        Bath spectral density cutoff frequency.

    renorm : bool
        Apply renormalisation to coupling terms
        Can be useful if using SI units for planck and boltzmann

    bnd_cut_approx : bool
        Use boundary cut off approximation
        Can be
    """

    def __init__(self, H_sys, H_sys_td, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=1.0, boltzmann=1.0,
                     renorm=True, bnd_cut_approx=True,
                     options=None, progress_bar=None, stats=None):

        self.reset()

        if options is None:
            self.options = Options()
        else:
            self.options = options

        self.progress_bar = False
        if progress_bar is None:
            self.progress_bar = BaseProgressBar()
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()

        # the other attributes will be set in the configure method
        self.configure(H_sys,   H_sys_td, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=planck, boltzmann=boltzmann,
                     renorm=renorm, bnd_cut_approx=bnd_cut_approx, stats=stats)

    def reset(self):
        """
        Reset any attributes to default values
        """
        HEOMSolver.reset(self)
        self.cut_freq = 1.0
        self.renorm = False
        self.bnd_cut_approx = False

    def configure(self, H_sys,  H_sys_td, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=None, boltzmann=None,
                     renorm=None, bnd_cut_approx=None,
                     options=None, progress_bar=None, stats=None):
        """
        Calls configure from :class:`HEOMSolver` and sets any attributes
        that are specific to this subclass
        """
        start_config = timeit.default_timer()

        HEOMSolver.configure(self, H_sys,  0,  coup_op, coup_strength,
                    temperature, N_cut, N_exp,
                    planck=planck, boltzmann=boltzmann,
                    options=options, progress_bar=progress_bar, stats=stats)
        self.cut_freq = cut_freq
        if renorm is not None: self.renorm = renorm
        if bnd_cut_approx is not None: self.bnd_cut_approx = bnd_cut_approx

        # Load local values for optional parameters
        # Constants and Hamiltonian.
        hbar = self.planck
        options = self.options
        progress_bar = self.progress_bar
        stats = self.stats


        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                ss_conf = stats.add_section('config')

        c, nu = self._calc_matsubara_params()

        if renorm:
            norm_plus, norm_minus = self._calc_renorm_factors()
            if stats:
                stats.add_message('options', 'renormalisation', ss_conf)
        # Dimensions et by system
        sup_dim = H_sys.dims[0][0]**2
        unit_sys = qeye(H_sys.dims[0])

        # Use shorthands (mainly as in referenced PRL)
        lam0 = self.coup_strength
        gam = self.cut_freq
        N_c = self.N_cut
        N_m = self.N_exp
        Q = coup_op # Q as shorthand for coupling operator
        beta = 1.0/(self.boltzmann*self.temperature)

        # Ntot is the total number of ancillary elements in the hierarchy
        # Ntot = factorial(N_c + N_m) / (factorial(N_c)*factorial(N_m))
        # Turns out to be the same as nstates from state_number_enumerate
        N_he, he2idx, idx2he = enr_state_dictionaries([N_c + 1]*N_m , N_c)

        unit_helems = sp.identity(N_he, format='csr')
        if self.bnd_cut_approx:
            # the Tanimura boundary cut off operator
            if stats:
                stats.add_message('options', 'boundary cutoff approx', ss_conf)
            op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)

            approx_factr = ((2*lam0 / (beta*gam*hbar)) - 1j*lam0) / hbar
            for k in range(N_m):
                approx_factr -= (c[k] / nu[k])
            L_bnd = -approx_factr*op.data
            L_helems = sp.kron(unit_helems, L_bnd)
        else:
            L_helems = sp.csr_matrix((N_he*sup_dim, N_he*sup_dim),
                                     dtype=complex)

        # Build the hierarchy element interaction matrix
        if stats: start_helem_constr = timeit.default_timer()

        unit_sup = spre(unit_sys).data
        spreQ = spre(Q).data
        spostQ = spost(Q).data
        commQ = (spre(Q) - spost(Q)).data
        N_he_interact = 0

        for he_idx in range(N_he):
            he_state = list(idx2he[he_idx])
            n_excite = sum(he_state)

            # The diagonal elements for the hierarchy operator
            # coeff for diagonal elements
            sum_n_m_freq = 0.0
            for k in range(N_m):
                sum_n_m_freq += he_state[k]*nu[k]

            op = -sum_n_m_freq*unit_sup
            L_he = _pad_csr(op, N_he, N_he, he_idx, he_idx)
            L_helems += L_he

            # Add the neighour interations
            he_state_neigh = copy(he_state)
            for k in range(N_m):

                n_k = he_state[k]
                if n_k >= 1:
                    # find the hierarchy element index of the neighbour before
                    # this element, for this Matsubara term
                    he_state_neigh[k] = n_k - 1
                    he_idx_neigh = he2idx[tuple(he_state_neigh)]

                    op = c[k]*spreQ - np.conj(c[k])*spostQ
                    if renorm:
                        op = -1j*norm_minus[n_k, k]*op
                    else:
                        op = -1j*n_k*op

                    L_he = _pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                    L_helems += L_he
                    N_he_interact += 1

                    he_state_neigh[k] = n_k

                if n_excite <= N_c - 1:
                    # find the hierarchy element index of the neighbour after
                    # this element, for this Matsubara term
                    he_state_neigh[k] = n_k + 1
                    he_idx_neigh = he2idx[tuple(he_state_neigh)]

                    op = commQ
                    if renorm:
                        op = -1j*norm_plus[n_k, k]*op
                    else:
                        op = -1j*op

                    L_he = _pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                    L_helems += L_he
                    N_he_interact += 1

                    he_state_neigh[k] = n_k

        if stats:
            stats.add_timing('hierarchy contruct',
                             timeit.default_timer() - start_helem_constr,
                            ss_conf)
            stats.add_count('Num hierarchy elements', N_he, ss_conf)
            stats.add_count('Num he interactions', N_he_interact, ss_conf)

        # Setup Liouvillian
        if stats: start_louvillian = timeit.default_timer()
        H_he = sp.kron(unit_helems, liouvillian(H_sys).data)

        L_helems += H_he

        if stats:
            stats.add_timing('Liouvillian contruct',
                             timeit.default_timer() - start_louvillian,
                            ss_conf)

        if stats: start_integ_conf = timeit.default_timer()

        r = scipy.integrate.ode(cy_ode_rhs)

        r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('Liouvillian contruct',
                             time_now - start_integ_conf,
                            ss_conf)
            if ss_conf.total_time is None:
                ss_conf.total_time = time_now - start_config
            else:
                ss_conf.total_time += time_now - start_config

        self._ode = r
        self._N_he = N_he
        self._sup_dim = sup_dim
        self._configured = True

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
        stats = self.stats
        r = self._ode

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")
        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                raise RuntimeError("No config section for solver stats")
            ss_run = stats.sections.get('run')
            if ss_run is None:
                ss_run = stats.add_section('run')

        # Set up terms of the matsubara and tanimura boundaries
        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        if stats: start_init = timeit.default_timer()
        output.states.append(Qobj(rho0))
        rho0_flat = rho0.full().ravel('F') # Using 'F' effectively transposes
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        r.set_initial_value(rho0_he, tlist[0])

        if stats:
            stats.add_timing('initialize',
                             timeit.default_timer() - start_init, ss_run)
            start_integ = timeit.default_timer()

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('integrate',
                             time_now - start_integ, ss_run)
            if ss_run.total_time is None:
                ss_run.total_time = time_now - start_run
            else:
                ss_run.total_time += time_now - start_run
            stats.total_time = ss_conf.total_time + ss_run.total_time

        return output

    def _calc_matsubara_params(self):
        """
        Calculate the Matsubara coefficents and frequencies

        Returns
        -------
        c, nu: both list(float)

        """
        c = []
        nu = []
        lam0 = self.coup_strength
        gam = self.cut_freq
        hbar = self.planck
        beta = 1.0/(self.boltzmann*self.temperature)
        N_m = self.N_exp

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

        self.exp_coeff = c
        self.exp_freq = nu
        return c, nu

    def _calc_renorm_factors(self):
        """
        Calculate the renormalisation factors

        Returns
        -------
        norm_plus, norm_minus : array[N_c, N_m] of float
        """
        c = self.exp_coeff
        N_m = self.N_exp
        N_c = self.N_cut

        norm_plus = np.empty((N_c+1, N_m))
        norm_minus = np.empty((N_c+1, N_m))
        for k in range(N_m):
            for n in range(N_c+1):
                norm_plus[n, k] = np.sqrt(abs(c[k])*(n + 1))
                norm_minus[n, k] = np.sqrt(float(n)/abs(c[k]))

        return norm_plus, norm_minus

		
class HSolverDLMultiBaths(HEOMSolver):
    """
    HEOM solver based on the Drude-Lorentz model for spectral density.
    Drude-Lorentz bath the correlation functions can be exactly analytically
    expressed as an infinite sum of exponentials which depend on the
    temperature, these are called the Matsubara terms or Matsubara frequencies

    For practical computation purposes an approximation must be used based
    on a small number of Matsubara terms (typically < 4).
	
	This version is generalized to multiple baths, as determined by the number of operators
	in coup_op, which is now a list
	
	Note: I have not fixed the renormalization terms, the boundary contition stuff, or the matsubara parameters.
	to make it totally general this needs to be done.
	

    Attributes
    ----------
    cut_freq : float
        Bath spectral density cutoff frequency.

    renorm : bool
        Apply renormalisation to coupling terms
        Can be useful if using SI units for planck and boltzmann

    bnd_cut_approx : bool
        Use boundary cut off approximation
        Can be
    """

    def __init__(self, H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=1.0, boltzmann=1.0,
                     renorm=True, bnd_cut_approx=True,
                     options=None, progress_bar=None, stats=None):

        self.reset()

        if options is None:
            self.options = Options()
        else:
            self.options = options

        self.progress_bar = False
        if progress_bar is None:
            self.progress_bar = BaseProgressBar()
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()

        # the other attributes will be set in the configure method
        self.configure(H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=planck, boltzmann=boltzmann,
                     renorm=renorm, bnd_cut_approx=bnd_cut_approx, stats=stats)

    def reset(self):
        """
        Reset any attributes to default values
        """
        HEOMSolver.reset(self)
        self.cut_freq = 1.0
        self.renorm = False
        self.bnd_cut_approx = False

    def configure(self, H_sys, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq, planck=None, boltzmann=None,
                     renorm=None, bnd_cut_approx=None,
                     options=None, progress_bar=None, stats=None):
        """
        Calls configure from :class:`HEOMSolver` and sets any attributes
        that are specific to this subclass
        """
        start_config = timeit.default_timer()

        HEOMSolver.configure(self, H_sys, coup_op, coup_strength,
                    temperature, N_cut, N_exp,
                    planck=planck, boltzmann=boltzmann,
                    options=options, progress_bar=progress_bar, stats=stats)
        self.cut_freq = cut_freq
       
        if renorm is not None: self.renorm = renorm
        if bnd_cut_approx is not None: self.bnd_cut_approx = bnd_cut_approx

        # Load local values for optional parameters
        # Constants and Hamiltonian.
        hbar = self.planck
        options = self.options
        progress_bar = self.progress_bar
        stats = self.stats


        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                ss_conf = stats.add_section('config')

        c, nu = self._calc_matsubara_params()

        if renorm:
            norm_plus, norm_minus = self._calc_renorm_factors()
            if stats:
                stats.add_message('options', 'renormalisation', ss_conf)
        # Dimensions et by system
        sup_dim = H_sys.dims[0][0]**2
        unit_sys = qeye(H_sys.dims[0])

        # Use shorthands (mainly as in referenced PRL)
        lam0 = self.coup_strength
        gam = self.cut_freq
        N_c = self.N_cut 
        N_m = self.N_exp 
        Q = coup_op # Q as shorthand for coupling operator
        beta = [1.0/(self.boltzmann*T) for T in self.temperature]  ###MIGHT Need this if we generalize temperature to a list 
        #beta = 1.0/(self.boltzmann*self.temperature)
        
        N_baths = len(coup_op)
        # Ntot is the total number of ancillary elements in the hierarchy
		#########each bath has Nm matsubara terms.... thus we just multiply N_baths times N_m
        # Ntot = factorial(N_c + N_m) / (factorial(N_c)*factorial(N_m))
        # Turns out to be the same as nstates from state_number_enumerate
        N_he, he2idx, idx2he = enr_state_dictionaries([N_c + 1]*(N_m*N_baths) , N_c)

        unit_helems = sp.identity(N_he, format='csr')
        #####slightly generalized this, just to make it work, but needs checking
        if self.bnd_cut_approx:
            # the Tanimura boundary cut off operator
            if stats:
                stats.add_message('options', 'boundary cutoff approx', ss_conf)
            L_bnd=0.*spre(Q[0]).data #####this is stupid, sorry
            for n,Qn in enumerate(Q):
                op = -2*spre(Qn)*spost(Qn.dag()) + spre(Qn.dag()*Qn) + spost(Qn.dag()*Qn)

                approx_factr = ((2*lam0[n] / (beta[n]*gam[n]*hbar)) - 1j*lam0[n]) / hbar
                for k in range(N_m):
                    approx_factr -= (c[k,n] / nu[k,n])
                L_bnd -= approx_factr*op.data
            L_helems = sp.kron(unit_helems, L_bnd)
        else:
            L_helems = sp.csr_matrix((N_he*sup_dim, N_he*sup_dim),
                                     dtype=complex)

        # Build the hierarchy element interaction matrix
        if stats: start_helem_constr = timeit.default_timer()

        unit_sup = spre(unit_sys).data
        spreQ = [spre(Qn).data for Qn in Q]
        spostQ = [spost(Qn).data for Qn in Q]
        commQ = [(spre(Qn) - spost(Qn)).data for Qn in Q]
        N_he_interact = 0

        for he_idx in range(N_he):
            he_state = list(idx2he[he_idx])
            n_excite = sum(he_state)

            # The diagonal elements for the hierarchy operator
            # coeff for diagonal elements
            sum_n_m_freq = 0.0
            for n in range(N_baths):
                for k in range(N_m):
                    sum_n_m_freq += he_state[k+n*N_m]*nu[k,n]

            op = -sum_n_m_freq*unit_sup
            L_he = _pad_csr(op, N_he, N_he, he_idx, he_idx)
            #print sup_dim
            #print N_he
            #print N_he*sup_dim
            #print L_he.shape[0]
            #print L_helems.shape[0]
            L_helems += L_he

            # Add the neighour interations
            he_state_neigh = copy(he_state)
            for n in range(N_baths):
                for k in range(N_m):

                    n_k = he_state[k+n*N_m]
                    if n_k >= 1:
                        # find the hierarchy element index of the neighbour before
                        # this element, for this Matsubara term and bath
                        he_state_neigh[k+n*N_m] = n_k - 1
                        he_idx_neigh = he2idx[tuple(he_state_neigh)]

                        op = c[k,n]*spreQ[n]- np.conj(c[k,n])*spostQ[n]
                        if renorm:
                            op = -1j*norm_minus[n_k, k,n]*op
                        else:
                            op = -1j*n_k*op

                        L_he = _pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                        L_helems += L_he
                        N_he_interact += 1

                        he_state_neigh[k+n*N_m] = n_k

                    if n_excite <= N_c - 1:
                        # find the hierarchy element index of the neighbour after
                        # this element, for this Matsubara term
                        he_state_neigh[k+n*N_m] = n_k + 1
                        he_idx_neigh = he2idx[tuple(he_state_neigh)]

                        op = commQ[n]
                        if renorm:
                            op = -1j*norm_plus[n_k, k,n]*op
                        else:
                            op = -1j*op

                        L_he = _pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                        L_helems += L_he
                        N_he_interact += 1

                        he_state_neigh[k+n*N_m] = n_k

        if stats:
            stats.add_timing('hierarchy contruct',
                             timeit.default_timer() - start_helem_constr,
                            ss_conf)
            stats.add_count('Num hierarchy elements', N_he, ss_conf)
            stats.add_count('Num he interactions', N_he_interact, ss_conf)

        # Setup Liouvillian
        if stats: start_louvillian = timeit.default_timer()
        H_he = sp.kron(unit_helems, liouvillian(H_sys).data)

        L_helems += H_he

        if stats:
            stats.add_timing('Liouvillian contruct',
                             timeit.default_timer() - start_louvillian,
                            ss_conf)

        if stats: start_integ_conf = timeit.default_timer()

        r = scipy.integrate.ode(cy_ode_rhs)

        r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('Liouvillian contruct',
                             time_now - start_integ_conf,
                            ss_conf)
            if ss_conf.total_time is None:
                ss_conf.total_time = time_now - start_config
            else:
                ss_conf.total_time += time_now - start_config

        self._ode = r
        self._N_he = N_he
        self._sup_dim = sup_dim
        self._configured = True

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
        stats = self.stats
        r = self._ode

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")
        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                raise RuntimeError("No config section for solver stats")
            ss_run = stats.sections.get('run')
            if ss_run is None:
                ss_run = stats.add_section('run')

        # Set up terms of the matsubara and tanimura boundaries
        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        if stats: start_init = timeit.default_timer()
        output.states.append(Qobj(rho0))
        rho0_flat = rho0.full().ravel('F') # Using 'F' effectively transposes
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        r.set_initial_value(rho0_he, tlist[0])

        if stats:
            stats.add_timing('initialize',
                             timeit.default_timer() - start_init, ss_run)
            start_integ = timeit.default_timer()

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('integrate',
                             time_now - start_integ, ss_run)
            if ss_run.total_time is None:
                ss_run.total_time = time_now - start_run
            else:
                ss_run.total_time += time_now - start_run
            stats.total_time = ss_conf.total_time + ss_run.total_time

        return output

    def _calc_matsubara_params(self):
        """
        Calculate the Matsubara coefficents and frequencies

        Returns
        -------
        c, nu: array[N_m,N_baths] of complex_ float
        """
        N_m = self.N_exp
        N_c = self.N_cut
        lam0 = self.coup_strength
        gam = self.cut_freq
        hbar = self.planck
        N_baths=len(lam0)
        beta =[ 1.0/(self.boltzmann*T) for T in self.temperature]
        
        
        
        c = np.empty((N_m,N_baths),dtype=np.complex_)
        nu = np.empty((N_m,N_baths),dtype=np.complex_)
        
        g = [2*np.pi / (b*hbar) for b in beta]
        for n in range(N_baths):
            for k in range(N_m):
                if k == 0:
                    nu[k,n]=(gam[n])
                    c[k,n]=(lam0[n]*gam[n]*
                        (1.0/np.tan(gam[n]*hbar*beta[n]/2.0) - 1j) / hbar)
                else:
                    nu[k,n]=(k*g[n])
                    c[k,n]=(4*lam0[n]*gam[n]*nu[k,n] /
                          ((nu[k,n]**2 - gam[n]**2)*beta[n]*hbar**2))

        self.exp_coeff = c
        self.exp_freq = nu
        return c, nu

    def _calc_renorm_factors(self):
        """
        Calculate the renormalisation factors

        Returns
        -------
        norm_plus, norm_minus : array[N_c, N_m,N_baths] of float
        """
        c = self.exp_coeff
        N_m = self.N_exp
        N_c = self.N_cut
        N_baths=len(self.coup_strength)
        norm_plus = np.empty((N_c+1, N_m,N_baths))
        norm_minus = np.empty((N_c+1, N_m,N_baths))
        for nb in range(N_baths):
            for k in range(N_m):
                for n in range(N_c+1):
                    norm_plus[n, k,nb] = np.sqrt(abs(c[k,nb])*(n + 1))
                    norm_minus[n, k,nb] = np.sqrt(float(n)/abs(c[k,nb]))

        return norm_plus, norm_minus

class HSolverDLMultiBathsWithRes(HEOMSolver):
    """
    HEOM solver based on the Drude-Lorentz model for spectral density.
    Drude-Lorentz bath the correlation functions can be exactly analytically
    expressed as an infinite sum of exponentials which depend on the
    temperature, these are called the Matsubara terms or Matsubara frequencies

    For practical computation purposes an approximation must be used based
    on a small number of Matsubara terms (typically < 4).
	
	This version is generalized to multiple baths, as determined by the number of operators
	in coup_op, which is now a list
	
	Note: I have not fixed the renormalization terms, the boundary contition stuff, or the matsubara parameters.
	to make it totally general this needs to be done.
	

    Attributes
    ----------
    cut_freq : float
        Bath spectral density cutoff frequency.

    renorm : bool
        Apply renormalisation to coupling terms
        Can be useful if using SI units for planck and boltzmann

    bnd_cut_approx : bool
        Use boundary cut off approximation
        Can be
    """


    def __init__(self, H_sys, H_sys_td, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq,res_list,ck_corr = None,vk_corr = None,  planck=1.0, boltzmann=1.0,
                     renorm=True, bnd_cut_approx=True,
                     options=None, progress_bar=None, stats=None,op_other=None):

        self.reset()

        if options is None:
            self.options = Options()
        else:
            self.options = options

        self.progress_bar = False
        if progress_bar is None:
            self.progress_bar = BaseProgressBar()
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()

        # the other attributes will be set in the configure method
        self.configure(H_sys,  H_sys_td, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq,res_list,ck_corr = ck_corr,vk_corr = vk_corr,  planck=planck, boltzmann=boltzmann,
                     renorm=renorm, bnd_cut_approx=bnd_cut_approx, stats=stats,op_other=op_other)

    def reset(self):
        """
        Reset any attributes to default values
        """
        HEOMSolver.reset(self)
        self.cut_freq = 1.0
        self.renorm = False
        self.bnd_cut_approx = False

    def configure(self, H_sys, H_sys_td, coup_op, coup_strength, temperature,
                     N_cut, N_exp, cut_freq,res_list, ck_corr = None,vk_corr = None, planck=None, boltzmann=None,
                     renorm=None, bnd_cut_approx=None,
                     options=None, progress_bar=None, stats=None,op_other=None):
        """
        Calls configure from :class:`HEOMSolver` and sets any attributes
        that are specific to this subclass
        """
        start_config = timeit.default_timer()

        HEOMSolver.configure(self, H_sys, H_sys_td, coup_op, coup_strength,
                    temperature, N_cut, N_exp,
                    planck=planck, boltzmann=boltzmann,
                    options=options, progress_bar=progress_bar, stats=stats)
        self.cut_freq = cut_freq
        self.res_list = res_list
        self.ck_corr = ck_corr
        self.vk_corr = vk_corr
        
        if renorm is not None: self.renorm = renorm
        if bnd_cut_approx is not None: self.bnd_cut_approx = bnd_cut_approx

        # Load local values for optional parameters
        # Constants and Hamiltonian.
        hbar = self.planck
        options = self.options
        progress_bar = self.progress_bar
        stats = self.stats


        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                ss_conf = stats.add_section('config')

        c, nu = self._calc_matsubara_params()

        if renorm:
            norm_plus, norm_minus = self._calc_renorm_factors()
            if stats:
                stats.add_message('options', 'renormalisation', ss_conf)
        # Dimensions et by system
        sup_dim = H_sys.dims[0][0]**2
        unit_sys = qeye(H_sys.dims[0])

        # Use shorthands (mainly as in referenced PRL)
        lam0 = self.coup_strength
        gam = self.cut_freq
        N_c = self.N_cut 
        N_m = self.N_exp 
        w0_list = res_list
        Q = coup_op # Q as shorthand for coupling operator
        beta = []
        for T in self.temperature:
            if T == 0:
                beta.append(np.inf)
            else:
                beta.append(1.0/(self.boltzmann*T))
        #beta = [1.0/(self.boltzmann*T) for T in self.temperature]  ###MIGHT Need this if we generalize temperature to a list 
        #beta = 1.0/(self.boltzmann*self.temperature)
        
        N_baths = len(coup_op)
       
            
        # Ntot is the total number of ancillary elements in the hierarchy
        # the coeffiecients and expononents for these are stored in c_corr and vk_corr etc
        
		#########each bath has Nm matsubara terms.... thus we just multiply N_baths times N_m
        # Ntot = factorial(N_c + N_m) / (factorial(N_c)*factorial(N_m))
        # Turns out to be the same as nstates from state_number_enumerate
        
        #
        ######note: because i dont properly do the matsubara terms for the underdamped baths, this only works for N_m,N_exp=1.
        N_he, he2idx, idx2he = enr_state_dictionaries([N_c + 1]*(N_m*N_baths) , N_c)

        unit_helems = sp.identity(N_he, format='csr')
        #####slightly generalized this, just to make it work, but needs checking
        
        L_helems = sp.csr_matrix((N_he*sup_dim, N_he*sup_dim),
                                     dtype=complex)
        #I am removing this for now
        
        if self.bnd_cut_approx:
            # the Tanimura boundary cut off operator
            if stats:
                stats.add_message('options', 'boundary cutoff approx', ss_conf)
            L_bnd=0.*spre(Q[0]).data #####this is stupid, sorry
            for n,Qn in enumerate(Q):
                if w0_list[n]==0:
                    op = -2*spre(Qn)*spost(Qn.dag()) + spre(Qn.dag()*Qn) + spost(Qn.dag()*Qn)

                    approx_factr = ((2*lam0[n] / (beta[n]*gam[n]*hbar)) - 1j*lam0[n]) / hbar
                    for k in range(N_m):
                        approx_factr -= (c[k,n] / nu[k,n])
                    L_bnd -= approx_factr*op.data
            #This dosent work so removing it for now    
            #it gives a negative rate;  is this unphysical?

#                else:
#                    op = -2*spre(Qn)*spost(Qn.dag()) + spre(Qn.dag()*Qn) + spost(Qn.dag()*Qn)
#                    bet = beta[n]
#                    Gam=gam[n]*0.5
#                    Om=np.sqrt(np.imag(w0_list[n])**2-Gam**2)
#                    lam1=np.real(lam0[n])
#                    approx_factr =(lam1**2*(4*Gam*Om*np.cos(bet*Gam) - 4*Gam*Om*np.cosh(bet*Om)+bet*(Gam**2 + Om**2)*(Om*np.sin(bet*Gam)+Gam*np.sinh(bet*Om)))) / (2*bet*Om*(Gam**2+Om**2)**2*(np.cos(bet*Gam)-np.cosh(bet*Om)))
#                    print(approx_factr)
                    
                    
                    #L_bnd -= approx_factr*op.data    
           
                    
            L_helems = sp.kron(unit_helems, L_bnd)
            if op_other: #overwrite default with additional lindblad
                            #this allows us to treat some baths with a simple lindblad
                            #and others with heom.
                            #a total bodge, but useful
                L_helems = sp.kron(unit_helems, L_bnd+op_other.data)
        else:
            L_helems = sp.csr_matrix((N_he*sup_dim, N_he*sup_dim),
                                     dtype=complex)
            if op_other: #overwrite default with additional lindblad
                L_helems = sp.kron(unit_helems, op_other.data)                         
        
        
        # Build the hierarchy element interaction matrix
        if stats: start_helem_constr = timeit.default_timer()

        unit_sup = spre(unit_sys).data
        spreQ = [spre(Qn).data for Qn in Q]
        spostQ = [spost(Qn).data for Qn in Q]
        commQ = [(spre(Qn) - spost(Qn)).data for Qn in Q]
        N_he_interact = 0

        for he_idx in range(N_he):
            he_state = list(idx2he[he_idx])
            n_excite = sum(he_state)

            # The diagonal elements for the hierarchy operator
            # coeff for diagonal elements
            sum_n_m_freq = 0.0
            for n in range(N_baths):
                for k in range(N_m):
                    #sum_n_m_freq += he_state[k+n*N_m]*nu[k,n]
                    sum_n_m_freq += he_state[k+n*N_m]*nu[k,n]

            op = -sum_n_m_freq*unit_sup
            L_he = _pad_csr(op, N_he, N_he, he_idx, he_idx)
            #print sup_dim
            #print N_he
            #print N_he*sup_dim
            #print L_he.shape[0]
            #print L_helems.shape[0]
            L_helems += L_he

            # Add the neighour interations
            he_state_neigh = copy(he_state)
            for n in range(N_baths):
                for k in range(N_m):

                    n_k = he_state[k+n*N_m]
                    if n_k >= 1:
                        # find the hierarchy element index of the neighbour before
                        # this element, for this Matsubara term and bath
                        he_state_neigh[k+n*N_m] = n_k - 1
                        he_idx_neigh = he2idx[tuple(he_state_neigh)]
                        if w0_list[n] == 'Mats':
                            op = c[k,n]*spreQ[n]- np.conj(c[k,n])*spostQ[n]
                            if renorm:
                                op = -1j*norm_minus[n_k, k,n]*op
                            else:
                                op = -1j*n_k*op
                            
                        else:
                            if w0_list[n]==0:  #do normal overdamped terms
                                op = c[k,n]*spreQ[n]- np.conj(c[k,n])*spostQ[n]
                                if renorm:
                                    op = -1j*norm_minus[n_k, k,n]*op
                                else:
                                    op = -1j*n_k*op
                            else:   #do resonant terms
                                #op = c[k,n]*spreQ[n]- np.conj(c[k,n])*spostQ[n]
                                #define c(k) for resonant terms with real and imaginary parts
                                if np.sign(np.imag(lam0[n]))>0:
                                    #print("a")
                                    #print(he_state)
                                    temp_param = beta[n]*0.5*(np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2) - 1.0j*gam[n]*0.5)
                                    #temp_param = beta[n]*0.5*(np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2))
                                else:
                                    #print("b")
                                    #print(he_state)
                                    temp_param = beta[n]*0.5*(np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2) + 1.0j*gam[n]*0.5)
                                    #temp_param = beta[n]*0.5*(np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2))
                                cr = 0.5 * ((np.real(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2))) * (coth(temp_param))) 
                                #old 
                                ci = 0.5 * (1.0j*np.imag(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2)))
                                #ci= 0.5*(np.imag(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2)))
                                #old 
                                opA = - 1j*cr*(spreQ[n]-spostQ[n]) 
                                #opA = cr*(spreQ[n]-spostQ[n]) 
                                #old 
                                opB = - ci*(spreQ[n]+spostQ[n]) #check sign here
                                #opB = ci*(spreQ[n]+spostQ[n]) #check sign here
                                #opB = - 1j*ci*(spreQ[n]+spostQ[n])
                                #opA = -1j*np.real(c[k,n])*(spreQ[n]-spostQ[n]) 
                                #opB= - 1j*np.imag(c[k,n])*(spreQ[n]+spostQ[n])
                                if renorm:
                                    #renorm seperately as in amirs paper.  am not sure about this since now real and imag have the same
                                    #exponent so multiply the same aux matrix.  I think they are renormed by the sum, but leave it
                                    #slightly general for the moment
                                    opA = norm_minus[n_k, k,n]*opA
                                    opB = norm_minus[n_k, k,n]*opB
                                    op = opA + opB
                                else:
                                    op = n_k*(opA+opB)
                                    
                                
                        L_he = _pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                        L_helems += L_he
                        N_he_interact += 1

                        he_state_neigh[k+n*N_m] = n_k

                    if n_excite <= N_c - 1:
                        # find the hierarchy element index of the neighbour after
                        # this element, for this Matsubara term
                        he_state_neigh[k+n*N_m] = n_k + 1
                        he_idx_neigh = he2idx[tuple(he_state_neigh)]
                        
                        op = commQ[n]
                        if renorm:
                            #not sure about this for resonant cases.  should be ok?
                            op = -1j*norm_plus[n_k, k,n]*op
                            #op = -1.0*norm_plus[n_k, k,n]*op
                        else:
                            #op = -1.0*op
                            #old 
                            op = -1j*op
                        

                        L_he = _pad_csr(op, N_he, N_he, he_idx, he_idx_neigh)
                        L_helems += L_he
                        N_he_interact += 1

                        he_state_neigh[k+n*N_m] = n_k

        if stats:
            stats.add_timing('hierarchy contruct',
                             timeit.default_timer() - start_helem_constr,
                            ss_conf)
            stats.add_count('Num hierarchy elements', N_he, ss_conf)
            stats.add_count('Num he interactions', N_he_interact, ss_conf)

        # Setup Liouvillian
        #i add time dependant parts seperately later with a callback function
        if stats: start_louvillian = timeit.default_timer()
        H_he = sp.kron(unit_helems, liouvillian(H_sys).data)
        H_he_td = []
        for H_sys_t in H_sys_td: #H_sys_td should always be a list of lists. sorry bad functionality
            H_he_td.append(sp.kron(unit_helems, liouvillian(H_sys_t[0]).data))
        #L_helems += H_he

        if stats:
            stats.add_timing('Liouvillian contruct',
                             timeit.default_timer() - start_louvillian,
                            ss_conf)

        if stats: start_integ_conf = timeit.default_timer()

        #r = scipy.integrate.ode(cy_ode_rhs)
        r = scipy.integrate.ode(_dsuper_list_td)
        #constant_func = lambda x, y: 1.0
        constant_func = lambda x: 1.0
        
        
#        L_list = [[L_helems,constant_func],[H_he,constant_func],[H_he_td,H_sys_td[1]]]
        L_list = [[L_helems,constant_func],[H_he,constant_func]]
        for H_pos,H_he_t in enumerate(H_he_td):
            L_list.append([H_he_t,H_sys_td[H_pos][1]])
        
        #r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_f_params(L_list)#, args)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('Liouvillian contruct',
                             time_now - start_integ_conf,
                            ss_conf)
            if ss_conf.total_time is None:
                ss_conf.total_time = time_now - start_config
            else:
                ss_conf.total_time += time_now - start_config

        self._ode = r
        self._N_he = N_he
        self._sup_dim = sup_dim
        self._configured = True
    
    
    
    
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
        N_he =  self._N_he 
        stats = self.stats
        r = self._ode

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")
        if stats:
            ss_conf = stats.sections.get('config')
            if ss_conf is None:
                raise RuntimeError("No config section for solver stats")
            ss_run = stats.sections.get('run')
            if ss_run is None:
                ss_run = stats.add_section('run')

        # Set up terms of the matsubara and tanimura boundaries
        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        if stats: start_init = timeit.default_timer()
        output.states.append(Qobj(rho0))
        rho0_flat = rho0.full().ravel('F') # Using 'F' effectively transposes
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat
        r.set_initial_value(rho0_he, tlist[0])

        if stats:
            stats.add_timing('initialize',
                             timeit.default_timer() - start_init, ss_run)
            start_integ = timeit.default_timer()

        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        full_hierarchy = []
        hshape = (N_he, sup_dim)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                full_hierarchy.append(r.y.reshape(hshape))
                output.states.append(rho)

        if stats:
            time_now = timeit.default_timer()
            stats.add_timing('integrate',
                             time_now - start_integ, ss_run)
            if ss_run.total_time is None:
                ss_run.total_time = time_now - start_run
            else:
                ss_run.total_time += time_now - start_run
            stats.total_time = ss_conf.total_time + ss_run.total_time

        return output, full_hierarchy

    def _calc_matsubara_params(self):
        """
        Calculate the Matsubara coefficents and frequencies

        Returns
        -------
        c, nu: array[N_m,N_baths] of complex_ float
        """
        N_m = self.N_exp
        N_c = self.N_cut
        lam0 = self.coup_strength
        gam = self.cut_freq
        hbar = self.planck
        w0_list = self.res_list
        N_baths=len(lam0)
        ck = self.ck_corr
        vk = self.vk_corr
        beta = []
        for T in self.temperature:
            if T == 0:
                beta.append(np.inf)
            else:
                beta.append(1.0/(self.boltzmann*T))
        
        
        
        c = np.empty((N_m,N_baths),dtype=np.complex_)
        nu = np.empty((N_m,N_baths),dtype=np.complex_)
        
        g = [2*np.pi / (b*hbar) for b in beta]
        for n in range(N_baths):
        
            if w0_list[n] == 'Mats':
                #do additional terms from fittting
                #this means ck and vk are padded
                c[k,n] = ck[n]
                nu[k,n] = vk[n]
            else:
                if w0_list[n]==0:  #do normal overdamped terms
                    if beta[n] == np.inf:
                        print("Warning:  zero temperature note defined for underdamped baths(yet)")
                    for k in range(N_m):
                        if k == 0:
                            
                            nu[k,n]=(gam[n])
                            c[k,n]=(lam0[n]*gam[n]*
                                (1.0/np.tan(gam[n]*hbar*beta[n]/2.0) - 1j) / hbar)
                        else:
                            nu[k,n]=(k*g[n])
                            c[k,n]=(4*lam0[n]*gam[n]*nu[k,n] /
                                  ((nu[k,n]**2 - gam[n]**2)*beta[n]*hbar**2))
                    
                    
                    print(c[k,n])
                    print((lam0[n]*gam[n]))
                
                else:
                        #if abs(w0_list[n])! = 0.:
                    for k in range(N_m):
                        #do underdamped coeesf.  this is a stupid way to do it.
                        #I assume each underdamped bath comes with two lam0s so that 
                        #c[n] = lam/2Om(A1 + i)
                        #c[n+1] = lam/2Om(A1 - i)
                        #the - sign has to be inserted by hand into the parameter lam[n+1] for that bath!
                        if np.sign(np.imag(lam0[n]))>0:
                            temp_param = beta[n]*0.5*(np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2) - 1.0j*gam[n]*0.5)
                        else:
                            temp_param = beta[n]*0.5*(np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2) + 1.0j*gam[n]*0.5)
                        #temp_param = beta[n]*0.5*(np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2))
                        #extra 0.5 comes from cos = 0.5(e+ + e-) game   
                        if beta[n] != np.inf:
                            #c[k,n]=0.5*((np.real(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2))) * (np.cosh(temp_param)/np.sinh(temp_param)) + 1.0j*np.imag(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2)))
                            c[k,n]=0.5*((np.real(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2))) * (np.cosh(temp_param)/np.sinh(temp_param)) + 1.0j*np.imag(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2)))
                            
                            
                            nu[k,n]=1.0j*np.sign(np.imag(w0_list[n]))*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2) + np.real(w0_list[n])*0.5
                        else:
                            c[k,n]=0.5*((np.real(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2)))  + 1.0j*np.imag(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2)))
                            nu[k,n]=1.0j*np.sign(np.imag(w0_list[n]))*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2) + np.real(w0_list[n])*0.5
                        #nu[k,n]=1.0j*np.sqrt(np.imag(w0_list[n])**2
                        print(c[k,n])
                        print(np.real(lam0[n])/(2*np.sqrt(np.imag(w0_list[n])**2-(gam[n]/2.)**2)))
                
                    
        self.exp_coeff = c
        self.exp_freq = nu
        return c, nu

    def _calc_renorm_factors(self):
        """
        Calculate the renormalisation factors

        Returns
        -------
        norm_plus, norm_minus : array[N_c, N_m,N_baths] of float
        """
        c = self.exp_coeff
        N_m = self.N_exp
        N_c = self.N_cut
        N_baths=len(self.coup_strength)
        norm_plus = np.empty((N_c+1, N_m,N_baths))
        norm_minus = np.empty((N_c+1, N_m,N_baths))
        for nb in range(N_baths):
            for k in range(N_m):
                for n in range(N_c+1):
                    norm_plus[n, k,nb] = np.sqrt(abs(c[k,nb])*(n + 1))
                    norm_minus[n, k,nb] = np.sqrt(float(n)/abs(c[k,nb]))

        return norm_plus, norm_minus
        
        
def _dsuper_list_td(t, y, L_list):#, args):

    L = L_list[0][0] * L_list[0][1](t)#, args)
    for n in range(1, len(L_list)):
        #
        # L_args[n][0] = the sparse data for a Qobj in super-operator form
        # L_args[n][1] = function callback giving the coefficient
        #
        #if L_list[n][2]:
            #L = L + L_list[n][0] * (L_list[n][1](t, args)) ** 2
        #    L = L + L_list[n][0] * (L_list[n][1](t)) ** 2
        #else:
            #L = L + L_list[n][0] * L_list[n][1](t, args)
        L = L + L_list[n][0] * L_list[n][1](t)

    return _ode_super_func(t, y, L)
            
            
            
def _ode_super_func(t, y, data):
    #ym = vec2mat(y)
    #return (data*ym).ravel('F')
    return (data*y)

def _pad_csr(A, row_scale, col_scale, insertrow=0, insertcol=0):
    """
    Expand the input csr_matrix to a greater space as given by the scale.
    Effectively inserting A into a larger matrix
         zeros([A.shape[0]*row_scale, A.shape[1]*col_scale]
    at the position [A.shape[0]*insertrow, A.shape[1]*insertcol]
    The same could be achieved through using a kron with a matrix with
    one element set to 1. However, this is more efficient
    """

    # ajgpitch 2016-03-08:
    # Clearly this is a very simple operation in dense matrices
    # It seems strange that there is nothing equivalent in sparse however,
    # after much searching most threads suggest directly addressing
    # the underlying arrays, as done here.
    # This certainly proved more efficient than other methods such as stacking
    #TODO: Perhaps cythonize and move to spmatfuncs

    if not isinstance(A, sp.csr_matrix):
        raise TypeError("First parameter must be a csr matrix")
    nrowin = A.shape[0]
    ncolin = A.shape[1]
    nrowout = nrowin*row_scale
    ncolout = ncolin*col_scale

    A._shape = (nrowout, ncolout)
    if insertcol == 0:
        pass
    elif insertcol > 0 and insertcol < col_scale:
        A.indices = A.indices + insertcol*ncolin
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")

    if insertrow == 0:
        A.indptr = np.concatenate((A.indptr,
                        np.array([A.indptr[-1]]*(row_scale-1)*nrowin)))
    elif insertrow == row_scale-1:
        A.indptr = np.concatenate((np.array([0]*(row_scale - 1)*nrowin),
                                   A.indptr))
    elif insertrow > 0 and insertrow < row_scale - 1:
         A.indptr = np.concatenate((np.array([0]*insertrow*nrowin), A.indptr,
                np.array([A.indptr[-1]]*(row_scale - insertrow - 1)*nrowin)))
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    return A