import numpy as np
import scipy.sparse as sp
import scipy.integrate

from numpy import matrix
from numpy import linalg
from scipy.misc import factorial
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip import spre, spost, sprepost, thermal_dm, mesolve, Options, dims
from qutip import tensor, identity, destroy, sigmax, sigmaz, basis, qeye
from qutip import liouvillian, mat2vec, state_number_enumerate
from functools import reduce
from operator import mul
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar

from copy import copy
import numpy as np

from scipy.misc import factorial
from scipy.integrate import ode
from scipy.constants import h as planck
from scipy.sparse import lil_matrix

from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip import commutator, Options
from qutip.solver import Result
from qutip import Qobj
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost


def add_at_idx(seq, k, val):
    """
    Add (subtract) a value in the tuple at position k
    """
    lst = list(seq)
    lst[k] += val
    return tuple(lst)

def prevhe(current_he, k, ncut):
    """
    Calculate the previous heirarchy index
    for the current index `n`.
    """
    nprev = add_at_idx(current_he, k, -1)
    if nprev[k] < 0:
        return False
    return nprev

def nexthe(current_he, k, ncut):
    """
    Calculate the next heirarchy index
    for the current index `n`.
    """
    nnext = add_at_idx(current_he, k, 1)
    if sum(nnext) > ncut:
        return False
    return nnext

def num_hierarchy(kcut, ncut):
    """
    Get the total number of auxiliary density matrices in the
    Hierarchy
    """
    return int(factorial(ncut + kcut)/(factorial(ncut)*factorial(kcut)))

class Heom(object):
    """
    The Heom class to tackle Heirarchy.
    
    Parameters
    ==========
    hamiltonian: :class:`qutip.Qobj`
        The system Hamiltonian
    
    coupling: :class:`qutip.Qobj`
        The coupling operator
    
    ck: list
        The list of amplitudes in the expansion of the correlation function

    vk: list
        The list of frequencies in the expansion of the correlation function

    ncut: int
        The Heirarchy cutoff
        
    kcut: int
        The cutoff in the Matsubara frequencies

    rcut: float
        The cutoff for the maximum absolute value in an auxillary matrix
        which is used to remove it from the heirarchy
    """
    def __init__(self, hamiltonian, coupling, ck, vk,
                 ncut, rcut=None, renorm=False, lam=0.):
        self.hamiltonian = hamiltonian
        self.coupling = coupling    
        self.ck, self.vk = ck, vk
        self.ncut = ncut
        self.renorm = renorm
        self.kcut = len(ck)
        nhe, he2idx, idx2he =_heom_state_dictionaries([ncut+1]*(len(ck)), ncut)
        # he2idx, idx2he, nhe = self._initialize_he()
        self.nhe = nhe
        self.he2idx = he2idx
        self.idx2he = idx2he
        self.N = self.hamiltonian.shape[0]
        
        total_nhe = int(factorial(self.ncut + self.kcut)/(factorial(self.ncut)*factorial(self.kcut)))
        self.total_nhe = total_nhe
        self.hshape = (total_nhe, self.N**2)
        self.weak_coupling = self.deltak()
        self.L = liouvillian(self.hamiltonian, []).data
        self.grad_shape = (self.N**2, self.N**2)
        self.spreQ = spre(coupling).data
        self.spostQ = spost(coupling).data
        self.L_helems = lil_matrix((total_nhe*self.N**2, total_nhe*self.N**2), dtype=np.complex)
        self.lam = lam

    def _initialize_he(self):
        """
        Initialize the hierarchy indices
        """
        zeroth = tuple([0 for i in range(self.kcut)])
        he2idx = {zeroth:0}
        idx2he = {0:zeroth}
        nhe = 1
        return he2idx, idx2he, nhe

    def populate(self, heidx_list):
        """
        Given a Hierarchy index list, populate the graph of next and
        previous elements
        """
        ncut = self.ncut
        kcut = self.kcut
        he2idx = self.he2idx
        idx2he = self.idx2he
        for heidx in heidx_list:
            for k in range(self.kcut):
                he_current = idx2he[heidx]
                he_next = nexthe(he_current, k, ncut)
                he_prev = prevhe(he_current, k, ncut)
                if he_next and (he_next not in he2idx):
                    he2idx[he_next] = self.nhe
                    idx2he[self.nhe] = he_next
                    self.nhe += 1

                if he_prev and (he_prev not in he2idx):
                    he2idx[he_prev] = self.nhe
                    idx2he[self.nhe] = he_prev
                    self.nhe += 1

    def deltak(self):
        """
        Calculates the deltak values for those Matsubara terms which are
        greater than the cutoff set for the exponentials.
        """
        # Needs some test or check here
        if self.kcut >= len(self.vk):
            return 0
        else:
            dk = np.sum(np.divide(self.ck[self.kcut:], self.vk[self.kcut:]))
            return dk
    
    def grad_n(self, he_n):
        """
        Get the gradient term for the Hierarchy ADM at
        level n
        """
        c = self.ck
        nu = self.vk
        L = self.L.copy()
        gradient_sum = -np.sum(np.multiply(he_n, nu))
        sum_op = gradient_sum*np.eye(L.shape[0])
        L += sum_op

        # Fill in larger L
        nidx = self.he2idx[he_n]
        block = self.N**2
        pos = int(nidx*(block))
        self.L_helems[pos:pos+block, pos:pos+block] = L

    def grad_prev(self, he_n, k, prev_he):
        """
        Get prev gradient
        """
        c = self.ck
        nu = self.vk
        spreQ = self.spreQ
        spostQ = self.spostQ
        nk = he_n[k]
        norm_prev = nk
            
        if k == 0:
            norm_prev = np.sqrt(float(nk)/abs(self.lam))
            op1 = -1j*norm_prev*(-self.lam*spostQ)
        elif k == 1 :
            norm_prev = np.sqrt(float(nk)/abs(self.lam))
            op1 = -1j*norm_prev*(self.lam*spreQ)
        else:
            norm_prev = np.sqrt(float(nk)/abs(c[k]))
            op1 = -1j*norm_prev*(c[k]*(spreQ - spostQ))
        # Fill in larger L
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[prev_he]
        block = self.N**2
        rowpos = int(rowidx*(block))
        colpos = int(colidx*(block))
        self.L_helems[rowpos:rowpos+block, colpos:colpos+block] = op1

    def grad_next(self, he_n, k, next_he):
        c = self.ck
        nu = self.vk
        spreQ = self.spreQ
        spostQ = self.spostQ
        
        nk = he_n[k]            
        
        if k < 2:
            norm_next = np.sqrt(self.lam*(nk + 1))
            op2 = -1j*norm_next*(spreQ - spostQ)
        else:
            norm_next = np.sqrt(abs(c[k])*(nk + 1))
            op2 = -1j*norm_next*(spreQ - spostQ)
        # Fill in larger L
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[next_he]
        block = self.N**2
        rowpos = int(rowidx*(block))
        colpos = int(colidx*(block))
        self.L_helems[rowpos:rowpos+block, colpos:colpos+block] = op2
    
    def rhs(self, progress=None):
        """
        Make the RHS
        """
        while self.nhe < self.total_nhe:
            heidxlist = copy(list(self.idx2he.keys()))
            self.populate(heidxlist)
        if progress != None:
            bar = progress(total = self.nhe*self.kcut)

        for n in self.idx2he:
            he_n = self.idx2he[n]
            self.grad_n(he_n)
            for k in range(self.kcut):
                next_he = nexthe(he_n, k, self.ncut)
                prev_he = prevhe(he_n, k, self.ncut)
                if next_he and (next_he in self.he2idx):
                    self.grad_next(he_n, k, next_he)
                if prev_he and (prev_he in self.he2idx):
                    self.grad_prev(he_n, k, prev_he)
    
    def solve(self, rho0, tlist, options=None, progress=None):
        """
        Solve the Hierarchy equations of motion for the given initial
        density matrix and time.
        """
        if options is None:
            options = Options()

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []
        output.states.append(Qobj(rho0))

        dt = np.diff(tlist)
        rho_he = np.zeros(self.hshape, dtype=np.complex)
        rho_he[0] = rho0.full().ravel("F")
        rho_he = rho_he.flatten()

        self.rhs()
        L_helems = self.L_helems.asformat("csr")
        r = ode(cy_ode_rhs)
        r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)
                        
        r.set_initial_value(rho_he, tlist[0])
        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        if progress:
            bar = progress(total=n_tsteps-1)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                r1 = r.y.reshape(self.hshape)
                r0 = r1[0].reshape(self.N, self.N).T
                output.states.append(Qobj(r0))
                if progress: bar.update()
        return output



def _heom_state_dictionaries(dims, excitations):
    """
    Return the number of states, and lookup-dictionaries for translating
    a state tuple to a state index, and vice versa, for a system with a given
    number of components and maximum number of excitations.
    Parameters
    ----------
    dims: list
        A list with the number of states in each sub-system.
    excitations : integer
        The maximum numbers of dimension
    Returns
    -------
    nstates, state2idx, idx2state: integer, dict, dict
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a dictionary for looking up state
        state tuples from state indices.
    """
    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in state_number_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1
    return nstates, state2idx, idx2state


def _heom_number_enumerate(dims, excitations=None, state=None, idx=0):
    """
    An iterator that enumerate all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.
    Example:
        >>> for state in state_number_enumerate([2,2]):
        >>>     print(state)
        [ 0.  0.]
        [ 0.  1.]
        [ 1.  0.]
        [ 1.  1.]
    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.
    state : list
        Current state in the iteration. Used internally.
    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.
    idx : integer
        Current index in the iteration. Used internally.
    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.
    """

    if state is None:
        state = np.zeros(len(dims))

    if excitations and sum(state[0:idx]) > excitations:
        pass
    elif idx == len(dims):
        if excitations is None:
            yield np.array(state)
        else:
            yield tuple(state)
            
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, excitations, state, idx + 1):
                yield s


def cot(x):
    """
    Calculate cotangent.
    Parameters
    ----------
    x: Float
        Angle.
    """
    return np.cos(x)/np.sin(x)

def hsolve_grad(H, Q, ckA, vkA, ck_corr, vk_corr, Nc, N, lam):      
    #Parameters and hamiltonian
    hbar=1.
    kb=1.

    #Set by system 
    dimensions = dims(H)
    #Nsup = dimensions[0][0] * dimensions[0][0] 
    #unit = qeye(dimensions[0])
    N_temp = reduce(mul, H.dims[0], 1)
    Nsup = N_temp**2
    unit = qeye(N_temp)
    #Ntot is the total number of ancillary elements in the hierarchy
    Ntot = int(round(factorial(Nc+N) / (factorial(Nc) * factorial(N))))
    LD1 = -2.* spre(Q) * spost(Q.dag()) + spre(Q.dag() * Q) + spost(Q.dag() * Q)
    
    #for arbitrary SD I need to generalize ck and vk
    c0=ckA[0]
    #c0 =(lam0 * gam * (cot(gam * hbar / (2. * kb * Temperature)) - (1j))) / hbar
    #pref = ((2. * lam0 * kb * Temperature / (gam * hbar)) -1j * lam0) / hbar
    pref=0.
    #gj = 2 * np.pi * kb * Temperature / hbar
    L12=0.*LD1;
    #L12 = -pref * LD1 + (c0 / gam) * LD1 
    #L12 = -pref * LD1 + (ckA[0] / vkA[0]) * LD1 

    #for i1 in range(1,N):
        #ci = (4 * lam0 * gam * kb * Temperature * i1
        #     * gj/((i1 * gj)**2 - gam**2)) / (hbar**2)
    #    L12 = L12 + (ckA[i1] / vkA[i1]) * LD1
    
    #Setup liouvillian

    L = liouvillian(H, [L12])
    Ltot = L.data
    unitthing=sp.identity(Ntot, dtype='complex', format='csr')
    #unitthing = sp.csr_matrix(np.identity(Ntot))
    Lbig = sp.kron(unitthing,Ltot.tocsr())
    #rho0big1 = np.zeros((Nsup * Ntot), dtype=complex)
    
    nstates, state2idx, idx2state =_heom_state_dictionaries([Nc+1]*(N),Nc)
    for nlabelt in _heom_number_enumerate([Nc+1]*(N),Nc):
        nlabel = list(nlabelt)                    
        ntotalcheck = 0
        for ncheck in range(N):
            ntotalcheck = ntotalcheck + nlabel[ncheck]                            
        current_pos = int(round(state2idx[tuple(nlabel)]))
        Ltemp = sp.lil_matrix((Ntot, Ntot))
        Ltemp[current_pos,current_pos] = 1.
        Ltemp.tocsr()
        Lbig = Lbig + sp.kron(Ltemp,(-nlabel[0] * vkA[0] * spre(unit).data))
        Lbig = Lbig + sp.kron(Ltemp,(-nlabel[1] * vkA[1] * spre(unit).data))
        #bi-exponential corrections:
        if N==3:
            Lbig = Lbig + sp.kron(Ltemp,(-nlabel[2] * vk_corr[0] * spre(unit).data))
        if N==4:
            Lbig = Lbig + sp.kron(Ltemp,(-nlabel[2] * vk_corr[0] * spre(unit).data))
            Lbig = Lbig + sp.kron(Ltemp,(-nlabel[3] * vk_corr[1] * spre(unit).data))
        
        #for kcount in range(N):
        #    Lbig = Lbig + sp.kron(Ltemp,(-nlabel[kcount] * (vkA[kcount])
        #                    * spre(unit).data))
        
        for kcount in range(N):
            if nlabel[kcount]>=1:
            #find the position of the neighbour
                nlabeltemp = copy(nlabel)
                nlabel[kcount] = nlabel[kcount] -1
                current_pos2 = int(round(state2idx[tuple(nlabel)]))
                Ltemp = sp.lil_matrix(np.zeros((Ntot,Ntot)))
                Ltemp[current_pos, current_pos2] = 1
                Ltemp.tocsr()
            # renormalized version:    
                #ci =  (4 * lam0 * gam * kb * Temperature * kcount
                #      * gj/((kcount * gj)**2 - gam**2)) / (hbar**2)
                if kcount==0:
                    
                    c0n=lam
                    Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                     * np.sqrt((nlabeltemp[kcount]
                                        / abs(c0n)))
                                     * (0.0*spre(Q).data
                                     - (lam)
                                     * spost(Q).data)))
                if kcount==1:     
                    cin=lam
                    ci =  ckA[kcount]
                    Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                     * np.sqrt((nlabeltemp[kcount]
                                        / abs(cin)))
                                     * ((lam) * spre(Q).data
                                     - (0.0)
                                     * spost(Q).data)))
                    
                if kcount==2:     
                    cin=ck_corr[0]
                    #ci =  ckA[kcount]
                    
                    Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                         * np.sqrt((nlabeltemp[kcount]
                                            / abs(cin)))
                                         * cin*(spre(Q).data - spost(Q).data)))
                if kcount==3:     
                    cin=ck_corr[1]
                    #ci =  ckA[kcount]
                    
                    Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                         * np.sqrt((nlabeltemp[kcount]
                                            / abs(cin)))
                                         * cin*(spre(Q).data - spost(Q).data)))
                nlabel = copy(nlabeltemp)

        for kcount in range(N):
            if ntotalcheck<=(Nc-1):
                nlabeltemp = copy(nlabel)
                nlabel[kcount] = nlabel[kcount] + 1
                current_pos3 = int(round(state2idx[tuple(nlabel)]))
            if current_pos3<=(Ntot):
                Ltemp = sp.lil_matrix(np.zeros((Ntot,Ntot)))
                Ltemp[current_pos, current_pos3] = 1
                Ltemp.tocsr()
            #renormalized   
                if kcount==0:
                    c0n=lam
                    Lbig = Lbig + sp.kron(Ltemp,-1.j
                                  * np.sqrt((nlabeltemp[kcount]+1)*((abs(c0n))))
                                  * (spre(Q)- spost(Q)).data)
                if kcount==1:
                    ci =ckA[kcount]
                    cin=lam
                    Lbig = Lbig + sp.kron(Ltemp,-1.j
                                  * np.sqrt((nlabeltemp[kcount]+1)*(abs(cin)))
                                  * (spre(Q)- spost(Q)).data)
                if kcount==2:
                    cin=ck_corr[0]
                    Lbig = Lbig + sp.kron(Ltemp,-1.j
                                  * np.sqrt((nlabeltemp[kcount]+1)*(abs(cin)))
                                  * (spre(Q)- spost(Q)).data)
                if kcount==3:
                    cin=ck_corr[1]
                    Lbig = Lbig + sp.kron(Ltemp,-1.j
                                  * np.sqrt((nlabeltemp[kcount]+1)*(abs(cin)))
                                  * (spre(Q)- spost(Q)).data)    
             
            nlabel = copy(nlabeltemp)
    return Lbig
