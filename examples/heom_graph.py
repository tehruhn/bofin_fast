"""
Plots the ADO levels in the Hierarchy
"""


from qutip import liouvillian, mat2vec, state_number_enumerate
import networkx as nx
import matplotlib.pyplot as plt


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


def heom_state_dictionaries(dims, excitations):
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


# Set the cutoffs
ncut = 3
kcut = 3

nhe, he2idx, idx2he = heom_state_dictionaries([ncut+1]*(kcut), ncut)

g = nx.Graph()


for n in idx2he:
    he_n = idx2he[n]    
    for k in range(kcut):
        next_he = nexthe(he_n, k, ncut)
        prev_he = prevhe(he_n, k, ncut)
        if next_he and (next_he in he2idx):
            g.add_edge(n, he2idx[next_he])
        if prev_he and (prev_he in he2idx):
            g.add_edge(n, he2idx[prev_he])

            
plt.figure(figsize=(10, 10))
nx.draw(g, labels=idx2he, with_labels=True)
plt.show()
