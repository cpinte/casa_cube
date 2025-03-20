from astroquery.lamda import Lamda
import scipy.constants as sc
import numpy as np

def LTE_equilibrium(mol,T):
    # Return the LTE distribution of a molecule levels
    # with the total population normalised to 1

    _, transitions, levels = Lamda.query(mol=mol)
    n_levels = len(levels)

    n = np.zeros(n_levels)
    n[0] = 1.0

    for l in range(1,n_levels):
        nu = transitions['Frequency'][l-1] * 1e9
        n[l] = n[l-1] * levels['Weight'][l] / levels['Weight'][l-1] * np.exp( - sc.h * nu / (sc.k * T))

    return n/np.sum(n)
