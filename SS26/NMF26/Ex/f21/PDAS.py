import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def PDAS(A, b, x0, lambda0):
    k1 = 1
    tol = 1e-12
    kmax = 50
    # initialize algorithm
    n = b.size
    c = np.zeros(n)
    lamb = lambda0
    k = 1
    Ik = lamb + k1*(c-x0) <= 0
    Ak = lamb + k1*(c-x0) > 0
    B = -sp.eye(n)
    C = sp.diags(Ak*1.0, 0)
    D = sp.diags(Ik*1.0, 0)
    L1 = sp.hstack((A, B))
    L2 = sp.hstack((C, D))
    LHS = sp.vstack((L1, L2))
    LHS = LHS.tocsc()
    RHS = np.hstack((b, C@c))
    y = spsolve(LHS, RHS)
    x = y[:n]
    lamb = y[n:]
    crit = (np.sqrt(np.dot(x-x0, x-x0))) < tol
    while ((not crit) and (k < kmax)):
        x0 = x
        Ik = lamb + k1*(c-x) <= 0
        Ak = lamb + k1*(c-x) > 0
        C = sp.diags(Ak*1.0, 0)
        D = sp.diags(Ik*1.0, 0)
        L1 = sp.hstack((A, B))
        L2 = sp.hstack((C, D))
        LHS = sp.vstack((L1, L2))
        LHS = LHS.tocsc()
        RHS = np.hstack((b, C@c))
        y = spsolve(LHS, RHS)
        x = y[:n]
        lamb = y[n:]
        k = k+1
        crit = (np.sqrt(np.dot(x-x0, x-x0))) < tol
    if ~crit:
        print('Warning: PDAS did not converge')
    return x, lamb
