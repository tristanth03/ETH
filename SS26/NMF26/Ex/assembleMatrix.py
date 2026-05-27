import numpy as np
import scipy.sparse as sp

def assembleMatrixAs(N, R, func):
    """
    Parameters:
        N (int): The number of intervals.
        R (float): The upper bound of the interval.
        func (function): The function used to calculate the values for the matrix.

    Returns:
        As (sparse matrix): Approximation of (N+1)x(N+1) Matrix As defined by As_{ij} := int_0^R b_i'(s) b_j'(s) func(s) ds.
        
    Assumptions:
        - Assumes a uniform mesh x_i = h*i, 0 <= i < N+1.
        - Assumes Dirichlet conditions at R (i.e. b_i are the piecewise linear functions defined by b_i(x_j) = delta_{ij}, 0 <= i,j <= N).
    """

    xi = np.linspace(0, R, N + 2)
    h = R / (N + 1)
    dof = xi[:-1]
    diag = np.zeros(N + 1)
    p1 = h / 2 - h / (2 * np.sqrt(3))
    p2 = h / 2 + h / (2 * np.sqrt(3))
    diag[0] = 1 / (2 * h) * (func(p1) + func(p2))
    diag[1:] = 1 / (2 * h) * (func(dof[:-1] + p1) + func(dof[:-1] + p2))
    diag[1:] += 1 / (2 * h) * (func(dof[1:] + p1) + func(dof[1:] + p2))
    lodiag = -1 / (2 * h) * (func(dof[:-1] + p1) + func(dof[:-1] + p2))
    updiag = lodiag
    As= sp.diags([lodiag, diag, updiag], [-1, 0, 1])
    return As


def assembleMatrixAb(N, R, func):
    """
    Parameters:
        N (int): The number of intervals.
        R (float): The upper bound of the interval.
        func (function): The function used to calculate the values for the matrix.

    Returns:
        Ab (sparse matrix): Approximation of (N+1)x(N+1) Matrix Ab defined by Ab_{ij} := int_0^R b_j'(s) b_i(s) func(s) ds.
        
    Assumptions:
        - Assumes a uniform mesh x_i = h*i, 0 <= i < N+1.
        - Assumes Dirichlet conditions at R (i.e. b_i are the piecewise linear functions defined by b_i(x_j) = delta_{ij}, 0 <= i,j <= N).
    """
    xi = np.linspace(0, R, N + 2)
    h = R / (N + 1)
    dof = xi[:-1]
    diag = np.zeros(N + 1)
    p1 = h / 2 - h / (2 * np.sqrt(3))
    p2 = h / 2 + h / (2 * np.sqrt(3))
    w1 = p1 / h
    w2 = p2 / h
    diag[0] = -1 / 2 * (func(p1) * w2 + func(p2) * w1)
    diag[1:] = 1 / 2 * (func(dof[:-1] + p1) * w1 + func(dof[:-1] + p2) * w2)
    diag[1:] += -1 / 2 * (func(dof[1:] + p1) * w2 + func(dof[1:] + p2) * w1)
    lodiag = -1 / 2 * (func(dof[:-1] + p1) * w1 + func(dof[:-1] + p2) * w2)
    updiag = 1 / 2 * (func(dof[:-1] + p1) * w2 + func(dof[:-1] + p2) * w1)
    Ab = sp.diags([lodiag, diag, updiag], [-1, 0, 1])
    return Ab


def assembleMatrixAm(N, R, func):
    """
    Parameters:
        N (int): The number of intervals.
        R (float): The upper bound of the interval.
        func (function): The function used to calculate the values for the matrix.

    Returns:
        Am (sparse matrix): Approximation of (N+1)x(N+1) Matrix Am defined by Am_{ij} := int_0^R b_i(s) b_j(s) func(s) ds.
        
    Assumptions:
        - Assumes a uniform mesh x_i = h*i, 0 <= i < N+1.
        - Assumes Dirichlet conditions at R (i.e. b_i are the piecewise linear functions defined by b_i(x_j) = delta_{ij}, 0 <= i,j <= N).
    """
    xi = np.linspace(0, R, N + 2)
    h = R / (N + 1)
    dof = xi[:-1]
    diag = np.zeros(N + 1)
    p1 = h / 2 - h / (2 * np.sqrt(3))
    p2 = h / 2 + h / (2 * np.sqrt(3))
    w1 = p1 / h
    w2 = p2 / h
    diag[0] = h / 2 * (func(p1) * w2**2 + func(p2) * w1**2)
    diag[1:] = h / 2 * (func(dof[:-1] + p1) * w1**2 + func(dof[:-1] + p2) * w2**2)
    diag[1:] += h / 2 * (func(dof[1:] + p1) * w2**2 + func(dof[1:] + p2) * w1**2)
    updiag = h / 2 * w1 * w2 * (func(dof[:-1] + p1) + func(dof[:-1] + p2))
    lodiag = updiag
    Am = sp.diags([lodiag, diag, updiag], [-1, 0, 1])
    return Am


def assembleMatrix(N, R, a, b, c):
    """
    Assembles the matrix by combining the matrices As, Ab, and Am.

    Parameters:
        N (int): The number of intervals.
        R (float): The upper bound of the interval.
        a (function): The function used to calculate the values for the first term of the matrix.
        b (function): The function used to calculate the values for the second term of the matrix.
        c (function): The function used to calculate the values for the third term of the matrix.

    Returns:
        A (sparse matrix): The assembled matrix. Approximation of (N+1)x(N+1) Matrix A defined by 
        A_{ij} := int_0^R a(s) b_i(s) b_j(s) ds + int_0^R b(s) b_i(s) b_j'(s) + int_0^R c(s) b_i'(s) b_j'(s).

    Assumptions:
        - Assumes a uniform mesh x_i = h*i, 0 <= i < N+1.
        - Assumes Dirichlet conditions at R (i.e. b_i are the piecewise linear functions defined by b_i(x_j) = delta_{ij}, 0 <= i,j <= N).
    """
    As = assembleMatrixAs(N, R, a)
    Ab = assembleMatrixAb(N, R, b)
    Am = assembleMatrixAm(N, R, c)
    A = As + Ab + Am
    return A