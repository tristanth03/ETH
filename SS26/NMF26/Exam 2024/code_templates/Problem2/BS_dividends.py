from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import numpy.linalg as lin
from scipy.stats import norm
import matplotlib.pyplot as plt


def sigmaT(t, T):
    return 0.3


def deltaT(t, T):
    return 0.02*(T-t)


def rT(t, T):
    return 0.05


def buildMassBS(N, R):
    """
    Parameters:
    - N (int)
    - R (float)

    Returns:
    - MassBS : The (N x N) mass matrix.
    """
    sup = [1/6]*(N-1)
    sub = [1/6]*(N-1)
    diag = [2/3]*(N)
    h = 2*R / (N+1)

    return h* (sp.diags(sup,1)+sp.diags(sub,-1)+sp.diags(diag,0))


def buildABS(N, R, sigmaT, rT, deltaT, t, T):
    """
    Parameters:
    - N (int)
    - R (float)
    - sigmaT (function)
    - rT (function)
    - deltaT (function)
    - t (float)
    - T (float)

    Returns:
    - ABS : The (N x N) stiffness matrix.
    """
    h = 2*R / (N+1)
    sup = [-sigmaT(t,T)**2/(2*h)+0.5* (sigmaT(t,T)**2/2+deltaT(t,T)-rT(t,T))+rT(t,T)*(h/6)]*(N-1)
    sub = [-sigmaT(t,T)**2/(2*h)-0.5* (sigmaT(t,T)**2/2+deltaT(t,T)-rT(t,T))+rT(t,T)*(h/6)]*(N-1)
    diag = [sigmaT(t,T)**2/h+2*h/3 * rT(t,T)]*(N)
    return (sp.diags(sup,1)+sp.diags(sub,-1)+sp.diags(diag,0))


def u_init(x, K):
    return np.maximum(K-(np.exp(x)),0)


def exactu(x):
    d2 = (x - 0.005) / 0.3
    d1 = d2 + 0.3
    return np.exp(-0.05) * norm.cdf(-d2) - np.exp(x - 0.01) * norm.cdf(-d1)


def FEM_theta(N, M, R, sigmaT, rT, deltaT, K, T, theta):
    """
    Parameters:
    - N (int)
    - M (int)
    - R (float)
    - sigmaT (function)
    - rT (function)
    - deltaT (function)
    - K (float)
    - T (float)
    - theta (float)

    Returns:
    - u_sol (numpy.ndarray): The solution vector on the grid x_i, i=1,2,...,N.
    """
    k = T/M
    M_mat = buildMassBS(N,R)
    u = np.zeros(shape=(N,M+1))
    u[:,0] = u_init(np.linspace(-R,R,N+2)[1:-1],K)
   
    for m in range(1,M+1):
        A_mat = buildABS(N,R,sigmaT,rT,deltaT,(m-1)*k,T)
        A_mat_1 = buildABS(N,R,sigmaT,rT,deltaT,(m)*k,T)

        LHS = (M_mat+k*theta*A_mat_1)
        RHS = (M_mat-k*(1-theta)*A_mat)@u[:,m-1]
        u[:,m] = spsolve(LHS,RHS)
    
    return u[:,-1]


if __name__ == "__main__":


    #### Parameter setting
    T = 1
    R = 4
    K = 1
    theta = 0.5
    l = np.arange(4,4+5)
    N = 2**l-1
    M = 2**l


    ############################ Do not change any code below! ############################
    G = R /2
    error = np.zeros(5)
    k = T / M


    try:
        for i in range(5):
            grid = np.linspace(-R, R, N[i] + 2)[1:-1]
            ind = (np.abs(grid)) < G
            err = exactu(grid) - FEM_theta(N[i], M[i], R, sigmaT, rT, deltaT, K, T, theta)
            error[i] = (1 / (N[i] + 1)) ** (1 / 2) * lin.norm(err[ind], ord=2)
            if np.isnan(error[i]) == True:
                raise Exception("Error unbounded. Plots not shown.")
        conv_rate = np.polyfit(np.log(k), np.log(error), deg=1)
        if conv_rate[0] < 0:
            raise Exception("Error did not converge. Plots not shown.")
        print(
            f"FEM method with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}"
        )
        plt.figure(figsize=[10, 6])
        plt.loglog(k, error, "-x", label="error")
        plt.loglog(k, k, "--", label="$O(k)$")
        plt.loglog(k, k**2, "--", label="$O(k^2)$")
        plt.title("Convergence rate", fontsize=13)
        plt.xlabel("$k$", fontsize=13)
        plt.ylabel("error", fontsize=13)
        plt.legend()
        plt.plot()
        # plt.savefig(Path.home() / "questions" / "Problem2" / "plot_BS.pdf", format="pdf")
        plt.show()
    except Exception as e:
        print(e)
