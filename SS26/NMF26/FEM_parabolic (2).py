import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy.linalg as lin



def alpha(x):
    # Todo : Implement the function alpha
    return


def beta(x):
    # Todo : Implement the function beta
    return


def gamma(x):
    # Todo : Implement the function gamma
    return



def build_massMatrix(N):
    # Todo : Implement the function build_massMatrix
    return



def build_rigidityMatrix(N, alpha, beta, gamma):
    # Todo : Implement the function build_rigidityMatrix
    return


def f(t, x):
    # Todo : Implement the function f
    return


def initial_value(x):
    # Todo : Implement the function initial_value
    return


def exact_solution_at_1(x):
    # Todo : Implement the function exact_solution_at_1
    return


def build_F(t, N):
    # Todo : Implement the function build_F
    return


def FEM_theta(N, M, theta):
    # Todo : Implement the theta scheme, return the solution u_sol at final time
    return u_sol


#### error analysis ####
nb_samples = 5
N = # fill in this line for 3c)-d)
M = # fill in this line for 3c)-d)
theta = # fill in this line for 3c)-d)

#### Do not change any code below! ####
l2error = np.zeros(nb_samples)
k = 1 / M


try:
    for i in range(nb_samples):
        l2error[i] = (1 / (N[i] + 1)) ** (1 / 2) * lin.norm(
            exact_solution_at_1((1 / (N[i] + 1)) * (np.arange(N[i]) + 1))
            - FEM_theta(N[i], M[i], theta),
            ord=2,
        )
        if np.isnan(l2error[i]) == True:
            raise Exception("Error unbounded. Plots not shown.")
    conv_rate = np.polyfit(np.log(k), np.log(l2error), deg=1)
    if not conv_rate[0] >= 0:
        raise Exception("Error unbounded. Plots not shown.")
    print(
        f"FEM with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}"
    )
    plt.figure(figsize=[10, 6])
    plt.loglog(k, l2error, "-x", label="error")
    plt.loglog(k, k, "--", label="$O(k)$")
    plt.loglog(k, k**2, "--", label="$O(k^2)$")
    plt.title("$L^2$ convergence rate", fontsize=13)
    plt.xlabel("$k$", fontsize=13)
    plt.ylabel("error", fontsize=13)
    plt.legend()
    plt.plot()
    plt.show()
except Exception as e:
    print(e)
