import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy.linalg as lin


def build_massMatrix(N):
    a = 1 / 6 * np.ones(N - 1)
    M = sp.diags(a, -1) + sp.diags(a, 1) + 2 / 3 * sp.eye(N)
    return 1 / (N + 1) * M


def alpha(x):
    return 1 + x * x


def beta(x):
    return 2 * x


def gamma(x):
    return np.pi**2 * x**2



def build_rigidityMatrix(N, alpha, beta, gamma):
    X = np.linspace(0, 1, N + 2)
    Y = (X[:-1] + X[1:]) / 2
    h = 1 / (N + 1)

    a1 = alpha(X)
    a2 = alpha(Y)

    b1 = beta(X)
    b2 = beta(Y)

    c1 = gamma(X)
    c2 = gamma(Y)

    d3 = (
        -1 / h * (a1[1:-2] + 4 * a2[1:-1] + a1[2:-1])
        + b1[1:-2]
        + 2 * b2[1:-1]
        + h * c2[1:-1]
    ) / 6
    d1 = (
        -1 / h * (a1[1:-2] + 4 * a2[1:-1] + a1[2:-1])
        - b1[2:-1]
        - 2 * b2[1:-1]
        + h * c2[1:-1]
    ) / 6
    d2 = (
        1 / h * (a1[:-2] + 4 * a2[:-1] + a1[1:-1])
        + 2 * b2[:-1]
        + b1[1:-1]
        + h * (c2[:-1] + c1[1:-1])
        + 1 / h * (a1[1:-1] + 4 * a2[1:] + a1[2:])
        - 2 * b2[1:]
        - b1[1:-1]
        + h * (c2[1:] + c1[1:-1])
    ) / 6

    return sp.diags([d1, d2, d3], [-1, 0, 1])


def f(t, x):
    return np.exp(-t) * (2 * np.pi**2 * x**2 + np.pi**2 - 1) * np.sin(np.pi * x)


def initial_value(x):
    return np.sin(np.pi * x)


def exact_solution_at_1(x):
    return np.exp(-1) * np.sin(np.pi * x)


def build_F(t, N):
    h = 1 / (N + 1)
    X = np.linspace(0, 1, N + 2)
    return h / 3 * (f(t, X[:-2] + h / 2) + f(t, X[1:-1]) + f(t, X[1:-1] + h / 2))


def FEM_theta(N, M, theta):
    k = 1 / M
    grid = (1 / (N + 1)) * np.arange(1, N + 1)
    u_sol = initial_value(grid)
    MatrixM = build_massMatrix(N)
    MatrixA = build_rigidityMatrix(N, alpha, beta, gamma)
    B_theta = MatrixM + k * theta * MatrixA
    C_theta = MatrixM - k * (1 - theta) * MatrixA

    B_theta = B_theta.tocsr()
    C_theta = C_theta.tocsr()

    for i in range(M):
        F_theta = k * theta * build_F(k * (i + 1), N) + k * (1 - theta) * build_F(k * i, N)
        RHS = C_theta @ u_sol + F_theta
        u_sol = spsolve(B_theta, RHS)

    return u_sol


#### error analysis ####
nb_samples = 5

# M and N for 3c
# N = np.power(2, np.arange(5, 5 + nb_samples)) - 1
# M = np.power(2, np.arange(5, 5 + nb_samples))
# M and N for 3d
N = np.power(2, np.arange(2, 2 + nb_samples)) - 1
M = np.power(4, np.arange(2, 2 + nb_samples))
print(M)
theta = 0.5

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
