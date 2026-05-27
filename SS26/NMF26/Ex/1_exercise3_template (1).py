import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

# Set how floating-point errors are handled.
np.seterr(all='raise')


def initial_value(x):
    return np.sin(np.pi/2 * x)


#### exact solution at t=1 ####
def exact_solution_at_1(x):
    # todo 3 a)
    return np.exp(-np.pi**2 / 4 * 1)*np.sin(np.pi/2 * x)


#### numerical scheme ####
def eulerexplicit(N, M):
    # todo 3 b)
    h = 1/N; k = 1/M
    sub = [1]*(N-1)
    sup = [1]*(N-1)
    diag = [-2]*(N)
    sub[-1] = 2
    Mass_mat = k/h**2 * (diags(sup,1)+diags(diag,0)+diags(sub,-1))
    u = np.zeros(shape=(N,M+1))
    u[:,0] = initial_value(np.linspace(0,1,N+1)[1:])
    # print(u[:,0])
    for m in range(1,M+1):
        u[:,m] = (Mass_mat+sp.eye(N))@u[:,m-1]
    return u[:,-1]


def eulerimplicit(N, M):
    # todo 3 b)
    h = 1/N; k = 1/M
    sub = [1]*(N-1)
    sup = [1]*(N-1)
    diag = [-2]*(N)
    sub[-1] = 2
    Mass_mat = k/h**2 * (diags(sup,1)+diags(diag,0)+diags(sub,-1))
    u = np.zeros(shape=(N,M+1))
    u[:,0] = initial_value(np.linspace(0,1,N+1)[1:])
    # print(u[:,0])
    for m in range(1,M+1):
        u[:,m] = spsolve(-Mass_mat+sp.eye(N),u[:,m-1])
    return u[:,-1]
    


#### error analysis ####
nb_samples = 5
N = np.array([2**l for l in range(2,nb_samples+2)]) # todo for 3 c)
M = np.array([4**l for l in range(2,nb_samples+2)])# todo  for 3 c) and 3 d)
l2errorexplicit = np.zeros(nb_samples)  # error vector for explicit method
l2errorimplicit = np.zeros(nb_samples)  # error vector for implicit method
h2k = 1 / (N ** 2) + 1 / M


#### Do not change any code below! ####
try:
    for i in range(nb_samples):
        l2errorexplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1)[1:]) - eulerexplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorexplicit), deg=1)
    if np.isnan(conv_rate[0]):
        raise Exception("Error unbounded for explicit method. Plots not shown.")
    print("Explicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: " + str(
        conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorexplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='$O(h^2+k)$')
    plt.title('$L^2$ convergence rate for explicit method', fontsize=13)
    plt.xlabel('$h^2+k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except Exception as e:
    print(f"Exception: {e}")

try:
    for i in range(nb_samples):
        l2errorimplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1)[1:]) - eulerimplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorimplicit), deg=1)
    if np.isnan(conv_rate[0]):
        raise Exception("Error unbounded for implicit method. Plots not shown.")
    print("Implicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: " + str(
        conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorimplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='$O(h^2+k)$')
    plt.title('$L^2$ convergence rate for implicit method', fontsize=13)
    plt.xlabel('$h^2+k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except Exception as e:
    print(f"Exception: {e}")

plt.show()