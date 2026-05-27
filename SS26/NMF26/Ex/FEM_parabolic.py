import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy.linalg as lin



def alpha(x):
    # Todo : Implement the function alpha

    return 1+x**2


def beta(x):
    # Todo : Implement the function beta
    return 2*x


def gamma(x):
    # Todo : Implement the function gamma
    return np.pi**2 * x**2



def build_massMatrix(N):
    # Todo : Implement the function build_massMatrix
    h = 1/(N+1)
    sup = [1/6]*(N-1)
    sub = [1/6]*(N-1)
    diag = [2/3]*(N)

    return h* (sp.diags(sup,offsets=1)+sp.diags(diag,offsets=0)+sp.diags(sub,offsets=-1))


def simpson(x1,x2,g):
    h = x2-x1
    return h/6 * (g(x2)+4*g((x2+x1)/2)+g(x1))

def build_rigidityMatrix(N, alpha, beta, gamma):

    A = np.zeros(shape=(N,N))
    h = 1/(N+1)
    for i in range(0,N):
        if i > 0:
            xi = i*h; xip = (i+1)*h
            gir = lambda x: (xip-x)
            gil = lambda x: (x-xi)

            j = i-1
            A[j,j] = A[j,j]+(
                            1/h**2 * simpson(xi,xip,alpha)
                            -1/h**2 * simpson(xi,xip,lambda x: beta(x)*gir(x))
                            +1/h**2 * simpson(xi,xip,lambda x: gamma(x)*gir(x)**2))
            
            A[j+1,j+1] = A[j+1,j+1]+(
                            1/h**2 * simpson(xi,xip,alpha)
                            +1/h**2 * simpson(xi,xip,lambda x: beta(x)*gil(x))
                            +1/h**2 * simpson(xi,xip,lambda x: gamma(x)*gil(x)**2))
            
            A[j,j+1] = A[j,j+1]+(
                            -1/h**2 * simpson(xi,xip,alpha)
                            +1/h**2 * simpson(xi,xip,lambda x: beta(x)*gir(x))
                            +1/h**2 * simpson(xi,xip,lambda x: gamma(x)*gil(x)*gir(x)))
            
            A[j+1,j] = A[j+1,j]+(
                            -1/h**2 * simpson(xi,xip,alpha)
                            -1/h**2 * simpson(xi,xip,lambda x: beta(x)*gil(x))
                            +1/h**2 * simpson(xi,xip,lambda x: gamma(x)*gir(x)*gil(x)))
        if i == 0:
            xi = 0*h; xip = 1*h
            gir = lambda x: (xip-x)
            gil = lambda x: (x-xi)
            A[i,i] = A[i,i]+(
                1/h**2 * simpson(xi,xip,alpha)
                +1/h**2 * simpson(xi,xip,lambda x: beta(x)*gil(x))
                +1/h**2 * simpson(xi,xip,lambda x: gamma(x)*gil(x)**2))

        if i == N-1:
            xi = N*h; xip = (N+1)*h
            gir = lambda x: (xip-x)
            gil = lambda x: (x-xi)

            A[i,i] = A[i,i]+(
                            1/h**2 * simpson(xi,xip,alpha)
                            -1/h**2 * simpson(xi,xip,lambda x: beta(x)*gir(x))
                            +1/h**2 * simpson(xi,xip,lambda x: gamma(x)*gir(x)**2))    

    return sp.csr_matrix(A) 
            
def f(t, x):
    # Todo : Implement the function f
    return np.exp(-t)*np.sin(np.pi * x) * (2*np.pi**2 * x**2 + np.pi**2 - 1)


def initial_value(x):
    # Todo : Implement the function initial_value
    return np.sin(np.pi*x)


def exact_solution_at_1(x):
    # Todo : Implement the function exact_solution_at_1
    return np.exp(-1)*np.sin(np.pi*x)


# def build_F(t, N):
#     # Todo : Implement the function build_F
#     f_vec = np.zeros(shape=(N))
#     h = 1/(N+1)
#     for i in range(N):
#         xim = (i)*h; xi = (i+1)*h; xip = (i+2)*h
#         gir = lambda x: (xip-x)
#         gil = lambda x: (x-xim)

#         f_vec[i] = simpson(xim,xi,lambda x: f(t,x)*gil(x))+simpson(xi,xip,lambda x: f(t,x)*gir(x))
#     return 1/h * f_vec

def build_F(t, N):
    h = 1 / (N + 1)
    x = (np.arange(N) + 1) * h
    print(x.shape)
    return h / 3 * (f(t, x - h/2) + f(t, x) + f(t, x + h/2))

def FEM_theta(N, M, theta):
    # Todo : Implement the theta scheme, return the solution u_sol at final time
    u = np.zeros(shape=(N,M+1))
    A_mat = build_rigidityMatrix(N,alpha,beta,gamma)
    M_mat = build_massMatrix(N)
    k = 1/M
    LHS = (M_mat+k*theta*A_mat)
    u[:,0] = initial_value(np.linspace(0,1,N+2)[1:-1])

    for m in range(1,M+1):
        RHS = (M_mat-k*(1-theta)*A_mat)@u[:,m-1]+k*(theta*(build_F((m)*k,N))+(1-theta)*build_F((m-1)*k,N))
        print(RHS.shape)
        u[:,m] = spsolve(LHS,RHS)
    return u[:,-1]

#### error analysis ####
nb_samples = 5
l = np.arange(2,2+nb_samples)
N = 2**l-1# fill in this line for 3c)-d)
M = 2**l# fill in this line for 3c)-d)

# l = np.arange(2,2+nb_samples)
# N = 2**l-1# fill in this line for 3c)-d)
# M = 4**l# fill in this line for 3c)-d)
print(M)

theta = 0.5 # fill in this line for 3c)-d)

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
