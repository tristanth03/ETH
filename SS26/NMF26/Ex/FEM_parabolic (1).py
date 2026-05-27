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
    sup = [1/6]*(N-1)
    sub = [1/6]*(N-1)
    diag = [2/3]*(N)
    h = 1/(N+1)
    return h*(sp.diags(sup,1)+sp.diags(sub,-1)+sp.diags(diag,0))


def simpson(a,b,f):
    return ((b-a)/6)*(f(a)+4*f((b+a)/2)+f(b))

def build_rigidityMatrix(N, alpha, beta, gamma):
    # Todo : Implement the function build_rigidityMatrix
    A_mat = np.zeros(shape=(N,N))
    h = 1/(N+1)
    for i in range(N):
        if i > 0:
            xi = i*h; xip = (i+1)*h
            j = i-1
            A_mat[j,j] = A_mat[j,j]+(
                simpson(xi,xip,lambda x: alpha(x))+
                simpson(xi,xip,lambda x: -beta(x)*(xip-x))+
                simpson(xi,xip,lambda x: gamma(x)*(xip-x)**2))
            A_mat[j+1,j+1] = A_mat[j+1,j+1]+(
                simpson(xi,xip,lambda x: alpha(x))+
                simpson(xi,xip,lambda x: beta(x)*(x-xi))+
                simpson(xi,xip,lambda x: gamma(x)*(x-xi)**2))
            A_mat[j,j+1] = A_mat[j,j+1]+(
                simpson(xi,xip,lambda x: -alpha(x))+
                simpson(xi,xip,lambda x: beta(x)*(xip-x))+
                simpson(xi,xip,lambda x: gamma(x)*(x-xi)*(xip-x)))
            A_mat[j+1,j] = A_mat[j+1,j]+(
                simpson(xi,xip,lambda x: -alpha(x))+
                simpson(xi,xip,lambda x: -beta(x)*(x-xi))+
                simpson(xi,xip,lambda x: gamma(x)*(xip-x)*(x-xi)))
        if i == 0:
            A_mat[0,0] = A_mat[0,0]+(
                simpson(0,h,lambda x: alpha(x))+
                simpson(0,h,lambda x: beta(x)*(x-0))+
                simpson(0,h,lambda x: gamma(x)*(x-0)**2))
        if i == N-1:
            A_mat[N-1,N-1] = A_mat[N-1,N-1]+(
                simpson(N/(N+1),1,lambda x: alpha(x))+
                simpson(N/(N+1),1,lambda x: -beta(x)*(1-x))+
                simpson(N/(N+1),1,lambda x: gamma(x)*(1-x)**2))
            
    return sp.csr_matrix(A_mat/h**2)


def f(t, x):
    # Todo : Implement the function f
    return 2*np.pi**2 *x**2*np.exp(-t)*np.sin(np.pi*x)+np.pi**2*np.exp(-t)*np.sin(np.pi*x)-np.exp(-t)*np.sin(np.pi*x)


def initial_value(x):
    # Todo : Implement the function initial_value
    return np.sin(np.pi*x)


def exact_solution_at_1(x):
    # Todo : Implement the function exact_solution_at_1
    return np.exp(-1)*np.sin(np.pi*x)


def build_F(t, N):
    # Todo : Implement the function build_F
    h = 1/(N+1)
    x = np.linspace(0,1,N+2)[1:-1]
    return h/3 * (f(t,x-h/2)+f(t,x)+f(t,x+h/2))


def FEM_theta(N, M, theta):
    # Todo : Implement the theta scheme, return the solution u_sol at final time
    k = 1/M
    A_mat = build_rigidityMatrix(N,alpha,beta,gamma)
    M_mat = build_massMatrix(N)
    LHS = M_mat+k*theta*A_mat
    u = np.zeros(shape=(N,M+1))
    u[:,0] = initial_value(np.linspace(0,1,N+2)[1:-1])
    for m in range(1,M+1):
        RHS = (M_mat-k*(1-theta)*A_mat)@u[:,m-1]+k*(theta*build_F(k*m,N)+(1-theta)*build_F(k*(m-1),N))
        print(((M_mat-k*(1-theta)*A_mat)@u[:,m-1]).shape, (k*(theta*build_F(k*m,N)+(1-theta)*build_F(k*(m-1),N))).shape)
        u[:,m] = spsolve(LHS,RHS)

        
    return u[:,-1]


#### error analysis ####
nb_samples = 5
l = np.arange(2,2+nb_samples)
N = 2**l-1# fill in this line for 3c)-d)
M = 2**l# fill in this line for 3c)-d)
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
