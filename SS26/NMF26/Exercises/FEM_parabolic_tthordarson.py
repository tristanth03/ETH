import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy.linalg as lin


def simpson(g,a,b):
    h = b-a
    return h/6 * (g(a)+4*g((a+b)/2)+g(b))


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
    return h*(np.diag([1/6]*(N-1),k=1)+np.diag([4/6]*N)+np.diag([1/6]*(N-1),k=-1))




def build_rigidityMatrix(N, alpha, beta, gamma):
    # Todo : Implement the function build_rigidityMatrix
    A = np.zeros(shape=(N,N))
    h = 1/(N+1)
  
    for i in range(0,N+1):
        if i > 0 and i < N:
            F1 = lambda x : 1/h**2 * (alpha(x)-beta(x)*(h*(i+1)-x)+gamma(x)*(h*(i+1)-x)**2)
            F2 = lambda x : 1/h**2 * (alpha(x)+beta(x)*(x-h*i)+gamma(x)*(x-h*i)**2)

            F3 = lambda x : 1/h**2 * (-alpha(x)+beta(x)*(h*(i+1)-x)+gamma(x)*(h*(i+1)-x)*(x-i*h))
            F4 = lambda x : 1/h**2 * (-alpha(x)-beta(x)*(x-h*i)+gamma(x)*(h*(i+1)-x)*(x-i*h))


            A[i-1,i-1] += simpson(F1,i*h,h*(i+1))
            A[i,i] += simpson(F2,i*h,h*(i+1))
            A[i-1,i] += simpson(F3,i*h,h*(i+1))
            A[i,i-1] += simpson(F4,i*h,h*(i+1))

        elif i == 0:
            F2 = lambda x : 1/h**2 * (alpha(x)+beta(x)*(x-h*i)+gamma(x)*(x-h*i)**2)
            A[i,i] += simpson(F2,i*h,h*(i+1))
        else:
            F1 = lambda x : 1/h**2 * (alpha(x)-beta(x)*(h*(i+1)-x)+gamma(x)*(h*(i+1)-x)**2)
            A[i-1,i-1] += simpson(F1,i*h,h*(i+1))


    return A

def f(t, x):
    # Todo : Implement the function f
    return np.sin(np.pi*x)*np.exp(-t)*(2 * np.pi**2 * x**2 + np.pi**2 -1)


def initial_value(x):
    # Todo : Implement the function initial_value
    return np.exp(-0)*np.sin(np.pi*x)


def exact_solution_at_1(x):
    # Todo : Implement the function exact_solution_at_1
    return np.exp(-1)*np.sin(np.pi*x)


def build_F(t, N):
    # Todo : Implement the function build_F
    h = 1/(N+1)
    x = h*np.arange(1,N+1)
    return  h/3 *(f(t,x-h/2)+f(t,x)+f(t,x+h/2))


def FEM_theta(N, M, theta):
    # Todo : Implement the theta scheme, return the solution u_sol at final time
    u = np.zeros(shape=(N,M+1))
    h = 1/(N+1)
    k = 1/M
    x = h*np.arange(1,N+1)
    mass_mat = build_massMatrix(N)
    rig_mat = build_rigidityMatrix(N,alpha,beta,gamma)
    B = (mass_mat+k*theta*rig_mat)
    B_inv = lin.pinv(B)
    C = (mass_mat-k*(1-theta)*rig_mat)

    u[:,0] = initial_value(x)
    for m in range(1,M+1):
            u_m = u[:,m-1]
            F_m = theta*build_F((m)*k,N)+(1-theta)*build_F(((m-1)*k),N)
            u[:,m] = B_inv @ (C @ u_m + k * F_m) 
    
    u_sol = u[:,-1]
    return u_sol

# #### error analysis ####
nb_samples = 5
l = np.arange(5,10)
part_f = False
part_banana = True
if part_f:
    M = np.power(2,l,dtype=int)
    N = np.power(2,l,dtype=int)-1
    print("## Part c -- ")

# elif part_banana:
#     M = np.power(4,l,dtype=int)
#     N = np.power(2,l,dtype=int)-1
#     print("## Part d -- ")
else:
    M = np.power(4,l-3,dtype=int)
    N = np.power(2,l-3,dtype=int)-1
    print("## Part d -- ")

theta = 0.5 # fill in this line for 3c)-d)
print(f"theta = {theta} \n")
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
