import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def kappa_integral(x,y):
    # todo 3 a)

    return y**2/2+y - (x**2/2+x)


def build_massMatrix(N):
    # todo 3 b)

    h = 1/(N+1)
    return h*(np.diag([1/6]*(N-1),k=1)+np.diag([4/6]*N)+np.diag([1/6]*(N-1),k=-1))


def build_rigidityMatrix(N):
    # todo 3 b)
    # Be careful with the indices!
    # kappa_integral could be helpful here
    
    M = np.zeros((N,N))
    h = 1/(N+1)

    # The case of i=j
    for i in range(N): # BEGIN WITH ZERO NOT SURE ....
        M[i,i] = 2/h *(1-i*h) # to be filled

    # The case of i-j=1
    for i in range(1,N):
        M[i, i-1] = 1/h *(-h/2+i*h-1)# to be filled

    # The case of j-i=1
    for i in range(N-1):
        M[i, i+1] =  1/h *(h/2+i*h-1) # to be filled

    return M


def f(t,x):
    # todo 3 c)
    return ((x+1)*np.pi**2-1)*np.exp(-t)*np.sin(np.pi*x)-np.pi*np.exp(-t)*np.cos(np.pi*x)
    

def initial_value(x):
    # todo 3 c)
    return np.sin(np.pi*x)


def exact_solution_at_1(x):
    # todo 3 c)
    t = 1
    return np.exp(-t)*np.sin(np.pi*x)


def build_F(t,N):
    # todo 3 d) #### TO BE DONE
    h = 1/(N+1)
    x_ = np.linspace(start=0,stop=1,num=N+2)
    x = x_[1:-1]
    return h/3 * (f(t,x-h/2)+f(t,x)+f(t,x+h/2))


def FEM_theta(N,M,theta):
    # todo 3 e)
    M_mat = build_massMatrix(N)
    A = build_rigidityMatrix(N)
    k = 1/M
    # h = 1/(N+1)
    x_ = np.linspace(start=0,stop=1,num=N+2)
    x = x_[1:-1]
    B_theta = (1/k *M_mat+theta*A)
    B_theta_inv = lin.pinv(B_theta)
    C_theta = (1/k *M_mat-(1-theta)*A)
    t = np.linspace(start=0,stop=1,num=M)
    u = np.zeros((N,M))
    for j in range(M):
        if j == 0:
            u_m = initial_value(x)
            u[:,j] = u_m
        F_theta = theta*build_F(t[j],N)+(1-theta)*build_F(t[j-1],N)
        u_m = u[:,j-1]
        u[:,j] = B_theta_inv@(C_theta@u_m+F_theta)



    return u[:,-1]

N = 100; M = 10_000
theta = 0.5

plt.plot(FEM_theta(N,M,theta))
plt.plot(exact_solution_at_1(np.linspace(0,1,N+1)))
plt.show()

#### error analysis ####
# nb_samples = 5
# N = # fill in this line for f)-g)
# M = # fill in this line for f)-g)
# theta= # fill in this line for f)-g)


#### Do not change any code below! ####
# l2error = np.zeros(nb_samples) 
# k =  1 / M

# try:
#    for i in range(nb_samples):
#       l2error[i] = (1 / (N[i]+1)) ** (1 / 2) * lin.norm(exact_solution_at_1((1/(N[i]+1))*(np.arange(N[i])+1)) - FEM_theta(N[i], M[i],theta), ord=2)
#       if np.isnan(l2error[i])==True:
#           raise Exception("Error unbounded. Plots not shown.")
#    conv_rate = np.polyfit(np.log(k), np.log(l2error), deg=1)
#    if conv_rate[0]<0:
#        raise Exception("Error unbounded. Plots not shown.")
#    print(f"FEM method with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}")
#    plt.figure(figsize=[10, 6])
#    plt.loglog(k, l2error, '-x', label='error')
#    plt.loglog(k, k, '--', label='$O(k)$')
#    plt.loglog(k, k**2, '--', label='$O(k^2)$')
#    plt.title('$L^2$ convergence rate', fontsize=13)
#    plt.xlabel('$k$', fontsize=13)
#    plt.ylabel('error', fontsize=13)
#    plt.legend()
#    plt.plot()
#    plt.show()
# except Exception as e:
#     print(e)