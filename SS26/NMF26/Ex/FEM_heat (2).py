import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def kappa_integral(y,x):
    # todo 3 a)
    return (y-x)+(y**2-x**2)/2


def build_massMatrix(N):
    # todo 3 b)
    sup = [1/6]*(N-1)
    sub = [1/6]*(N-1)
    diag = [2/3]*(N)
    h = 1/(N+1)
    return h*(sp.diags(sup,1)+sp.diags(diag,0)+sp.diags(sub,-1))


def build_rigidityMatrix(N):
    # todo 3 b)
    # Be careful with the indices!
    # kappa_integral could be helpful here
    
    M = np.zeros((N,N))
    h = 1/(N+1)
    # The case of i=j
    for i in range(N):
        xim = i*h; xi = (i+1)*h; xip = (i+2)*h
        M[i,i] = kappa_integral(xi,xim)+kappa_integral(xip,xi)  # to be filled
        if i < N-1:
            M[i,i+1] = -kappa_integral(xip,xi)
            M[i+1,i] = -kappa_integral(xip,xi)
    return M/h**2


def f(t,x):
    # todo 3 c)
    return ((x+1)*np.pi**2-1)*np.exp(-t)*np.sin(np.pi*x)-np.pi*np.exp(-t)*np.cos(np.pi*x)
    

def initial_value(x):
    # todo 3 c)
    return np.sin(np.pi*x)


def exact_solution_at_1(x):
    # todo 3 c)
    return np.exp(-1)*np.sin(np.pi*x)


def build_F(t,N):
    # todo 3 d)
    x = np.linspace(0,1,N+2)[1:-1]
    h = 1/(N+1)
    return h*(f(t,x-h/2)+f(t,x)+f(t,x+h/2))/3


def FEM_theta(N,M,theta):
    # todo 3 e)
    k = 1/M
    A_mat = build_rigidityMatrix(N)
    M_mat = build_massMatrix(N)
    LHS = (M_mat+k*theta*A_mat)
    u=np.zeros(shape=(N,M+1))
    u[:,0] = initial_value(np.linspace(0,1,N+2)[1:-1])
    for m in range(1,M+1):
        RHS = (M_mat-k*(1-theta)*A_mat)@u[:,m-1]+k*(theta*build_F(k*m,N)+(1-theta)*build_F(k*(m-1),N))
        u[:,m] = spsolve(LHS,RHS.T)
    print(u[:,-1])
    return u[:,-1]


#### error analysis ####
nb_samples = 5
l = np.arange(2,7,1)
N = 2**l-1# fill in this line for f)-g)
M = 2**l# fill in this line for f)-g)
theta= 0.5 # fill in this line for f)-g)

#### Do not change any code below! ####
l2error = np.zeros(nb_samples) 
k =  1 / M

try:
   for i in range(nb_samples):
      l2error[i] = (1 / (N[i]+1)) ** (1 / 2) * lin.norm(exact_solution_at_1((1/(N[i]+1))*(np.arange(N[i])+1)) - FEM_theta(N[i], M[i],theta), ord=2)
      if np.isnan(l2error[i])==True:
          raise Exception("Error unbounded. Plots not shown.")
   conv_rate = np.polyfit(np.log(k), np.log(l2error), deg=1)
   if conv_rate[0]<0:
       raise Exception("Error unbounded. Plots not shown.")
   print(f"FEM method with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}")
   plt.figure(figsize=[10, 6])
   plt.loglog(k, l2error, '-x', label='error')
   plt.loglog(k, k, '--', label='$O(k)$')
   plt.loglog(k, k**2, '--', label='$O(k^2)$')
   plt.title('$L^2$ convergence rate', fontsize=13)
   plt.xlabel('$k$', fontsize=13)
   plt.ylabel('error', fontsize=13)
   plt.legend()
   plt.plot()
   plt.show()
except Exception as e:
    print(e)