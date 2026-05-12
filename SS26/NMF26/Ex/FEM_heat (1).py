import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def kappa_integral(x,y):
    # todo 3 a)
    return 0.5*(y**2-x**2)+(y-x)

def build_massMatrix(N):
    # todo 3 b)
    h = 1/(N+1)
    sup = [1]*(N-1)
    sub = [1]*(N-1)
    diag = [4]*(N)
    return (h/6)*(sp.diags(sup,offsets=1)+sp.diags(diag,offsets=0)+sp.diags(sub,offsets=-1))

def build_rigidityMatrix(N):
    # todo 3 b)
    # Be careful with the indices!
    # kappa_integral could be helpful here
    h = 1/(N+1)
    mat = np.zeros((N,N))
    # <--- careful of indices!!!
    # The case of i=j
    for i in range(N):

        mat[i,i] = h**(-2)*(kappa_integral((i-1+1)*h,(i+1)*h)+kappa_integral((i+1)*h,(i+1+1)*h))  # to be filled

    # The case of i-j=1 --> i = j+1
    for i in range(1,N):
        mat[i, i-1] = - h**(-2)*(kappa_integral((i-1+1)*h,(i+1)*h)) # to be filled

    # The case of j-i=1
    for i in range(N-1):
        mat[i, i+1] = - h**(-2)*(kappa_integral((i+1)*h,(i+1+1)*h)) # to be filled

    return mat


def f(t,x):
    # todo 3 c)
    f_sum = ((x+1)*np.pi**2-1)*np.exp(-t)*np.sin(np.pi*x)
    f_sum += -np.pi*np.exp(-t)*np.cos(np.pi*x)

    return f_sum
    
def initial_value(x):
    # todo 3 c)
    return np.sin(np.pi*x)

def exact_solution_at_1(x):
    # todo 3 c)
    return np.exp(-1)*np.sin(np.pi*x)

def build_F(t,N):
    # todo 3 d)
    h = 1/(N+1)
    x = np.linspace(0,1,num=N+2)
    return (h/3)*(f(t,x[1:-1]-h/2)+f(t,x[1:-1])+f(t,x[1:-1]+h/2))

def FEM_theta(N,M,theta):
    # todo 3 e)
    k = 1/M
    mass_mat = build_massMatrix(N)
    rig_mat = build_rigidityMatrix(N)
    B_theta = mass_mat+k*theta*rig_mat
    C_theta = mass_mat-k*(1-theta)*rig_mat
    u = np.zeros(shape=(N+2,M+1))
    x = np.linspace(0,1,num=N+2)

    u[1:-1,0] = initial_value(x[1:-1])
    for t in range(1,M+1): # Careful with multiplycation of k !
        LHS = C_theta@u[1:-1,t-1]+k*((theta*build_F(t*k,N)+(1-theta)*build_F((t-1)*k,N)))
        u[1:-1,t] = spsolve(B_theta,LHS.T)
    return u[1:-1,-1]


#### error analysis ####
nb_samples = 5
l = np.arange(2,5+2)
N = 2**l-1# fill in this line for f)-g)
M = 2**l # fill in this line for f)-g)
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