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
    for i in range(N): 
        M[i,i] = 2/h+2*(i+1)
        # Note that this would also work = 1/h**2 * kappa_integral(h*(i),h*(i+2)) # to be filled
    # The case of i-j=1 --> i = j+1
    for i in range(1,N):
        M[i, i-1] = -(1/h+1/2+i)# to be filled

    # The case of j-i=1 --> i=j-1
    for i in range(N-1):
        M[i, i+1] = -(1/h+1/2+(i+1)) # to be filled

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
    x = np.array([h*i for i in range(1,N+1)])
    return h/3 * (f(t,x-h/2)+f(t,x)+f(t,x+h/2))


def FEM_theta(N,M,theta):
    # todo 3 e)
    M_mat = build_massMatrix(N)
    A = build_rigidityMatrix(N)
    k = 1/M
    h = 1/(N+1)
    x = np.array([h*i for i in range(1,N+1)])
    B_theta = (1/k *M_mat+theta*A)
    B_theta_inv = lin.pinv(B_theta)
    C_theta = (1/k *M_mat-(1-theta)*A)
    u = np.zeros((N+2,M))
    for j in range(M):
        if j == 0:
            u_m = initial_value(x)
            u[1:-1,j] = u_m
        else:
            u_m = u[1:-1,j-1]

        F_theta = theta*build_F(j*k,N)+(1-theta)*build_F((j-1)*k,N)
        u[1:-1,j] = B_theta_inv@(C_theta@u_m+F_theta)



    return u[1:-1,-1]



### error analysis ####
nb_samples = 5
N = np.array([2**l-1 for l in range(2,nb_samples+2)])# fill in this line for f)-g)

part_f = False
if part_f:
    M = np.array([2**l for l in range(2,nb_samples+2)])
else:
    M = np.array([4**l for l in range(2,nb_samples+2)])

theta_v = np.array([0.3,0.5,1])
theta= theta_v[0] # fill in this line for f)-g)

### Do not change any code below! ####
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