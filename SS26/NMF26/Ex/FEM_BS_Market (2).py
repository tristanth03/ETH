import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import numpy.linalg as lin
from scipy.stats import norm

def build_massMatrix(N,R):
    h = 2*R /(N+1)
    sup = [1/6]*(N-1)
    sub = [1/6]*(N-1)
    diag = [2/3]*(N)
    return h*(sp.diags(sup,1)+sp.diags(sub,-1)+sp.diags(diag,0))

def build_BSMatrix(N,sigma,r,R):
    h = 2*R /(N+1)
    sup = [r*h / 6 - 0.5*sigma**2/h + 0.5*(sigma**2/2-r)]*(N-1)
    sub = [r*h / 6 - 0.5*sigma**2/h - 0.5*(sigma**2/2-r)]*(N-1)
    diag = [2/3 * r*h + sigma**2/h]*(N)
    return (sp.diags(sup,1)+sp.diags(sub,-1)+sp.diags(diag,0))

def bs_formula_C(s,t,sigma,K,r):
    d1 = (np.log(s/K)+(sigma**2/2+r)*t)/(sigma*np.sqrt(t))
    d2 = d1-sigma*np.sqrt(t)
    return s*norm.cdf(d1)-K*np.exp(-r*t)*norm.cdf(d2)

def bs_formula_BF(s,t,sigma,K0,K2,r):
    K1 = (K0+K2)/2
    return bs_formula_C(s,t,sigma,K0,r)-2*bs_formula_C(s,t,sigma,K1,r)+bs_formula_C(s,t,sigma,K2,r)

def u0(x,K):
    return np.maximum((np.exp(x)-K*0.5),0)-2*np.maximum((np.exp(x)-K),0)+np.maximum((np.exp(x)-K*1.5),0)

def exactu(t,x,sigma,K,r):
    val = bs_formula_BF(np.exp(x),t,sigma,0.5*K,1.5*K,r) 
    return val


def FEM_theta(N,M,theta,R,sigma,K,r,T):
    k = T/M
    grid = np.linspace(-R,R,N+2)
    grid = grid[1:-1]
    u_sol = u0(grid,K).reshape(N,1)
    MatrixM = build_massMatrix(N,R)
    MatrixA = build_BSMatrix(N,sigma,r,R)
    B_theta = MatrixM + k*theta*MatrixA
    C_theta = MatrixM - k*(1-theta)*MatrixA
    F=0
    
    B_theta = B_theta.tocsr() 
    C_theta = C_theta.tocsr()
    for i in range(M):
        RHS = C_theta@u_sol + F
        u_sol = spsolve(B_theta, RHS).reshape(N,1)
    return u_sol


#### Parameter setting
sigma = 0.3
T = 1
r = 0.05
R = 3
K = 1
G = R/2 
theta= 0.5
nb_samples = 5
N = np.power(2, np.arange(4, 4 + nb_samples))-1
M = np.power(2, np.arange(4, 4 + nb_samples))


#### Do not change any code below! ####
error = np.zeros(nb_samples) 
k =  T / M

grid = np.linspace(-R,R,N[nb_samples-1]+2)
grid = grid[1:-1].reshape(N[nb_samples-1],1)
plt.plot( grid
        , FEM_theta(N[nb_samples-1],M[nb_samples-1],theta,R,sigma,K,r,T)
        , 'r-', label='FEM solution'  # plot with the color red, as line
        )
plt.plot( grid
        , exactu(T,grid,sigma,K,r).reshape(N[nb_samples-1],1)
        , 'b-', label='BS-formula solution'  # plot with the color red, as line
        )

plt.legend()
plt.show()

try:
   for i in range(nb_samples):
      grid = np.linspace(-R,R,N[i]+2)
      grid = grid[1:-1]
      ind = (np.abs(grid)-np.log(K)) < G
      err=exactu(T,grid,sigma,K,r).reshape(N[i],1)-FEM_theta(N[i],M[i],theta,R,sigma,K,r,T)
      error[i]=lin.norm(err[ind],ord=np.inf)
      if np.isnan(error[i])==True:
          raise Exception("Error unbounded. Plots not shown.")
   conv_rate = np.polyfit(np.log(k), np.log(error), deg=1)
   if conv_rate[0]<0:
       raise Exception("Error unbounded. Plots not shown.")
   print("FEM method with theta="+str(theta)+" converges: Convergence rate in discrete $L^inf$ norm with respect to time step $k$: " + str(
        conv_rate[0]))
   plt.figure(figsize=[10, 6])
   plt.loglog(k, error, '-x', label='error')
   plt.loglog(k, k, '--', label='$O(k)$')
   plt.loglog(k, k**2, '--', label='$O(k^2)$')
   plt.title('$Convergence rate', fontsize=13)
   plt.xlabel('$k$', fontsize=13)
   plt.ylabel('error', fontsize=13)
   plt.legend()
   plt.plot()
   plt.show()
   
   plt.figure(figsize=[10, 6])
   RR = np.arange(1.25, 4.25, 0.25)
   NN = 2**7 * RR ** 1-1
   MM = 2**7 * RR ** 0
   NN = NN.astype(int)
   MM = MM.astype(int)
   error2 = np.zeros(len(RR))
   for i in range(len(RR)):
       grid2 = np.linspace(-RR[i],RR[i],NN[i]+2)
       grid2 = grid2[1:-1]
       err2 = exactu(T,grid2,sigma,K,r).reshape(NN[i],1)-FEM_theta(NN[i],MM[i],theta,RR[i],sigma,K,r,T)
       ind2 = (np.abs(grid2)-np.log(K)) < 1
       error2[i] = lin.norm(err2[ind2],ord=np.inf)
   plt.semilogy(RR, error2, '-x', label='error w.r.t R')
   plt.title('$log(error)$ against $R$', fontsize=13)
   plt.xlabel('$R$', fontsize=13)
   plt.ylabel('$log(err)$', fontsize=13)
   plt.legend()
   plt.plot()
   plt.show()
   
except Exception as e:
    print(e)
