import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import numpy.linalg as lin


def kappa_integral(x,y):
    return 0.5*(y**2-x**2) + y-x


def build_massMatrix(N):
    a = 1/6 * np.ones(N-1)
    M = sp.diags(a,-1) + sp.diags(a,1) + 2/3 * sp.eye(N)
    return 1/(N+1) * M


def build_rigidityMatrix(N):
    M = np.zeros((N,N))
    
    for i in range(N):
        M[i,i] = (N+1)**2 * (kappa_integral((i)/(N+1), (i+1)/(N+1)) + kappa_integral((i+1)/(N+1), (i+2)/(N+1)))

    for i in range(1,N):
        M[i, i-1] = -(N+1)**2 * kappa_integral((i)/(N+1), (i+1)/(N+1))

    for i in range(N-1):
        M[i, i+1] = -(N+1)**2 * kappa_integral((i+1)/(N+1), (i+2)/(N+1))

    return sp.csr_matrix(M)


def f(t, x):
    return x * t


def initial_value(x):
    return ((x >= 0.25) * (x <= 0.75)).astype(float)


def build_F(t, N):
    X = np.linspace(0, 1, N + 2)
    h = np.diff(X)
    a = f(t, X[1:-1])
    ab2 = f(t, (X[:-2] + X[1:-1]) / 2)
    b = f(t, (X[1:-1] + X[2:]) / 2)

    return a * (h[:-1] + h[1:]) / 6 + ab2 * h[:-1] / 3 + b * h[1:] / 3


def FEM_theta(N, M, theta, beta):
    # implement the theta scheme for time step t_j = (j/M)^beta
    A_mat = build_rigidityMatrix(N)
    M_mat = build_massMatrix(N)
    u = np.zeros(shape=(N,M+1))
    u[:,0] = initial_value(np.linspace(0,1,N+2)[1:-1])
    for m in range(1,M+1):
        tm = (m/M)**beta
        tm_ = ((m-1)/M)**beta
        k = tm-tm_
        LHS = (M_mat+k*theta*A_mat)
        F = theta*build_F(tm_,N)+(1-theta)*build_F(tm,N)
        RHS = (M_mat-k*(1-theta)*A_mat)@u[:,m-1]+k*F
        u[:,m] = spsolve(LHS,RHS)
    return u[:,-1]

#### error analysis ####
nb_samples = 3
N = np.power(2, np.arange(9, 9 + nb_samples)) - 1
M = np.power(2, np.arange(9, 9 + nb_samples))
theta = 0.5
beta = 1 # set beta according to b) and d)
u_v = []
for i in range(nb_samples):
    u_v.append(FEM_theta(N[i],M[i],theta,beta))

plt.plot(u_v[2])
plt.show()

print(u_v[2][-1])
def rate_calc(N,u_v):
    norm1 = 0
    norm2 = 0
    for i in range(0,N[0]):
        norm1 += (u_v[0][i]-u_v[1][2*i+1])**2
    
    for i in range(0,N[1]):
        norm2 += (u_v[1][i]-u_v[2][2*i+1])**2
    norm1 *= 1/(N[0]+1)
    norm2 *= 1/(N[1]+1)

    return np.log(np.sqrt(norm1)/np.sqrt(norm2))/np.log(2)


conv_rate = rate_calc(N,u_v)# Estimate the convergence rate

print(
    f"FEM with theta={theta}, beta={beta}: Convergence rate in discrete l^2 norm with respect to time step $k$: {conv_rate}"
)