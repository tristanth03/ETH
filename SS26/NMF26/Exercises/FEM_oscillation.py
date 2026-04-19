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
    mass_mat = build_massMatrix(N)
    rig_mat = build_rigidityMatrix(N)
    h = 1/(N+1)

    u = np.zeros((N,M+1))
    u[:,0] = initial_value(np.linspace(0+h,1-h,num=N))
    for i in range(1,M+1):
        k = (i/M)**beta-((i-1)/M)**beta
        LHS = (mass_mat+k*theta*rig_mat)
        f_m = build_F((i/M)**beta,N)
        f_m_ = build_F(((i-1)/M)**beta,N)

        RHS = (mass_mat-k*(1-theta)*rig_mat)

        u[:,i] = spsolve(LHS,RHS @ u[:,i-1] + k*(theta*f_m+(1-theta)*f_m_))

        # if i%100 == 0:
        #     print(f"{i}/{M}")

    return u[:,-1]





# #### error analysis ####
nb_samples = 3
N = np.power(2, np.arange(9, 9 + nb_samples)) - 1
M = np.power(2, np.arange(9, 9 + nb_samples))
theta = 0.5
# beta = 1 # set beta according to b) and d)

for beta in range(1,20,2):
    u_l = []
    for i in range(nb_samples):
        u_l.append(FEM_theta(N[i],M[i],theta,beta))
    h_l = 1/N
    def h_norm(h,u1,u2):
        norm = 0
        for i in range(len(u1)):
            norm+=(u1[i]-u2[2*i+1])**2
        return np.sqrt(h*norm)
    conv_rate = np.log(h_norm(h_l[0],u_l[0],u_l[1])/h_norm(h_l[1],u_l[1],u_l[2]))/np.log(2) # Estimate the convergence rate
    print(
        f"FEM with theta={theta}, beta={beta}: Convergence rate in discrete l^2 norm with respect to time step $k$: {conv_rate}"
    )

    plt.plot(beta,conv_rate,'ro')
    plt.xlabel(fr"$\beta$")
    plt.ylabel(fr"Convergence rate $p$")
plt.show()





