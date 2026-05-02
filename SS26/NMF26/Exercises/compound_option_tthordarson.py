import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
    

def mass_matrix(N,h):
    sup = [1/6]*(N-1)
    sub = sup
    diag = [4/6]*(N)
    return h*(diags(sup,offsets=1)+diags(sub,offsets=-1)+diags(diag,offsets=0))


def rig_matrix(r,sigma,h,N):
    sup = [-(sigma**2)/(2*h)+0.5*(sigma**2/2-r)+(r*h)/6]*(N-1)
    sub = [-(sigma**2)/(2*h)-0.5*(sigma**2/2-r)+(r*h)/6]*(N-1)
    diag = [(4*r*h)/6+sigma**2/h]*(N)
    return diags(sup,offsets=1)+diags(sub,offsets=-1)+diags(diag,offsets=0)

def u0(x,K1):
    return np.maximum(np.exp(x)-K1,0)

def v0(x,K):
    return np.maximum(x-K,0)

def solve_fem_two_stage(N, M, R, sigma, r, K, K1, T, T1, theta):
    h = 2*R / N
    k = T1 / M
    M_mat = mass_matrix(N-1,h)
    A_bs = rig_matrix(r,sigma,h,N-1)

    RHS_mat = (1/k)*M_mat-(1-theta)*A_bs
    LHS_mat = ((1/k)*M_mat+theta*A_bs).tocsc()

    M_first = int((T1-T)/k-1)
    M_second = int(T/k - 1)
    u = np.zeros(shape=(N+1,M_first+1))
    v = np.zeros(shape=(N+1,M_second+1))
    x = np.linspace(-R,R,num=N+1)
    u[1:-1,0] = u0(x[1:-1],K1)
    for t in range(1,M_first+1):
        RHS = RHS_mat@u[1:-1,t-1]
        u[1:-1,t] = spsolve(LHS_mat,RHS)
    v[1:-1,0] = v0(u[1:-1,-1],K)
    for t in range(1,M_second+1):
        RHS = RHS_mat@v[1:-1,t-1]
        v[1:-1,t] = spsolve(LHS_mat,RHS)

    return x[1:-1],u[1:-1,-1],v[1:-1,-1]
    

# Parameters
sigma = 0.5
T = 1
T1 = 1.5
r = 0.01
K = 10
K1 = 15
R = 6
N = 3* 2**8 -1
M = 3* 2**8

theta = 0.5
# solve_fem_two_stage(N, M, R, sigma, r, K, K1, T, T1, theta)

x, u_final, v_final = solve_fem_two_stage(N, M, R, sigma, r, K, K1, T, T1, theta)

mask = np.abs(x) <= 4
x_plot = x[mask]

plt.plot(np.exp(x_plot), v_final[mask], 'b', linewidth=2, label='Compound option value')
plt.plot(np.exp(x_plot), u_final[mask], 'r', linewidth=2, label='Initial condition/European call value')
plt.xlabel('Spot price')
plt.ylabel('Option value')
plt.title('Compound call in the BS model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()