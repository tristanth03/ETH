import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
    
def g(s,K):
    return np.maximum(s-K,0)

def mass_mat(N,h):
    sup = [1/6]*(N-1)
    sub = [1/6]*(N-1)
    diag = [2/3]*(N)
    return h*(diags(sup,1)+diags(diag,0)+diags(sub,-1))

def rig_mat(N,r,sigma,h):
    sup = [r*h/6+1/2*(1/2*sigma**2-r)-sigma**2/(2*h)]*(N-1)
    sub = [r*h/6-1/2*(1/2*sigma**2-r)-sigma**2/(2*h)]*(N-1)
    diag = [2*r*h/3+sigma**2/(h)]*(N)
    return (diags(sup,1)+diags(diag,0)+diags(sub,-1))


def solve_fem_two_stage(N, M, R, sigma, r, K, K1, T, T1, theta):
    k = T1/M; h = 2*R/N
    A_mat = rig_mat(N,r,sigma,h)
    M_mat = mass_mat(N,h)
    M0 = int((T1-T)/k-1)
    M1 = int((T/k)-1)
    LHS = (M_mat+k*theta*A_mat)

    u = np.zeros(shape=(N,M0+1))
    v = np.zeros(shape=(N,M1+1))
    u[:,0] = g(np.exp(np.linspace(-R,R,N+2)[1:-1]),K1)
    for m in range(1,M0+1):
        RHS = (M_mat-k*(1-theta)*A_mat)@u[:,m-1]
        u[:,m] = spsolve(LHS,RHS)

    
    v[:,0] = g(u[:,-1],K)

    for m1 in range(1,M1+1):
        RHS1 = (M_mat-k*(1-theta)*A_mat)@v[:,m1-1]
        v[:,m1] = spsolve(LHS,RHS1)


    return np.linspace(-R,R,N+2)[1:-1],u[:,-1],v[:,-1]
    

# Parameters
sigma = 0.3
T = 1
T1 = 1.5
r = .01
K = 10
K1 = 15
R = 6
N = 3*2**8-1
M = N+1

theta = 0.5
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