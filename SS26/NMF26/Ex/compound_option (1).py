import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
    

def mass_mat(N,h):
    sub = [1/6]*(N-1)
    sup = [1/6]*(N-1)
    diag = [2/3]*(N)

    return h* (diags(sup,1)+diags(sub,-1)+diags(diag,0))

def rig_mat(N,h,r,sigma):
    sup = [r*h / 6 + (sigma**2 /2-r)*0.5 - 0.5*sigma**2/h]*(N-1)
    sub = [r*h / 6 - (sigma**2/2-r)*0.5 - 0.5*sigma**2/h]*(N-1)
    diag = [2/3 * r*h + sigma**2 /h]*(N)

    return (diags(sup,1)+diags(sub,-1)+diags(diag,0))

def initial(x,K):
    return np.maximum((x-K),0)

def solve_fem_two_stage(N, M, R, sigma, r, K, K1, T, T1, theta):
    k = T1/M
    h = 2*R / N
    M1 = int((T1-T)/k-1)
    M2 = int((T/k)-1)
    A_mat = rig_mat(N-1,h,r,sigma)
    M_mat = mass_mat(N-1,h)

    LHS = (M_mat+k*theta*A_mat)
    u = np.zeros(shape=(N-1,M1+1))
    u[:,0] = initial(np.exp(np.linspace(-R,R,N+1)[1:-1]),K1)
    for m in range(1,M1+1):
        RHS1 = (M_mat-k*(1-theta)*A_mat)@u[:,m-1]
        u[:,m] = spsolve(LHS,RHS1)

    v = np.zeros(shape=(N-1,M2+1))
    v[:,0] = initial(u[:,-1],K)
    for m in range(1,M2+1):
        RHS2 = (M_mat-k*(1-theta)*A_mat)@v[:,m-1]
        v[:,m] = spsolve(LHS,RHS2)

    return np.linspace(-R,R,N+1)[1:-1],u[:,-1],v[:,-1]


    

# Parameters
sigma = 0.3
T = 1
T1 = 1.5
r = 0.01
K = 10
K1 = 15
R = 6
N = 3*2**8-1
M = 3*2**8

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