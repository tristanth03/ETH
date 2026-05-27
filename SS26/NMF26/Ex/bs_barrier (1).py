import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

def bseucall(S, T, K, r, sigma):
    """
    Computes the value of a European put option with the Black-Scholes model using analytic formulas.

    Parameters:
        S (float or array-like): Stock prices at time 0.
        T (float): Maturity.
        K (float): Strike.
        r (float): Interest rate.
        sigma (float): Volatility.

    Returns:
        P (float or array-like): Option price at time 0.
    """
    # call option
    d1 = (np.log(S / K) + (0.5 * sigma ** 2 + r) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    P = S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
    return P

# Compute Generator/ Source Term / Initial Data
def initial_value(x):
    return np.maximum((np.exp(x)-1),0)

def buildMassBS(N, h):
    sup = [1/6]*(N-1)
    sub = [1/6]*(N-1)
    diag = [2/3]*(N)

    return h* (diags(sup,1)+diags(sub,-1)+diags(diag,0))

def buildABS(N, h, sigma, r):
    sup = [r*h / 6 + (1/2 * sigma**2 - r)*0.5- (1/2*sigma**2) / h]*(N-1)    
    sub = [r*h / 6 - (1/2 * sigma**2 - r)*0.5- (1/2*sigma**2) / h]*(N-1)    
    diag = [2/3 * r*h+ sigma**2 / h]*(N)

    return (diags(sup,1)+diags(sub,-1)+diags(diag,0))

    


# Solver
def FEM_theta(N, M, R, B, r, sigma, K, T, theta):
    h = (R+np.log(B/K))/(N+1)
    k = T/M

    A_mat = buildABS(N,h,sigma,r)
    M_mat = buildMassBS(N,h)

    u = np.zeros(shape=(N,M+1))
    u[:,0] = initial_value(np.linspace(-R,np.log(B/K),N+2)[1:-1])

    LHS = (M_mat+k*theta*A_mat)
    for m in range(1,M+1):
        RHS = (M_mat-k*(1-theta)*A_mat)@u[:,m-1]
        u[:,m] = spsolve(LHS,RHS)

    
    return u[:,-1]

if __name__ == '__main__':
    # Set Parameters
    N = 255                     # number of nodes
    M = 256                     # number of time steps
    R = 3                       # localization
    T = 1                       # maturity
    K = 60                      # strike
    r = 0.01                    # interest rate
    sigma = 0.3                 # volatility
    B = 80                      # barrier
    theta = 0.5
    x = np.linspace(-R, np.log(B/K), N + 2)[1:-1]   # uniform grid in log-price
    S = np.exp(x)*K                                 # corresponding real prices

    # compute up-and-out barrier
    uout = FEM_theta(N=N,M=M,R=R,B=B,r=r,sigma=sigma,K=K,T=T,theta=theta)                                          # price of up-and-out barrier option

    # transform solution (Log-moneyness)
    uout *= K


    # compute up-and-in barrier
    ubs = bseucall(S, T, K, r, sigma)
    uin = ubs-uout

    # Postprocessing
    # area of interest
    I = np.abs(x) < 0.75


    # plot solution
    plt.figure(1)
    plt.plot(S[I], uout[I], 'bx-', label='Knock-out barrier', markersize=2, linewidth=0.5)
    plt.plot(S[I], uin[I], 'go-', label='Knock-in barrier', markersize=2, linewidth=0.5, fillstyle='none')
    plt.plot(S[I], ubs[I], 'rs-', label='Plain vanilla', markersize=2, linewidth=0.5, fillstyle='none')
    plt.plot(K * np.exp(np.linspace(-0.75, np.log(B/K), 1000)), K * initial_value(np.linspace(-0.75, np.log(B/K), 1000)), 'k-', label='Payoff', linewidth=0.5)
    plt.xlabel('s')
    plt.ylabel('Option price')
    plt.legend(loc='upper right')
    plt.savefig('price.eps', format='eps')
    plt.show()