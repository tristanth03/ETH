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
    sub = sup
    diag = [4/6]*(N)

    return h*(diags(sup,offsets=1)+diags(diag,offsets=0)+diags(sub,offsets=-1))


def buildABS(N, h, sigma, r):
    sup = [-1/(2*h) * sigma**2+0.5*(sigma**2/2-r)+(r*h)/6]*(N-1)
    sub = [-1/(2*h) * sigma**2-0.5*(sigma**2/2-r)+(r*h)/6]*(N-1)
    diag = [(4*r*h)/6+sigma**2/h]*(N)

    return (diags(sup,offsets=1)+diags(diag,offsets=0)+diags(sub,offsets=-1))



# Solver
def FEM_theta(N, M, R, B, r, sigma, K, T, theta):
    k = T/M; h = (R+np.log(B/K))/(N+1)
    A_mat = buildABS(N,h,sigma,r)
    mass_mat = buildMassBS(N,h)
    C = (1/k)*mass_mat-(1-theta)*A_mat
    D = (1/k)*mass_mat+(theta)*A_mat
    u = np.zeros(shape=(N+2,M+1))
    x = np.linspace(-R,np.log(B/K),num=N+2)
    u[:,0] = initial_value(x)


    for t in range(1,M+1):
        RHS = C@u[1:-1,t-1]
        u[1:-1,t] = spsolve(D,RHS)
        # print(u[:,t])
 
    return u[1:-1,-1]



if __name__ == '__main__':
    # Set Parameters
    N = 255                # number of nodes
    M = 256                     # number of time steps
    R = 3                       # localization
    T = 1                       # maturity
    K = 60                      # strike
    r = 0.01                    # interest rate
    sigma = 0.3                 # volatility
    B = 80                      # barrier

    x = np.linspace(-R, np.log(B/K), N + 2)[1:-1]   # uniform grid in log-price
    S = np.exp(x)*K                                 # corresponding real prices

    # compute up-and-out barrier
    uout = FEM_theta(N=N,M=M,R=R,B=B,r=r,sigma=sigma,K=K,T=T,theta=0.5)   # price of up-and-out barrier option

    # # transform solution (Log-moneyness)
    uout = uout*K


    # # compute up-and-in barrier
    ubs = bseucall(S, T, K, r, sigma)
    uin = ubs-uout

    # # Postprocessing
    # # area of interest
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
    # plt.savefig('price.eps', format='eps')
    plt.show()