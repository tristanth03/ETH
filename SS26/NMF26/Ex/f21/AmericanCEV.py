######################################################################
######################################################################
#
# Template file
# AmericanCEV.py:
#
# Pricing of an American option in a CEV model
#
#
######################################################################
######################################################################


######################################################################
# Importing modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from PDAS import PDAS  # Make sure PDAS.py is in the same folder as this file.
# Make sure assembleMatrix.py is in the same folder as this file.
from assembleMatrix import assembleMatrix

######################################################################
# Spot-price grid

def hval(N, R):
    """Inputs:
    N: dimension of VN
    R: truncation parameter for the spot-price

    Returns the log-price grid step"""

    #
    # Your code here
    #
    # return ...

def sgrid(N, R):
    """Inputs:
    N: dimension of VN
    R: truncation parameter for the spot-price

    Returns the spot-price grid [s0,s1,...,sN,sN+1]"""

    #
    # Your code here
    #
    # return ...


######################################################################
# Time grid


def Mval(N, tau, R):
    """Inputs:
    N: dimension of VN
    R: truncation parameter for the spot price
    tau: size of the time interval for the theta-scheme

    Returns the number M of time steps to perform in the theta-scheme """

    h = hval(N, R)
    #
    # Your code here
    #
    # return ...


def kval(tau,M):
    """Inputs:
    tau: size of the time interval for the theta-scheme
    M: number of steps to perform in the theta-scheme


    Returns the time-step size k"""

    #
    # Your code here
    #
    # return ...

######################################################################
# FEM assembling routines


def assemble_M(N, R, mu):
    """Inputs:
    N: dimension of VN
    R: spot price truncation parameter
    mu: parameter of the CEV model

    Returns the NxN matrix M^{CEV}
    """

    #
    # Your code here
    #
    # return ...


def assemble_A(N, R, r, sigma, rho, mu):
    """Inputs:
    N: dimension of VN
    R: spot price truncation parameter
    r: riskless interest rate
    sigma, rho: CEV volatility = sigma*rho^{s}
    mu: parameter of the CEV model

    Returns the NxN matrix A^{CEV}
    """

    #
    # Your code here
    #
    # return ...


# Matrix B
def assemble_B(M, A, k):
    """Inputs:
    M: Matrix M^{CEV} (mass matrix)
    A: matrix A^{CEV}
    k: time-step size

    Returns the matrix B defined in the exam
    """

    #
    # Your code here
    #
    # return ...

######################################################################
# FEM plotting routine


def plotFE(R, vec, lab, leftLim, rightLim):
    """Inputs:
    R: spot price truncation constant
    vec: a 1-dimensional numpy array
    lab: a chain of character (used in legend of plot)
    leftLim, rightLim: interval on which the function is plotted

    Plots the graph of the FE function whose coordinates are given by vec.
    """
    N = vec.size
    si = sgrid(N, R)
    vals = 0*si
    vals[1:-1] = vec
    ind = (si >= leftLim) & (si <= rightLim)
    plt.plot(si[ind], vals[ind], label=lab)


######################################################################
# Option pricing routines

# American option
def computeCEVAmericanOption(N, R, r, sigma, rho, mu, T, gvec):
    """ Inputs:
    N: number of dofs
    R: such that G = (0,R)
    r: riskless interest rate
    rho, sigma: sigma rho^s is the underlying volatility
    mu: parameter of the FEM
    T: option maturity
    gvec: vector (g(s_1),...,g(s_N)), where g is the payoff and
    s_i are the spot price grid points.

    Output:
    Vector v approx (v(T,s_1),...,v(T,s_N)) where
    v(t,s) is the value of the American option at time-to-maturity t and spot price s
    """

    #
    # Your code here
    #
    # return ...



print('Pricing of an American put option')

# Grid definition
R = 0  # Truncation parameter in spot price
N = 0  # Number of dofs
si = 0 # spot-price grid [s0,...,sN+1]
sdof = 0 # array [s1,...,sN]

print('Resolution on (0,{R}) with {N} degrees of freedom'.format(R=R, N=N))

# Market model: CEV
rho = 0  # CEV model constant rho
sigma = 0  # so that sigma(s) = sigma*rho^{s}
mu = 0 # parameter mu of the method
r = 0  # Interest rate

print('CEV model: rho = {rho}, sigma = {sigma}, r = {r}'.format(
    rho=rho, sigma=sigma, r=r))
print(
    'Setting parameter mu to {mu} -> rho + mu = {rhopmu}'.format(mu=mu, rhopmu=rho+mu))

# Option definition
T = 0  # Maturity
K = 0  # Strike price
# Payoff
def gput(S):
    K = 0 # Defined two times to avoid constant variables overwriting
    #
    # Your code here
    #
    # return ...


print('Put option with maturity {T} and strike price {K}'.format(T=T, K=K))
gvec = 0 # numpy array [g(s1),...,g(sN)] where g is the payoff

# Area of interest
leftLim = 0
rightLim = 0
print('Pricing for stock prices s in ({l},{r})'.format(l=leftLim, r=rightLim))

# Computation of option values:
print('Computing put option price...')
Vam = 0 # Nx1 numpy array approximating [Vam(t,s1),...,Vam(t,sN)]
print('... Done!')


# Plot American put
print('Plotting results')
plotFE(R, Vam, 'CEV American put',leftLim, rightLim)
ind = (sdof > leftLim)*(sdof < rightLim)
plt.plot(sdof[ind], gput(sdof[ind]), 'k--', label='Payoff')
plt.xlabel('Spot price s')
plt.ylabel('Value')
plt.legend()
plt.show()
