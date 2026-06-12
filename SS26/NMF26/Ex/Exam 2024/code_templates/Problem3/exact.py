import numpy as np
from scipy.stats import ncx2

def circpl_CIR(r0, alpha, beta, sigma, T, T1, K):
    kappa = beta
    theta = alpha / kappa

    # Variables
    h = np.sqrt(kappa**2 + 2 * sigma**2)

    # T1-dependent 
    AT1 = (2 * h * np.exp((kappa + h) * T1 / 2) / (2 * h + (kappa + h) * (np.exp(T1 * h) - 1)))**((2 * kappa * theta) / sigma**2)
    BT1 = 2 * (np.exp(T1 * h) - 1) / (2 * h + (kappa + h) * (np.exp(T1 * h) - 1))

    # T-dependent 
    AT = (2 * h * np.exp((kappa + h) * T / 2) / (2 * h + (kappa + h) * (np.exp(T * h) - 1)))**((2 * kappa * theta) / sigma**2)
    BT = 2 * (np.exp(T * h) - 1) / (2 * h + (kappa + h) * (np.exp(T * h) - 1))

    # T-T1-dependent
    ATT1 = (2 * h * np.exp((kappa + h) * (T1 - T) / 2) / (2 * h + (kappa + h) * (np.exp((T1 - T) * h) - 1)))**((2 * kappa * theta) / sigma**2)
    BTT1 = 2 * (np.exp((T1 - T) * h) - 1) / (2 * h + (kappa + h) * (np.exp((T1 - T) * h) - 1))

    # price cir
    PT1 = AT1 * np.exp(-BT1 * r0)
    PT = AT * np.exp(-BT * r0)

    # call option on simply compounded interest rate
    rhat = np.log(ATT1 * (1 + K * (T1 - T))) / BTT1
    rho = 2 * h / (sigma**2 * (np.exp(h * T) - 1))
    psi = (kappa + h) / sigma**2
    x1 = 2 * rhat * (rho + psi + BTT1)
    x2 = 2 * rhat * (rho + psi)
    v = 4 * kappa * theta / sigma**2
    
    delta1 = 2 * rho**2 * r0 * np.exp(h * T) / (rho + psi + BTT1)
    delta2 = 2 * rho**2 * r0 * np.exp(h * T) / (rho + psi)

    C = PT1 * ncx2.cdf(x1, v, delta1) - 1 / (1 + K * (T1 - T)) * PT * ncx2.cdf(x2, v, delta2)
    P = (1 + K * (T1 - T)) * C

    return P