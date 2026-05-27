from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from assembleMatrix import assembleMatrix
from assembleMatrix import assembleMatrixAm

def buildMassCIR(N, R, mu):

    return assembleMatrixAm(N,R,lambda x: x**(2*mu))


def buildACIR(N, R, alpha, beta, sigma, mu):
    return assembleMatrix(N,R,
                          lambda x: x**(1+2*mu) * 0.5*sigma**2,
                          lambda x: x**(2*mu) * (sigma**2*(0.5+mu)+beta*x-alpha),
                          lambda x: x**(2*mu+1))


def g(s, T, T1, K):

    return (T1-T)*np.maximum(K-(1-s)/((T1-T)*s),0)


def FEM_theta(N, k, R, t, alpha, beta, sigma, mu, u_init, theta):
    M =  int(t/k)
    A_mat = buildACIR(N,R,alpha,beta,sigma,mu)
    M_mat = buildMassCIR(N,R,mu)
    print(A_mat.shape)
    print(M_mat.shape)

    LHS = (M_mat+k*theta*A_mat)
    u=np.zeros(shape=(N+1,M+1))
    u[:,0] = u_init
    for m in range(1,M+1):
        RHS = (M_mat-k*(1-theta)*A_mat)@u[:,m-1]
        u[:,m] = spsolve(LHS,RHS)

    return u[:,-1]


def plot_FEM(fig, ax, rightlim, u_sol, N, R, label, markersize=2):
    grid = np.linspace(0, R, N + 2)[:-1]
    I = grid <= rightlim
    ax.plot(grid[I], u_sol[I], "rx-", label=label, markersize=markersize, linewidth=0.5)


if __name__ == "__main__":
    # Parameters
    T = 2
    T1 = 4
    K = 0.05
    R = 4
    alpha = 0.05 * 0.2
    beta = 0.2
    sigma = 0.1

    N = 2**9 - 1
    k = T1 / (N + 1)
    theta = 0.5
    mu = -0.1

    # Compute zero coupon bond price
    # Initial condition
    u_init = np.ones(shape=(N+1))
    # zero coupon bond price at time T
    u_0 = FEM_theta(N,k,R,T1-T,alpha,beta,sigma,mu,u_init,theta)

    # Compute floorlet price
    u_1 = FEM_theta(N,k,R,T,alpha,beta,sigma,mu,g(u_0,T,T1,K),theta)


    ##################### You do not need to modify any code below this line #####################

    # Plot the zero coupon bond price
    fig_1, ax_1 = plt.subplots()
    plot_FEM(fig_1, ax_1, 4, u_0, N, R, r"$V_0(r, T)$", 0)
    ax_1.set_title(r"Zero coupon bond price at $T=2$")
    ax_1.set_xlabel(r"$r$")
    ax_1.set_ylabel("Price")
    ax_1.legend()
    # plt.savefig(Path.home() / "questions" / "Problem3" / "plot_bond.pdf", format="pdf")
    plt.show(block=False)

    from exact import circpl_CIR

    grid = np.linspace(0, R, 1000)[:-1]
    I = grid <= 0.1
    exact = circpl_CIR(grid[I], alpha, beta, sigma, T, T1, K)

    # Plot the floorlet price
    fig_2, ax_2 = plt.subplots()
    ax_2.plot(
        grid[I], exact, label=r"Exact", linewidth=1.5, linestyle="dashed", color="black"
    )
    plot_FEM(fig_2, ax_2, 0.1, u_1, N, R, r"$V_1(r, t)$")
    ax_2.set_title(r"Floorlet price at $T=2$")
    ax_2.set_xlabel(r"$r$")
    ax_2.set_ylabel("Price")
    ax_2.legend()
    # plt.savefig(Path.home() / "questions" / "Problem3" / "plot_floorlet.pdf", format="pdf")
    plt.show()
