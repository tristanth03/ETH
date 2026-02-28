import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin


# Set how floating-point errors are handled.
# np.seterr(all='raise')  


def initial_value(x):
    return np.sin(np.pi/2 * x)


#### exact solution at t=1 ####
def exact_solution_at_1(x):
    # todo 3 a)
    return np.exp(-np.pi**2 / 4)*np.sin(0.5*np.pi*x)


#### numerical scheme ####
def eulerexplicit(N, M):
    # todo 3 b)
    T = 1; a=0;b=1 # Grid endpoints
    k = T/M; h = (b-a)/(N)
    x = np.linspace(a,b,N+1) # The whole grid including the left boundary
    # the interior (i.e. the open spatial set on the left, for the Neumann boundary is already absorbed into  G)
    x_d = x[1:] 
    v = k/h**2
    G = np.diag([1]*(N-1),k=-1)+np.diag([-2]*N)+np.diag([1]*(N-1),k=1)
    G[-1,-2] = 2 # Neumann (as shown in ex 2)
    C = np.identity(N)+v*G
    u = np.zeros(shape=(N,M+1)) # We iterate from 0 to M, hence M+1, and we skip the left boundar, hence N
    for m in range(M):
        if m == 0:
            u[:,m] = initial_value(x_d)

        u_m = u[:,m]
        u[:,m+1] = C@u_m
    
    u_M = u[:,-1]
        
    return u_M

def eulerimplicit(N, M):
    # todo 3 b)
    T = 1; a=0;b=1 # Grid endpoints
    k = T/M; h = (b-a)/(N)
    x = np.linspace(a,b,N+1) # The whole grid including the left boundary
    # the interior (i.e. the open spatial set on the left, for the Neumann boundary is already absorbed into  G)
    x_d = x[1:] 
    v = k/h**2
    G = np.diag([1]*(N-1),k=-1)+np.diag([-2]*N)+np.diag([1]*(N-1),k=1)
    G[-1,-2] = 2 # Neumann (as shown in ex 2)
    C = np.identity(N)-v*G
    C_inv = lin.pinv(C)
    u = np.zeros(shape=(N,M+1)) # We iterate from 0 to M, hence M+1, and we skip the left boundar, hence N
    for m in range(M):
        if m == 0:
            u[:,m] = initial_value(x_d)

        u_m = u[:,m]
        u[:,m+1] = C_inv@u_m
    
    u_M = u[:,-1]
        
    return u_M

# #### error analysis ####
nb_samples = 5
L = [x for x in range(2,nb_samples+2)]
N = np.asarray([2**(l) for l in L]) # todo for 3 c)
part_d = False # to do  for 3 c) and 3 d)

if part_d:
    M = np.asarray([4**(l) for l in L]) # 3d
else:
    M = np.asarray([2* 4**(l) for l in L]) # 3c
l2errorexplicit = np.zeros(nb_samples)  # error vector for explicit method
l2errorimplicit = np.zeros(nb_samples)  # error vector for implicit method
h2k = 1 / (N ** 2) + 1 / M

def plot_sol(arg=True):
    if arg:
        N_e,M_e = N[-1],M[-1] # Largest grid
        x = np.linspace(0,1,N_e+1)[1:]
        plt.plot(x,exact_solution_at_1(x),label="True")
        plt.plot(x,eulerexplicit(N_e,M_e),'r',label="Explicit")
        plt.plot(x,eulerimplicit(N_e,M_e),'b',label="Implicit")
        plt.legend()
        plt.show()

plot_sol(False)

#### Do not change any code below! ####
try:
    for i in range(nb_samples):
        l2errorexplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1)[1:]) - eulerexplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorexplicit), deg=1)
    if np.isnan(conv_rate[0]):
        raise Exception("Error unbounded for explicit method. Plots not shown.")
    print("Explicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: " + str(
        conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorexplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='$O(h^2+k)$')
    plt.title('$L^2$ convergence rate for explicit method', fontsize=13)
    plt.xlabel('$h^2+k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except Exception as e:
    print(f"Exception: {e}")

try:
    for i in range(nb_samples):
        l2errorimplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1)[1:]) - eulerimplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorimplicit), deg=1)
    if np.isnan(conv_rate[0]):
        raise Exception("Error unbounded for implicit method. Plots not shown.")
    print("Implicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: " + str(
        conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorimplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='$O(h^2+k)$')
    plt.title('$L^2$ convergence rate for implicit method', fontsize=13)
    plt.xlabel('$h^2+k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except Exception as e:
    print(f"Exception: {e}")

plt.show()


# the explicit erro
print(f"Implicit L2 error: {l2errorimplicit.round(6)}")
print(f"Explicit L2 error: {l2errorexplicit.round(6)}")

#### -- Comments from me:
# --> the error handling below is wrong if one uses numpy's raise all, for the catch never happens
#  (it is done at compute not afterwards)
# --> the LaTex handling to display $math$ is wrong 