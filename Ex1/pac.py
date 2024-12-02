import numpy as np
import scipy.linalg as spli
from lyapunov import *
from consts import *

def PAC(Yd0: np.ndarray, DF:np.ndarray, mu: float):
    # Pseudo Arclength Continuation algorithm
    # From a design vector [x0, v0, T] coming from a nonlinear Lyapunov
    # orbit, the algorithm computes the design vector [xE, vE, T] corresponding
    # to another nonlinear orbit of the same family
    # Could be extended to allow n orbit computations

    # Input: 
    # Yd0: Initial design vector, solution of the nonlinear Lyapunov orbit (3x1)
    # DF: Jacobian of the nonlinear Lyapunov orbit (2x3)
    # mu: Mass ratio
    # Output:
    # Yd_k: Design vector of the new orbit (3x1)
    # DF: Jacobian of the new orbit (2x3)
    # F_X: Constraint vector of the new orbit (2x1)
    # Note: The exercise suggests to store the tangent of the curve tau
    #       and the step size delta_s to perform the bifurcation analysis

    delta_s = 1e-3
    tau = spli.null_space(DF) # Tangent of the curve

    Yd_k = Yd0 + delta_s*tau

    # Allocate G and DG
    G = np.ones((3,1))
    DG = np.zeros((3,3))

    while np.linalg.norm(G) > eps:
        DX, DF, F_X = shooting(Yd_k, mu)
        # G_X = [F_X(2x1), tan_tau'*(Y_guess_new - Y_guess_old)-delta_s(1x1)]
        G[0:2] = F_X
        G[2] = tau.T@(Yd_k - Yd0) - delta_s
        if np.linalg.norm(G) > eps:
            # Correct the design vector
            # DG = [DF, tau.T]
            DG[0:2, :] = DF
            DG[2, :] = tau.T
            # Compute the correction
            Yd_k = Yd_k - np.linalg.inv(DG)@G
        else:
            break
    return Yd_k, DF, F_X



    

