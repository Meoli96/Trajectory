import numpy as np
# This files contains formulae related to the problem at hand.
# In particular:
# - Potential U, gradient dU, and Hessian d2U
from potential import U, dU, d2U


def A(t: float, r: np.ndarray, mu: float) -> np.ndarray:
    # Jacobian of f: (function of position r only)
    #               0  0  1  0
    #               0  0  0  1
    # df/dY = A = Uxx Uxy 0  2
    #             Uyx Uyy -2 0
    x, y = r

    d2U_ = d2U(np.array([x, y]), mu)
    Uxx = d2U_[0, 0]
    Uxy = d2U_[0, 1]
    Uyy = d2U_[1, 1]

    A = np.zeros((4, 4))
    A[0, :] = [0, 0, 1, 0] # df/dx
    A[1, :] = [0, 0, 0, 1] # df/dy
    A[2, :] = [Uxx, Uxy, 0, 2] # df/du
    A[3, :] = [Uxy, Uyy, -2, 0] # df/dv

    return t, A

# Dynamic system exploiting the state transition matrix PHI

def f(t: float, Y: np.ndarray, mu: float) -> np.ndarray:
    # Y contains the state vector [x, y, u, v] and the state transition matrix PHI
    x, y, u, v = Y[:4]
    PHI = Y[4:].reshape((4, 4))
    
    # Compute the Jacobian A
    _, A_ = A(t, np.array([x, y]), mu)
    # Compute the potential gradient dU
    dU_ = dU(np.array([x, y]), mu)

    # State derivation
    dy = np.zeros(4)
    dy[:4] = [u, v, 2*v + dU_[0], -2*u + dU_[1]]
    dPHI = A_ @ PHI
    
    # Now stitch the state vector and the state transition matrix
    return np.concatenate((dy, dPHI.reshape(16)))

   

def shooting(Y0, Yt, mu):
    # This function "shoots" the initial guess Y0 to find the orbit with two crossing points
    # perpendicular to the x-axis
    # It integrates over tau, such as T = t*tau is the orbital period, and the final condition
    # is evaluated at tau = 1/2 
    from scipy.integrate import LSODA
    # Initial time
    tau0 = 0
    # Final time
    tau1 = 1/2

    # Initial conditions
    Y0 = np.concatenate((Y0, np.eye(4).reshape(16)))
    sol = LSODA(f, tau0, Y0, tau1, mu)

    # Final conditions
    Y1 = sol.y[-1]
    # Final state vector
    Yf = Y1[:4]
    # Final state transition matrix
    PHIf = Y1[4:].reshape((4, 4))
    # Compute error and correction step
    err = Yt - Yf
    # Compute DF from the state transition matrix
