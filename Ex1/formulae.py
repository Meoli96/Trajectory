import numpy as np
from scipy.integrate import solve_ivp
# This files contains formulae related to the problem at hand.
# In particular:
# - Potential U, gradient dU, and Hessian d2U
from potential import U, dU, d2U


def A(r: np.ndarray, mu: float) -> np.ndarray:
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

    return A

# Dynamic system exploiting the state transition matrix PHI

def f(t: float, Y: np.ndarray, T:float, mu: float) -> np.ndarray:
    # Y contains the state vector [x, y, u, v] and the state transition matrix PHI
    # T is the period of the orbit, mu is the mass ratio
    x, y, u, v = Y[:4]
    PHI = Y[4:].reshape((4, 4))
    
    # Compute the Jacobian A
    A_ = A(np.array([x, y]), mu)
    # Compute the potential gradient dU
    dU_ = dU(np.array([x, y]), mu)

    # State derivation
    dy = np.zeros(4)
    # f = [u, v, 2*v + dU_x, -2*u + dU_y]
    dy[:4] = T*np.array([u, v, 2*v + dU_[0], -2*u + dU_[1]])
    dPHI = T*A_ @ PHI
    
    # Now stitch the state vector and the state transition matrix
    return np.concatenate((dy, dPHI.flatten()))

   

def shooting(Y_guess, Yt, mu):
    # This function "shoots" the initial guess Y0 to find the orbit with two crossing points
    # perpendicular to the x-axis
    # It integrates over tau, such as T = t*tau is the orbital period, and the final condition
    # is evaluated at tau = 1/2 

    # Design variables X = [x0, v0, T]
    # Constraint vector C = [y(T/2), vx(T/2)] = 0

    # Initial time
    tau0 = 0
    # Final time
    tau1 = 1/2

    x0 = Y_guess[0]
    v0 = Y_guess[1]
    T = Y_guess[2]

    # Initial state vector
    Y0 = np.array([x0, 0, 0, v0])
    
    # Initial conditions
    Y0 = np.concatenate((Y0, np.eye(4).flatten())) # Expand with the STM
    sol = solve_ivp(f, [tau0, tau1], Y0, args=(T, mu), method='LSODA')

    # Final conditions
    Yf = sol.y[:, -1]
    # Final state vector, take only y and u
    F_X = Yf[1:2]
    PHIf = Yf[4:].reshape(4, 4)
    # Compute corrections 

    DF = np.zeros((2,3))
    f0 = f(tau0, Y0, T, mu)
    # Strip the state transition matrix from f0
    f0 = f0[:4]

    # The 2x2 block of DF is composed by by the second
    # and third row and the first and fourth column of PHIf
    # [PHI_yx, PHI_yv]
    # [PHI_ux, PHI_uv]

    DF[0,0] = PHIf[1, 0] # PHI_yx
    DF[1,0] = PHIf[2, 0] # PHI_ux
    DF[0,1] = PHIf[1, 3] # PHI_yv
    DF[1,1] = PHIf[2, 3] # PHI_uv

    # Last column is 0.5*PHI_f(1,:)*f0  
    #                0.5*PHI_f(2,:)*f0

    DF[0,2] = 0.5*PHIf[1, :].dot(f0)
    DF[1,2] = 0.5*PHIf[2, :].dot(f0)
    # Now solve for dx
    # Compute the pseudo inverse of DF
    pi = np.linalg.inv(DF.T@DF)@DF.T
    # Compute the correction
    dX = -pi@F_X
    
    return dX, DF, F_X

    



