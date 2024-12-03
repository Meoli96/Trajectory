import numpy as np
import scipy.linalg as spli
from scipy.integrate import solve_ivp
# This files contains formulae related to the problem at hand.
# In particular:
# - Potential U, gradient dU, and Hessian d2U
from potential import U, dU, d2U, Jacobi


## This file contains the formulae for the problem of the Lyapunov orbit
## in the circular restricted three-body problem. 
# Functions present in this file:


# - f: The nonlinear dynamical system to be integrated
# - A: Jacobian of the dynamical system
# - lin_lyapunov_orbit: Computes the linearized Lyapunov orbit
# - shooting: Implements the shooting method to find the nonlinear periodic orbit
# - nonlin_lyapunov_orbit: Computes the nonlinear Lyapunov orbit



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

   


def A(r: np.ndarray, mu: float) -> np.ndarray:
    # Jacobian of f: (function of position r only)
    #               0  0  1  0
    #               0  0  0  1
    # df/dY = A = Uxx Uxy 0  2
    #             Uyx Uyy -2 0
  

    d2U_ = d2U(r, mu)
    Uxx = d2U_[0, 0]
    Uxy = d2U_[0, 1]
    Uyy = d2U_[1, 1]

    A = np.zeros((4, 4))
    A[0, :] = [0, 0, 1, 0] # df/dx
    A[1, :] = [0, 0, 0, 1] # df/dy
    A[2, :] = [Uxx, Uxy, 0, 2] # df/du
    A[3, :] = [Uxy, Uyy, -2, 0] # df/dv

    return A


def lin_lyapunov_orbit(XL: np.ndarray, om_ly: float, alpha0: float, V: np.ndarray, tpoints = 1000) -> np.ndarray:
    # This function computes the linearized Lyapunov orbit given the 
    # Lyapunov frequency om_ly, the initial guess for alpha alpha0
    # and the eigenvector V of the Jacobian A(xL) corresponding to the
    # complex eigenvalue with positive imaginary part
    # points is the number of points to evaluate the orbit

    # About dimensionality:
    # XL can be regarded as the state vector [xL, yL, 0, 0] or
    # only the positions [xL, yL]
    # Obviously, coming from A(xL) the eigenvector V is a 4x1 vector, 
    # so if we consider the state vector [x, y, u, v], we can take the
    # the whole eigenvector V, or only the first two components if the
    # state is simplified to [x, y]

    if len(XL) ==  len(V):
        dim = len(XL)
    elif len(XL) == 2 and len(V) == 4:
        # Fall back to the full state vector if XL is only the position
        XL = np.concatenate((XL, np.zeros(2))).reshape(4, 1)
        dim = 4
    elif len(XL) == 4 and len(V) == 2:
        # Raise an error if the state vector is full and the eigenvector is only the position
        raise ValueError("The eigenvector V must be a 4x1 vector")
    else:
        # Something went wrong
        raise ValueError("The dimensions of the state vector and the eigenvector do not match")

    # The linearized Lyapunov orbit is given by:
    # x = xL + alpha0(cos(om_ly*t)*V.real - sin(om_ly*t)*V.imag)
    # "integrated" over t = [0, 2*pi/om_ly], one period of the orbit
    time_span = np.linspace(0, 2*np.pi/om_ly, tpoints)
    # Allocate the state vector
    X = np.zeros((dim, tpoints)) 
    for i, t in enumerate(time_span):
        X[:, i:i+1] = XL + alpha0*(np.cos(om_ly*t)*V.real - np.sin(om_ly*t)*V.imag)
    return X




def shooting(Y_guess, mu):
    # This function "shoots" the initial guess Y0 to find the orbit with two crossing points
    # perpendicular to the x-axis
    # It integrates over tau, such as T = t*tau is the orbital period, and the final condition
    # is evaluated at tau = 1/2 

    # Design variables X = [x0, v0, T]
    # Constraint vector C = [y(T/2), u(T/2)] = 0

    # Initial time
    tau0 = 0
    # Final time
    tau1 = 1/2
    # Flatten Y_guess or numpy goes mad
    Y_guess = Y_guess.flatten()
    x0 = Y_guess[0]
    v0 = Y_guess[1]
    T = Y_guess[2]

    # Initial state vector
    Y0 = np.array([x0, 0, 0, v0])
    
    # Initial conditions
    Y0 = np.concatenate((Y0, np.eye(4).flatten())) # Expand with the STM
    sol = solve_ivp(f, [tau0, tau1], Y0, args=(T, mu), method='LSODA')

    # Unpack Yf = [x(T/2), y(T/2), u(T/2), v(T/2), PHI(T/2)]
    Yf = sol.y[:, -1]
    # F_X = [y(T/2), u(T/2)] - [0, 0](Yt)
    F_X = Yf[1:3].reshape(2,1) # - Yt, but Yt = 0
    PHIf = Yf[4:].reshape(4, 4)
    
    f0 = f(tau0, Y0, T, mu)[:4] # Initial state_dot vector
                                # stripped of the STM
    
    # DF = [PHI_yx, PHI_yv, 0.5*PHI_f(1,:)@f0]
    #      [PHI_ux, PHI_uv, 0.5*PHI_f(2,:)@f0]
    DF = np.zeros((2,3))
    
    DF[0,0] = PHIf[1, 0] # PHI_yx
    DF[1,0] = PHIf[2, 0] # PHI_ux
    DF[0,1] = PHIf[1, 3] # PHI_yv
    DF[1,1] = PHIf[2, 3] # PHI_uv
    DF[0,2] = 0.5*PHIf[1, :]@f0 
    DF[1,2] = 0.5*PHIf[2, :]@f0

    # Compute the correction
    dX = -spli.pinv(DF)@F_X
    
    return dX, DF, F_X    
   

def nonlin_lyapunov_orbit(XL: np.ndarray, mu: float, alpha0, tpoints = 1000) -> np.ndarray:
    # This function computes the nonlinear Lyapunov orbit given the
    # lagrangian point XL and the mass ratio mu
    # alpha0 is the amplitude of the orbit for the linearized case
    # tpoints is the number of points to evaluate the orbit
    
    # Returns the initial condition of the design vector orbit to be integrated X0, the Jacobi
    # constat C associated and DF and F_X from the shooting method


    # eps is used to avoid numerical ghost values interfering with some of the
    # computations. So if x < eps, x = 0 is considered
    eps = 10**-14

    # The nonlinear Lyapunov orbit is given by computing the linear lyapunov orbit and then
    # shooting at the crossing points to achieve a periodic orbit
    
    # Check XL dimensionality
    if len(XL) == 2:
        XL = np.concatenate((XL, np.zeros(2)))
    elif len(XL) == 4:
        pass 
    else:
        raise ValueError("The dimensionality of the Lagrangian point is wrong")


    # Compute the Jacobian A at the Lagrangian point
    A_ = A(XL[:2].flatten(), mu)
    # Compute the eigenvalues and eigenvectors
    eigvals, eigvecs = spli.eig(A_)
    # Find the complex eigenvalue with positive imaginary part
    
    Ec1_idx = [idx for idx in range(len(eigvals)) if isinstance(eigvals[idx], complex) 
               and eigvals[idx].imag > eps and abs(eigvals[idx].real) < eps]
    Ec1 = eigvecs[:, Ec1_idx]
    Lc1 = eigvals[Ec1_idx]
    # Compute the Lyapunov frequency
    om_ly = Lc1.imag

    # Compute the linearized Lyapunov orbit
    X_lyap = lin_lyapunov_orbit(XL, om_ly, alpha0, Ec1, tpoints)
    
    # Compute the crossing points on the x-axis and take the one at the right of the Lagrangian point
    idx_cross = [idx for idx in range(tpoints-1) if X_lyap[1, idx] * X_lyap[1,idx+1] < 0 and X_lyap[0, idx] - XL[0] > 0]
    X0 = X_lyap[:, idx_cross]

    # Build the initial guess for the shooting method and the final condition
    Y_guess = np.zeros((3, 1))
    Y_guess[0] = X0[0] # x0
    Y_guess[1] = X0[3] # v0
    Y_guess[2] = 2*np.pi/om_ly[0] # Initial guess for the period
    # Yt = [0,0] # [y(T/2), u(T/2)] = 0 is the constraint
    F_X = 1  # Initialize the constraint vector

    # Shooting loop
    while spli.norm(F_X) > eps:
        dX, DF, F_X = shooting(Y_guess, mu)
        Y_guess += dX
    # Return the design vector of the orbit to be integrated, DF and F_X
    print(f"Shooting converged at: \n {Y_guess} \n that satisfies the constraint with accuracy: \n {F_X}")

    return Y_guess, DF, F_X

   


def compute_orbit_Yd(Yd:np.ndarray, mu: float, n_points = 1000):
    # This function computes the full orbit from the design vector
    # Yd = [x0, v0, T]
    # mu is the mass ratio
    # Returns the state vector X and the state transition matrix PHI
    Yd = Yd.flatten()
    Y0 = [Yd[0], 0, 0, Yd[1]]
    Y0 = np.concatenate((Y0, np.eye(4).flatten()))
    T = Yd[2]

    tau_span = np.linspace(0, 1, n_points)
    sol = solve_ivp(f, [0, 1], Y0, args=(T, mu), method='LSODA', t_eval=tau_span)

    # Unpack solution and return
    X = sol.y[:4, :] # State
    PHI = sol.y[4:, :] # State transition matrix

    return X, PHI






