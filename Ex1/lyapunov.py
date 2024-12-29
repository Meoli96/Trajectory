import numpy as np

import scipy.linalg as spli
from scipy.integrate import solve_ivp
# This files contains formulae related to the problem at hand.
# In particular:
# - Potential U, gradient dU, and Hessian d2U
from potential import U, dU, d2U, Jacobi
# - Constants and parameters
from consts import *




"""
This module contains functions and classes to compute and analyze Lyapunov orbits around Lagrangian points in the context of the Circular Restricted Three-Body Problem (CRTBP).
Functions:
- fdyn(t: float, Y: np.ndarray, T: float, mu: float) -> np.ndarray:
    Computes the state derivative and state transition matrix derivative for the given state vector and parameters.
- A(r: np.ndarray, mu: float) -> np.ndarray:
    Computes the Jacobian matrix of the system at a given position and mass ratio.
- lin_lyapunov_orbit(XL: np.ndarray, om_ly: float, alpha0: float, V: np.ndarray, tpoints=1000) -> np.ndarray:
    Computes the linearized Lyapunov orbit given the Lyapunov frequency, initial guess for alpha, and the eigenvector of the Jacobian.
- nonlin_lyapunov_orbit(XL: np.ndarray, mu: float, alpha, tpoints=1000) -> np.ndarray:
    Computes the nonlinear Lyapunov orbit given the Lagrangian point and mass ratio.
- shooting(Y_guess, mu):
    Performs the shooting method to find the orbit with two crossing points perpendicular to the x-axis.
- shooting_loop(X0: np.ndarray, T0, mu, num_iter=100, verbose=False):
    Envelopes the shooting method to find a Lyapunov orbit that satisfies the constraint y(T/2) = u(T/2) = 0.
- compute_orbit_Yd(Yd: np.ndarray, mu: float, n_points=1000):
    Computes the full orbit from the design vector.
Classes:
--------
- LyapOrbit:
    Represents a Lyapunov orbit around a Lagrangian point.
    Attributes:
    - Yd : array-like
    - tau : array-like
    - XL : array-like
    - mu : float
    - ds : float
    - C : float
    - X : array-like
    - PHI : array-like
    Methods:
    - compute_orbit(Dtau=[0, 1], n_points=1000):
    - plot(plt_args=[]):
- Family:
    Represents a family of Lyapunov orbits around a Lagrangian point.
    Attributes:
    - XL : array-like
    - mu : float
    - family : list
    - C_family : list
    Methods:
    - find_family(alpha0, n_orbits, verbose=False):
    - plot_family():
    - plot_range(idx=list()):


"""



def fdyn(t: float, Y: np.ndarray, T:float, mu: float) -> np.ndarray:
    """
        Computes the state derivatives and state transition matrix for the given state vector.

        Parameters:
        - t (float): Time variable.
        - Y (np.ndarray): State vector containing [x, y, u, v] and the state transition matrix PHI.
        - T (float): Period of the orbit.
        - mu (float): Mass ratio.

        Returns:
        - np.ndarray: Concatenated array of state derivatives and flattened state transition matrix.
    """
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
    """
        Computes the Jacobian matrix of the system at a given position.

        Parameters:
        - r (np.ndarray): Position vector [x, y].
        - mu (float): Mass ratio.
        
        df/dY = A =     0   0   1  0
                        0   0   0  1
                        Uxx Uxy 0  2
                        Uyx Uyy -2 0
        Returns:
        - np.ndarray: Jacobian matrix of the system.
    """


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
    """
        Computes the linearized Lyapunov orbit.
        Parameters:
        - XL (np.ndarray): The state vector [xL, yL, 0, 0] or only the positions [xL, yL].
        - om_ly (float): The Lyapunov frequency.
        - alpha0 (float): The initial guess for alpha.
        - V (np.ndarray): The eigenvector of the Jacobian A(xL) corresponding to the complex eigenvalue with positive imaginary part.
        - tpoints (int, optional): The number of points to evaluate the orbit. Default is 1000.
        Returns:
        - np.ndarray: The linearized Lyapunov orbit evaluated over one period.
        Raises:
        - ValueError: If the dimensions of the state vector and the eigenvector do not match, or if the eigenvector is not a 4x1 vector when the state vector is full.
        Notes:
        - If XL is only the position [xL, yL] and V is a 4x1 vector, XL is extended to the full state vector [xL, yL, 0, 0].
        
        The linearized Lyapunov orbit is computed as:
        x = xL + alpha0(cos(om_ly*t)*V.real - sin(om_ly*t)*V.imag)
        integrated over t = [0, 2*pi/om_ly], one period of the orbit.
    """
    


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





def nonlin_lyapunov_orbit(XL: np.ndarray, mu: float, alpha, tpoints = 1000) -> np.ndarray:
    """
    Computes the nonlinear Lyapunov orbit given the Lagrangian point XL and the mass ratio mu.
    Parameters:
    XL (np.ndarray): The Lagrangian point coordinates. Should be a 2D or 4D vector.
    mu (float): The mass ratio.
    alpha: The amplitude of the orbit for the linearized case.
    tpoints (int, optional): The number of points to evaluate the orbit. Default is 1000.
    Returns:
    np.ndarray: The initial condition of the design vector orbit to be integrated (Y_guess),
                the Jacobian matrix (DF), and the function value at the shooting points (F_X).
    Raises:
    ValueError: If the dimensionality of the Lagrangian point XL is not 2 or 4.
    Notes:
    - This function first computes the linear Lyapunov orbit and then uses a shooting method
      to achieve a periodic orbit.
    - The function avoids numerical ghost values by setting values smaller than eps to zero.
    """
    


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
    om_ly = Lc1.imag[0]

    # Compute the linearized Lyapunov orbit
    X_lyap = lin_lyapunov_orbit(XL, om_ly, alpha, Ec1, tpoints)
    
    # Compute the crossing points on the x-axis and take the one at the right of the Lagrangian point
    idx_cross = [idx for idx in range(tpoints-1) if X_lyap[1, idx] * X_lyap[1,idx+1] < 0 and X_lyap[0, idx] - XL[0] > 0]
    X0 = X_lyap[:, idx_cross]
    
    Y_guess, DF, F_X = shooting_loop(X0, 2*np.pi/om_ly, mu)
   

    return Y_guess, DF, F_X

   


def shooting(Y_guess, mu):
    """
    Perform a shooting method to find the orbit with two crossing points perpendicular to the x-axis.
    Design variables X = [x0, v0, T]
    Constraint vector C = [y(T/2), u(T/2)] = 0
    Parameters:
    Y_guess (numpy.ndarray): Initial guess for the design variables [x0, v0, T].
    mu (float): Gravitational parameter.
    Returns:
    tuple:
        dX (numpy.ndarray): Correction vector for the design variables.
        DF (numpy.ndarray): Jacobian matrix of the constraint vector with respect to the design variables.
        F_X (numpy.ndarray): Constraint vector evaluated at the final time.

    
    """


    # It integrates over tau, such as T = t*tau is the orbital period, and the final condition
    # is evaluated at tau = 1/2 

    # 
    # 

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
    X0 = np.array([x0, 0, 0, v0])
    
    # Initial conditions
    X0 = np.concatenate((X0, np.eye(4).flatten())) # Expand with the STM
    sol = solve_ivp(fdyn, [tau0, tau1], X0, args=(T, mu), method='LSODA', rtol = 3*10**-14, atol = 10**-14)

    # Unpack Yf = [x(T/2), y(T/2), u(T/2), v(T/2), PHI(T/2)]
    Xf = sol.y[:, -1]
    # F_X = [y(T/2), u(T/2)] - [0, 0](Yt)
    F_X = Xf[1:3].reshape(2,1) # - Yt, but Yt = 0
    PHIf = Xf[4:].reshape(4, 4)
    
    f0 = fdyn(tau0, X0, T, mu)[:4] # Initial state_dot vector
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
   
def shooting_loop(X0: np.ndarray, T0, mu, num_iter = 100, verbose = False):
    
    """
    Envelopes the shooting method to find a Lyapunov orbit that satisfies the constraint y(T/2) = u(T/2) = 0.
    Parameters:
    X0 (np.ndarray): Initial state [x0, y0, u0, v0] used for the initial guess design vector Yd = [x0, v0, T].
    T0 (float): Initial guess for the period.
    mu (float): Mass ratio.
    num_iter (int, optional): Maximum number of iterations. Default is 100.
    verbose (bool, optional): If True, prints detailed information during the iterations. Default is False.
    Returns:
    tuple: 
        - Y_guess (np.ndarray): Design vector that satisfies the constraint.
        - DF (np.ndarray): Jacobian of the constraint.
        - F_X (np.ndarray): Constraint vector.
    """
    

    # Output:
    # Y_guess: Design vector that satisfies the constraint
    # DF: Jacobian of the constraint
    # F_X: Constraint vector

    X0 = X0.flatten()
    # Initial guess design vector
    Y_guess = np.array([X0[0], X0[3], T0]).reshape(3,1)
    F_X = 1 # Initialize the constraint vector
    # Stop conditions auxiliaries
    it = 0
    DX_old = np.full((3,1), 99)
    stop_flag = False
    # Shooting loop
    # (Ideally I need more than one stop condition, ie. a maximum number of iterations or unchanged guess)
    while spli.norm(F_X) > tol and it < num_iter and not stop_flag:
     
        dX, DF, F_X = shooting(Y_guess, mu)
        if spli.norm(dX) > spli.norm(DX_old):
            if verbose:
                print(f"Shooting loop diverged at iteration: {it}")
            stop_flag = True
            break
        Y_guess += dX
      
        it += 1
    # Check if the loop reached the maximum number of iterations
    if it == num_iter:
        if verbose:
            print(f"Shooting loop reached maximum number of iterations: {it}")
    if verbose:
        print(f"Shooting converged at: \n {Y_guess} \n with accuracy: \n {F_X}")
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
    sol = solve_ivp(fdyn, [0, 1], Y0, args=(T, mu), method='LSODA', t_eval=tau_span, rtol = 3*10**-14, atol = 10**-14)

    # Unpack solution and return
    X = sol.y[:4, :] # State
    PHI = sol.y[4:, :] # State transition matrix

    return X, PHI




from pac import PAC,LyapOrbit, Family


    
def find_Crange(fam: Family,  n_orbits, Crange = [2.95, 3.05], Ctol = 1e-4, verbose = False):
        # Finds the 
        # Take the orbit wihtin the Jacobi constant range [C0, C1]
        if len(fam.family) == 0:
            raise ValueError("The family is empty")
        
        mu = fam.mu
        orbits = [orbit for orbit in fam.family if orbit.C > Crange[0] and orbit.C < Crange[1]]
        if len(orbits) == 0:
            raise ValueError("No orbits in the range")
        
        if len(orbits) > n_orbits or len(orbits) == n_orbits:
            # If there are enough orbits in the range, return them
            ret_list = orbits
        
        # If there are no orbits in the range, find the closest ones
   
        else:
            # len(orbits) < n_orbits

            # Append already found orbits
            print(f"Found {len(orbits)} orbits in the range, PACing to find more")
            ret_list = orbits
            if len(orbits) < n_orbits:
                # PAC to find more orbits
                ref_orbit = orbits[-1]
                for i in range(n_orbits - len(orbits)):
                    
                                        
                    G_guess = 1
                    C_guess = 0
                    ds = 1e-2
                      # Take the last orbit as reference
                    Yd0 = ref_orbit.Yd
                    tau0 = ref_orbit.tau
                   
                    while spli.norm(G_guess) > 1e-12 or not(C_guess > Crange[0] and C_guess < Crange[1]):
                        Ydk, tauk, G_guess = PAC(Yd0, tau0, ds, mu)
                        C_guess = Jacobi(np.array([Ydk[0][0], 0]), np.array([0, Ydk[1][0]]), mu)
                        
                        if  spli.norm(G_guess) < 1e-12:
                            # Tolerance reached
                            if C_guess < C_range[0] or C_guess > C_range[1]:
                                # Ctol not reached 
                                if verbose:
                                    print(f"Orbit {i+1} not in the range, C = {C_guess}")
                                    print(f"ds: {ds}, C_guess: {C_guess}, G_guess: {spli.norm(G_guess)}")
                                
                                Yd0 = Ydk
                                tau0 = tauk
                                ds /= 1.1
                                    
                            else:
                                # Tolerance on C reached
                                break
                        else:
                            # Tolerance not reached, update ds
                            ds /= 1.1
                            if verbose:
                                print(f"Tolerance not met. ds: {ds}, G_guess: {spli.norm(G_guess)}")
                           
               

                    new_orbit = LyapOrbit(Ydk, tauk, fam.XL, fam.mu, ds)    
                    ret_list.append(new_orbit)
                    ref_orbit = new_orbit
                    if verbose:
                        print(f"Found orbit {i+1}")
                        print(f"ds: {ds}, C_guess: {C_guess}, G_guess: {spli.norm(G_guess)}")
            
        return ret_list    

                
def find_isoC(orbit_ref: LyapOrbit, family_targ: Family, Ctol = 1e-4, verbose = False):
    # This function finds the orbits in the target family that have the same Jacobi constant as the reference orbit
    # Input:
    # orbit_ref: Reference orbit
    # family_targ: Target family
    # Ctol: Tolerance for the Jacobi constant
    # Output:
    # orbits_isoC: List of orbits in the target family with the same Jacobi constant as the reference orbit
    if len(family_targ.family) == 0:
        raise ValueError("The target family is empty")
    
    C_target = orbit_ref.C
    if verbose:
        print(f"Finding orbits with C = {C_target}")
    ret_orbit = None

    for orbit in family_targ.family:
        # Maybe we are lucky
        if abs(orbit.C - C_target) < Ctol:
            ret_orbit = orbit
            if verbose:
                print(f"Found orbit with C = {C_target}")
            break
            

    if ret_orbit is None:
        # We're not lucky, PAC our way into it

        # Find the closest orbits from the target family to C_guess
        for i in range(len(family_targ.family)):
            C_k = family_targ.family[i].C
            C_kp = family_targ.family[i-1].C
            ## We're assuming the family is sorted by Jacobi constant
            if C_k < C_target and C_kp > C_target:
                l_idx = i-1
                r_idx = i
                break

        l_orb = family_targ.family[l_idx]

        G_guess = 1
        C_guess = 0 
        ds = l_orb.ds

        Yd0 = l_orb.Yd
        tau0 = l_orb.tau


        while spli.norm(G_guess) > 1e-12  or abs(C_guess - C_target) > Ctol:
            Ydk, tauk, G_guess = PAC(Yd0, tau0, ds, family_targ.mu)
            C_guess = Jacobi(np.array([Ydk[0][0], 0]), np.array([0, Ydk[1][0]]), family_targ.mu)
            if spli.norm(G_guess) < 1e-12:
                if abs(C_guess - C_target) < Ctol:
                    if verbose:
                        print(f"Found orbit with C = {C_target}, err = {abs(C_guess - C_target)}")
                    break
                else:
                    if verbose:
                        print(f"Orbit not in the C tolerance. C = {C_guess}, err = {abs(C_guess - C_target)}, ds = {ds}, G = {spli.norm(G_guess)}")
                   
                    # The strategy here is to move the ds in the direction of the target C,
                    # having in mind that as long as we're around the LP, the Jacobi constant 
                    # increases as the distance to the LP decreases

                    # So, if we're below the target C, we should move in the direction of the LP, 
                    # and thus a negative ds and vice versa

                    if C_guess < C_target:
                        if np.sign(ds) > 0:
                            ds = -ds
                        ds /= 1.1
                    elif C_guess > C_target:
                        if np.sign(ds) < 0:
                            ds = -ds
                        ds /= 1.1
                    Yd0 = Ydk
                    tau0 = tauk
            else:
                if verbose:
                    print(f"Orbit not in the G tolerance. G = {spli.norm(G_guess)}")
                ds /= 1.1
        new_orbit = LyapOrbit(Ydk, tauk, family_targ.XL, family_targ.mu, ds)
        return new_orbit
                
            

            



        

def find_isoC_bisection(orbit_ref: LyapOrbit, family_targ: Family, Ctol = 1e-4, verbose = False ):
    # This function finds the orbits in the target family that have the same Jacobi constant as the reference orbit
    # Input:
    # orbit_ref: Reference orbit
    # family_targ: Target family
    # Ctol: Tolerance for the Jacobi constant
    # Output:
    # orbits_isoC: List of orbits in the target family with the same Jacobi constant as the reference orbit
    if len(family_targ.family) == 0:
        raise ValueError("The target family is empty")
    
    C_target = orbit_ref.C
    if verbose:
        print(f"Finding orbits with C = {C_target}")
    ret_orbit = None

    for orbit in family_targ.family:
        # Maybe we are lucky
        if abs(orbit.C - C_target) < Ctol:
            ret_orbit = orbit
            if verbose:
                print(f"Found orbit with C = {C_target}")
            break
            

    if ret_orbit is None:
        # We're not lucky, PAC our way into it

        # Find the closest orbits from the target family to C_guess
        for i in range(len(family_targ.family)):
            C_k = family_targ.family[i].C
            C_kp = family_targ.family[i-1].C
            ## We're assuming the family is sorted by Jacobi constant
            if C_k < C_target and C_kp > C_target:
                l_idx = i-1
                r_idx = i
                break

        l_orb = family_targ.family[l_idx]
        r_orb = family_targ.family[r_idx]

        # Kickstart the bisection
        G_guess = 1
        C_guess = 0

        ds_a = 0
        ds_b = r_orb.ds
        ds_guess = ds_b/2

        Yd0 = l_orb.Yd
        tau0 = l_orb.tau

        while spli.norm(G_guess) > 1e-12  or abs(C_guess - C_target) > Ctol:
            # PAC
            Ydk, tauk, G_guess = PAC(Yd0, tau0, ds_guess, family_targ.mu)
            # Compute Jacobi constant
            C_guess = Jacobi(np.array([Ydk[0][0], 0]), np.array([0, Ydk[1][0]]), family_targ.mu)
            if spli.norm(G_guess) < 1e-12:
                if abs(C_guess - C_target) < Ctol:
                    if verbose:
                        print(f"Found orbit with C = {C_target}, err = {abs(C_guess - C_target)}")
                    break
                else:
                    if verbose:
                        print(f"Orbit not in the C tolerance. C = {C_guess}, err = {abs(C_guess - C_target)}, ds = {ds_guess}, G = {spli.norm(G_guess)}")
                    if C_guess < C_target:
                        ds_b = ds_guess
                    
                    elif C_guess > C_target:
                        ds_a = ds_guess    
                 
                    # Update the guess
                    ds_guess = (ds_a + ds_b)/2
                    if verbose:
                        print(f"New ds_guess = {ds_guess}")
            else:
                if verbose:
                    print(f"Orbit not in the G tolerance. G = {spli.norm(G_guess)}")
                ds_guess /= 1.1

        

        new_orbit = LyapOrbit(Ydk, tauk, family_targ.mu, ds_guess)
        return new_orbit	
    


