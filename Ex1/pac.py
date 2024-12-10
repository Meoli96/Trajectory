import numpy as np
import scipy.linalg as spli
from lyapunov import *
from consts import *



def PAC(Yd0: np.ndarray, tau :np.ndarray, ds: float,  mu: float, max_iter = 1000
         , verbose = False):
    # Pseudo Arclength Continuation algorithm
    # From a design vector [x0, v0, T] coming from a nonlinear Lyapunov
    # orbit, the algorithm computes the design vector [xE, vE, T] corresponding
    # to another nonlinear orbit of the same family
    # Could be extended to allow n orbit computations

    # Input: 
    # Yd0: Initial design vector, solution of the nonlinear Lyapunov orbit (3x1)
    # Array of previous tangents, current is tau[-1]
    # mu: Mass ratio
    # Output:
    # Yd_k: Design vector of the new orbit (3x1)
    # DF: Jacobian of the new orbit (2x3)
    # F_X: Constraint vector of the new orbit (2x1)
    # Note: The exercise suggests to store the tangent of the curve tau
    #       and the step size delta_s to perform the bisection analysis

    # Step
    Yd_k = Yd0 + ds*tau

    # Allocate G and DG
    G = np.ones((3,1))
    Gnorm = np.linalg.norm(G)
    DG = np.zeros((3,3))
    it = 0
    stop_flag = False
    dY_old = np.full((3, 1), np.inf)

    while Gnorm > tol  and not stop_flag and it < max_iter:
        DX, DF, F_X = shooting(Yd_k, mu)
        # G_X = [F_X(2x1), tan_tau'*(Y_guess_new - Y_guess_old)-delta_s(1x1)]
        G[0:2] = F_X
        G[2] = tau.T@(Yd_k - Yd0) - ds
        Gnorm = np.linalg.norm(G)
 
        # Correct the design vector
        # DG = [DF, tau.T]
        DG[0:2, :] = DF
        DG[2, :] = tau.T
        # Compute the correction
        dY = -np.linalg.inv(DG)@G
        # Check if the solution is diverging
        
        if np.linalg.norm(dY) > np.linalg.norm(dY_old):
            if verbose:
                print(f'Diverging solution. err = {np.linalg.norm(G)}, tol = {tol}')
            stop_flag = True
            break
        else:
            dY_old = dY
        
        Yd_k = Yd_k + dY
        # Update the iteration
        it += 1
        
    # Solution found

    if verbose:
        if Gnorm < tol:
            print(f'Solution found in {it} iterations. err = {Gnorm}, tol = {tol}')
        else:
            print(f'Solution not found in {it} iterations. err = {Gnorm}, tol = {tol}')
            print(f'Last error: {G}')
    # Update the tangent
    T = spli.null_space(DF)
    # tau should be a 3x1, if its a 3xN, we need to take the tau closest in direction to the previous tau.
    # This is done to prevent other families to taint the solution. 
    if T.shape[1] > 1:
        tau_k = T[:, np.max(np.abs(tau.T@T))]
    else:
        tau_k = T
    # We need to make sure to compute the tangent in the same direction as the previous one, so
    tau_k = np.sign(tau_k.T@tau)*tau_k

    return Yd_k, tau_k, G

from lyap_obj import LyapOrbit

def PACo(orbit0: LyapOrbit, ds0 = 0,  max_iter = 1000, verbose = False):
    # Pseudo Arclength Continuation algorithm
    # From a design vector [x0, v0, T] coming from a nonlinear Lyapunov
    # orbit, the algorithm computes the design vector [xE, vE, T] corresponding
    # to another nonlinear orbit of the same family
    # Could be extended to allow n orbit computations

    # Input: 
    # Yd0: Initial design vector, solution of the nonlinear Lyapunov orbit (3x1)
    # Array of previous tangents, current is tau[-1]
    # mu: Mass ratio
    # Output:
    # Yd_k: Design vector of the new orbit (3x1)
    # DF: Jacobian of the new orbit (2x3)
    # F_X: Constraint vector of the new orbit (2x1)
    # Note: The exercise suggests to store the tangent of the curve tau
    #       and the step size delta_s to perform the bisection analysis

    if ds0 == 0:
        ds = orbit0.ds
    else:
        ds = ds0

    Yd0 = orbit0.Yd
    tau = orbit0.tau
    mu = orbit0.mu

    # Step
    Yd_k = Yd0 + ds*tau

    # Allocate G and DG
    G = np.ones((3,1))
    Gnorm = np.linalg.norm(G)
    DG = np.zeros((3,3))
    it = 0
    stop_flag = False
    dY_old = np.full((3, 1), np.inf)

    while Gnorm > tol  and not stop_flag and it < max_iter:
        DX, DF, F_X = shooting(Yd_k, mu)
        # G_X = [F_X(2x1), tan_tau'*(Y_guess_new - Y_guess_old)-delta_s(1x1)]
        G[0:2] = F_X
        G[2] = tau.T@(Yd_k - Yd0) - ds
        Gnorm = np.linalg.norm(G)
 
        # Correct the design vector
        # DG = [DF, tau.T]
        DG[0:2, :] = DF
        DG[2, :] = tau.T
        # Compute the correction
        dY = -np.linalg.inv(DG)@G
        # Check if the solution is diverging
        
        if np.linalg.norm(dY) > np.linalg.norm(dY_old):
            if verbose:
                print(f'Diverging solution. err = {np.linalg.norm(G)}, tol = {tol}')
            stop_flag = True
            break
        else:
            dY_old = dY
        
        Yd_k = Yd_k + dY
        # Update the iteration
        it += 1
        
    # Solution found, build tau_k and return the new orbit

    if verbose:
        if Gnorm < tol:
            print(f'Solution found in {it} iterations. err = {Gnorm}, tol = {tol}')
        else:
            print(f'Solution not found in {it} iterations. err = {Gnorm}, tol = {tol}')
            print(f'Last error: {G}')
    # Update the tangent
    T = spli.null_space(DF)
    # tau should be a 3x1, if its a 3xN, we need to take the tau closest in direction to the previous tau.
    # This is done to prevent other families to taint the solution. 
    if T.shape[1] > 1:
        tau_k = T[:, np.max(np.abs(tau.T@T))]
    else:
        tau_k = T
    # We need to make sure to compute the tangent in the same direction as the previous one, so
    tau_k = np.sign(tau_k.T@tau)*tau_k

    return LyapOrbit(Yd_k, tau_k, mu, ds), Gnorm


