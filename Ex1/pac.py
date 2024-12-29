import numpy as np
import scipy.linalg as spli
from lyapunov import *
from consts import *


class LyapOrbit:
    """
    A class to represent a Lyapunov orbit around a Lagrangian point.
    Attributes
    ----------
    Yd : array-like
        Design vector of the orbit. 
        tau : array-like
        Tangent vector of the orbit.
        XL : array-like
        Lagrangian coordinate.
        mu : float
        Mass ratio.
        ds : float
        Step size for the PAC method.
        C : float
        Jacobi constant of the orbit.
        X : array-like
        State vector of the orbit.
        PHI : array-like
        State transition matrix of the orbit.
    Methods
    -------
    compute_orbit(Dtau=[0, 1], n_points=1000):
        Computes the orbit.
    plot(plt_args=[]):
        Plots the orbit.
    """

    def __init__(self, Yd, tau, XL, mu, ds = 1e-3):
        self.Yd = Yd
        self.tau = tau
        self.ds = ds
        self.mu = mu
        self.C = Jacobi(np.array([Yd[0][0], 0]), np.array([0, Yd[1][0]]), mu)

        self.X, self.PHI = None, None
    
    def propagate(self, Tf = 0, n_points = 1000 ):
        Yd = self.Yd.flatten()
        
        X0 = np.array([Yd[0], 0, 0, Yd[1]])
        T = Yd[2]
        # Add the state transition matrix
        X0 = np.concatenate((X0, np.eye(4).flatten()))
        
        if Tf == 0:
            Tf = T # T
        tau_span = np.linspace(0, Tf, n_points)
        mu = self.mu
    
        sol = solve_ivp(fdyn, [0, Tf], X0, args=(T, mu), method='LSODA', t_eval=tau_span, rtol = 3*10**-14, atol = 10**-14)

        # Unpack solution and return
        X = sol.y[:4, :] # State
        PHI = sol.y[4:, :].reshape(4,4, -1)
        
        return X, PHI
    
    def plot(self, plt_args = []):
        if self.X is None:
            self.X, self.PHI = self.propagate()
        import matplotlib.pyplot as plt
        plt.plot(self.X[0, :], self.X[1, :], plt_args)

    def save(self, file):
        np.savez(file, Yd = self.Yd, tau = self.tau, ds = self.ds, mu = self.mu)
    

def load_orbit(file):
    data = np.load(file)
    Yd = data['Yd']
    tau = data['tau']
    ds = data['ds']
    mu = data['mu']
    return LyapOrbit(Yd, tau, ds, mu)
    
 
# Write a function to save an orbit to an open file with the npz extension
# without overwriting the file



    

    
class Family:
    """
    A class to represent a family of Lyapunov orbits around a Lagrangian point.
    Attributes
    ----------
    XL : array-like
        Lagrangian coordinate.
    mu : float
        Mass ratio.
    family : list
        List to store the family of Lyapunov orbits.
    C_family : list
        List to store the Jacobi constants of the orbits in the family.
    Methods
    -------
    find_family(alpha0, n_orbits, verbose=False):
        Computes the Lyapunov family of orbits around the Lagrangian point.
    plot_family():
        Plots the family of Lyapunov orbits.
    plot_range(idx=list()):
        Plots a specified range of orbits from the family.
    """


    
    def __init__(self, XL, mu ):
        self.XL = XL
        self.mu = mu
        self.family = []
        self.C_family = []
        
    def find_family(self, alpha0, n_orbits, verbose = False):
        # This function computes the Lyapunov family of orbits around XL
        # Input:
        # XL: Lagrangian coordinate
        # alpha0: Initial guess for  the family
        # mu: Mass ratio

        # Start by taking the first non-linear lyapunov orbit as initial guess
        Yd0, DF, F_X = nonlin_lyapunov_orbit(self.XL, self.mu, alpha0)
        
        X0 = np.array([Yd0[0][0], 0, 0, Yd0[1][0]])
        T0 = Yd0[2]

        # Compute the Jacobi constant and the tangent vector
        C0 = Jacobi(np.array([Yd0[0][0], 0]), np.array([0, Yd0[1][0]]), self.mu)
        tau0 = spli.null_space(DF)
        
        orb0 = LyapOrbit(Yd0, tau0, self.XL, self.mu)
     
        # tau_family.append(tau0)
        # ds_family.append(1e-3)
        
        # Initialize ds
        ds = 1e-3
        i = 0 
        while i < n_orbits:
            # Try to find the next orbit
            try:
                # PAC the orbit
                Ydk, tauk, G_k = PAC(Yd0, tau0, ds, self.mu) 
                if verbose:
                    print(f"ds: {ds}, err={spli.norm(G_k)}")
                if spli.norm(G_k) < 1e-12:
                    # If the norm of the constraint is below tol, we have found a new orbit
                    
                    # Create the orbit object
                    orbk = LyapOrbit(Ydk, tauk, self.XL, self.mu, ds)
                    # Append to family
                    self.family.append(orbk)
                    self.C_family.append(orbk.C)
                   
                    # Update the initial guess for the next orbit
                    Yd0 = Ydk
                    tau0 = tauk
    
                    # Update ds
                    ds *= 1.2
                    if ds > 0.05:
                        ds = 0.05
                    # Increase the orbit counter
                    i += 1
                    if verbose:
                        print("Accepted: increasing ds")
                        print(f"N of orbits: {i}")
                else:
                    # We decrease ds to try to find a better solution
                    ds /= 1.1
                    if ds < 1e-9:
                        # Idk it works
                        ds = 0.05
                    if verbose:
                        print("Rejected: decreasing ds")
            except KeyboardInterrupt:
                # If keyboard interrupt, break the loop
                break
            except:
                # If an error occurs, try to decrease ds
                ds /= 1.1
                if verbose:
                    print("Error: decreasing ds")
        
        # Build the family of orbits
        # for i in range(n_orbits):
        # orb = Orbit(Yd0, tau0, ds)

    

    def plot_family(self):
        import matplotlib.pyplot as plt
        from lyapunov import compute_orbit_Yd
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        norm = Normalize(vmin=min(self.C_family), vmax=max(self.C_family))
        cmap = plt.get_cmap('jet')

        for orbit in self.family:
            X, PHI = orbit.propagate()
            C = orbit.C
            color = cmap(norm(C))
            plt.plot(X[0, :], X[1, :], color=color)
    
        plt.axis('equal')
        # # plot moon
        plt.plot([1-self.mu], [0], 'ro')
        # Plot Lagrange point
        plt.plot([self.XL[0]], [self.XL[1]], 'bo')
        # Add a colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='C value', ax = plt.gca())

        plt.show()

    # Plot any number of orbits around the lagrangian po
    # arg1: idx of the orbits in the family, default is all
    def plot_range(self, idx : list = []):
        import matplotlib.pyplot as plt
        from lyapunov import compute_orbit_Yd
        if len(idx) == 0:
            self.plot_family()
        else:
            for i in idx:
                orbit = self.family[i]
                orbit.plot()
            plt.axis('equal')
            plt.plot([1-self.mu], [0], 'ro')
            plt.plot([self.XL[0]], [self.XL[1]], 'bo')
            plt.show()
    



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


# TODO: Not working, something's broken
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


