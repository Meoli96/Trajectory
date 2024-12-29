from scipy.integrate import solve_ivp
import numpy as np
from scipy import linalg as splin
import matplotlib.pyplot as plt
from lyapunov import LyapOrbit, fdyn
from potential import U

de = 40/384400 # Perturbation parameter

def Poinc_hit(t, S, T, mu):
    # Assuming that the Poincare section is a plane, we can find the intersection
    # of the trajectory with the plane by solving for the intersection point.
    # We can then use the intersection point to find the intersection of the
    # trajectory with the Poincare section.

    # Assuming Poinc_sect  x = 1-mu
    return not(abs(S[0] - (1 - mu)) < 1e-3)
    

Poinc_hit.terminal = True

class Manifold:
    def __init__(self, orbit: LyapOrbit ):
        self.orbit = orbit
        self.sols_s = []
        self.sols_u = []

        # Compute potential of the orbit to be used to display
        # ZVCs
        mu = self.orbit.mu

        # Compute the zero velocity curve
        x = np.linspace(-2, 2, 1000)
        y = np.linspace(-2, 2, 1000)

        self.X, self.Y = np.meshgrid(x, y)

        self.U_grid = np.zeros_like(self.X)
        for i in range(len(x)):
            for j in range(len(y)):
                self.U_grid[i,j] = U(np.array([x[i], y[j]]), mu) 



    def compute_manifold(self, n_curves, verbose = False, k = 1):
         # Computes the stable and unstable manifolds of a given orbit

        # k: tau_f = k*T, where tau_f is the final time of the integration
        if k < 0 or k == 0:
            raise ValueError("k must be a positive real number")
        
        # Perturbation parameter
        de = 40/384400

        # Compute the periodic orbit
        X, PHI = self.orbit.propagate()
        T = self.orbit.Yd[2]
        mu = self.orbit.mu

        # Compute the length of the orbit
        length = len(X[0,:])
        
      
 

        p = X[:, -1]
        ### Compute the eigs (Es, Eu)[0] and eigenvectors (EVs, EVu) of the STM
        M = PHI[:,:,-1] # Monodromy matrix (PHI(T,0))
        eigvals, eigvecs = splin.eig(M)

        # Compute the stable and unstable eigenvectors
        # E is stable if Re(eigval) < 1 and unstable if Re(eigval) > 1
        # We also need to take into account that the eigenvalues are complex, 
        # we need only the real ones tho
        
        Es_idx = [i for i in range(len(eigvals)) if np.abs(eigvals[i]) < 1 and eigvals[i].imag == 0]
        Eu_idx = [i for i in range(len(eigvals)) if np.abs(eigvals[i]) > 1 and eigvals[i].imag == 0]

        Es0, Eu0 = eigvals[Es_idx], eigvals[Eu_idx]
        EVs0, EVu0 = eigvecs[:, Es_idx], eigvecs[:, Eu_idx]
        if verbose:
            print(f"Stable eigenvalues: {Es0}")
            print(f"Unstable eigenvalues: {Eu0}")

            print(f"Stable eigenvectors: {EVs0}")
            print(f"Unstable eigenvectors: {EVu0}")


        for i in range(n_curves):
            # We now need to choose some point on the periodic orbit. We also need
            # to take into account that M is the STM starting from the current choosen point
            # and then propagated for one period. Hence, we could choose the point and then
            # propagate it or employ the formula
            # Es(tau) = PHI(tau,0) * Es(0)
            # Eu(tau) = PHI(tau,0) * Eu(0)
            # for a point temporally distant tau from the initial point.
            if i == 0:
                # Initial case, we already have computed the eigenvectors needed
                EVs = EVs0
                EVu = EVu0
        
            else:
                # We need to recompute p, the associated eigenvectors and eigenvalues
                # idx moves the point on the periodic orbit by displacing it in time
                idx = i*length//n_curves
                p = X[:,idx]
                PHI_p = PHI[:,:,idx]

                EVs = PHI_p @ EVs0
                EVu = PHI_p @ EVu0

                if verbose:
                    print(f"Stable eigenvectors at point {i}: {EVs}")
                    print(f"Unstable eigenvectors at point {i}: {EVu}")
    
            # Only real part is needed, the imaginary part is due to casting to complex and 0
            # Stable 
            X_s_p = np.real(p + de*EVs[:,0]/np.linalg.norm(EVs[:2,0]))
            X_s_m = np.real(p - de*EVs[:,0]/np.linalg.norm(EVs[:2,0]))
            # Unstable
            X_u_p = np.real(p + de*EVu[:,0]/np.linalg.norm(EVu[:2,0]))
            X_u_m = np.real(p - de*EVu[:,0]/np.linalg.norm(EVu[:2,0]))
            
            ### Propagate the deviation vectors
            # Build the initial conditions
            PHI0 = np.eye(4)
            X0sp = np.concatenate((X_s_p, PHI0.flatten()))
            X0sm = np.concatenate((X_s_m, PHI0.flatten()))
            X0up = np.concatenate((X_u_p, PHI0.flatten()))
            X0um = np.concatenate((X_u_m, PHI0.flatten()))
            
            
            if verbose:
                print(f"Initial conditions for stable manifold: {X0sp}")
                print(f"Initial conditions for unstable manifold: {X0up}")


                
       
            # Stable (backward)
            sol_sp = solve_ivp(fdyn, [0, -k*T], X0sp, 
                                args = (T,mu), method='LSODA', 
                                rtol=3e-12, atol=1e-12)
            
            sol_sm = solve_ivp(fdyn, [0, -k*T], X0sm, 
                                args = (T,mu), method='LSODA',
                                rtol=3e-12, atol=1e-12)

            # Unstable (forward)
            sol_up = solve_ivp(fdyn, [0, k*T], X0up,
                                args = (T,mu), method='LSODA',
                                rtol=3e-12, atol=1e-12)
            sol_um = solve_ivp(fdyn, [0, k*T], X0um, 
                                args = (T,mu), method='LSODA',
                                rtol=3e-12, atol=1e-12)

            # Extract the solution
            Xsp = sol_sp.y[:4,:]
            Xsm = sol_sm.y[:4,:]
            Xup = sol_up.y[:4,:]
            Xum = sol_um.y[:4,:]

            # Stable and unstable tuple
            # return of the integration process



            # Append the solution to the list
            self.sols_s.append(Xsp)
            self.sols_s.append(Xsm)    
            self.sols_u.append(Xup)
            self.sols_u.append(Xum)

            print(f"Curve {i+1}/{n_curves} computed", end = '\r')
        # End of loop

        
        return self.sols_s, self.sols_u
    
    
    def surface_hit(self, surface_f, tol = 1e-3, direction = 'left'):
        # Surface hit detection
        # tol: tolerance for the surface hit
        # surface: function that defines the surface
        # The surface is defined as surface(X) = 0
        # The surface is defined in the form of a function
        # that takes a 4x1 vector and returns a scalar
        # If the surface is hit, the function returns 1, otherwise 0
        
        # cuts
        cut_s =  [None] * len(self.sols_s) 
        cut_u =  [None] * len(self.sols_s) 
        # Values of the state at surface hit
        sur_s = [None] * len(self.sols_s) 
        sur_u = [None] * len(self.sols_u)
        s_b = False # False if the surface is not hit, True otherwise
        u_b = False

        for i in range(len(self.sols_s)): # n_curves
            for j in range(len(self.sols_s[i][0,:])):
                # Stable
                if surface_f(self.sols_s[i][0:4,j], tol):
                    # We hit the surface
                    sur_s[i] = self.sols_s[i][:,j]
                    if direction == 'left':
                        cut_s[i] = self.sols_s[i][:,0:j]
                    elif direction == 'right':
                        cut_s[i] = self.sols_s[i][:,j:]
                    s_b = True 
                    break
            # Unstable
            for k in range(len(self.sols_u[i][0,:])):
                if surface_f(self.sols_u[i][0:4,k], tol):
                    # We hit the surface
                    sur_u[i] = self.sols_u[i][:,k]
                    if direction == 'left':
                        cut_u[i] = self.sols_u[i][:,0:k]
                    elif direction == 'right':
                        cut_u[i] = self.sols_u[i][:,k:]
                    u_b = True
                    break
            
            # If the surface is not hit, we keep the whole curve
            if not s_b:
                cut_s[i] = self.sols_s[i]
            if not u_b:
                cut_u[i] = self.sols_u[i]
            # Reset hit detection
            s_b = False
            u_b = False


        # Remove all the None values
        cut_s = [x for x in cut_s if x is not None]
        cut_u = [x for x in cut_u if x is not None]
        sur_s = [x for x in sur_s if x is not None]
        sur_u = [x for x in sur_u if x is not None]

        # Transform the list to numpy arrays

        sur_s = np.array(sur_s)
        sur_u = np.array(sur_u)
        

        return cut_s, sur_s, cut_u, sur_u
    
  
    
    
    def plot_ZVC(self, ax = None, **kwargs):
        # Plot the zero velocity curve
        if ax is None:
            ax = plt.gca()
        if kwargs == {}:
            kwargs = {'colors': 'black'}
        ax.contour(self.Y, self.X, self.U_grid, levels = [self.orbit.C/2], **kwargs)
        return ax
    
    def plot_stable(self, n_curves = 10, ax = None, **kwargs):
        # Plot the stable manifolds
        # n_Curves: number of curves to plot
        # ax: axis to plot the manifolds
        # kwargs: additional arguments to pass to the plot
        if self.sols_s == [] or self.sols_s == []:
            raise ValueError("Manifold not computed")
        if ax is None:
            ax = plt.gca()
        for i in range(n_curves):
            ax.plot(self.sols_s[i][0,:], self.sols_s[i][1,:], **kwargs)
        return ax





