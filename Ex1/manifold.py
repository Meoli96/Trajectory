from scipy.integrate import solve_ivp
import numpy as np
from scipy import linalg as splin

from lyapunov import LyapOrbit, fdyn, Poinc_hit

de = 40/384400 # Perturbation parameter

class Manifold:
    def __init__(self, orbit: LyapOrbit ):
        self.orbit = orbit
        self.sols_s = []
        self.sols_u = []

    def compute_manifold(self, n_curves, verbose = False, k = 1, terminal = True):
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


            # Stable (backwards)
            if terminal:
                # Terminal event set

                # Stable (backward)
                sol_sp = solve_ivp(fdyn, [0, -k*T], X0sp, 
                                    args = (T,mu), method='LSODA', 
                                    events=Poinc_hit, 
                                    rtol=3e-12, atol=1e-12)
                
                sol_sm = solve_ivp(fdyn, [0, -k*T], X0sm, 
                                    args = (T,mu), method='LSODA',
                                    events=Poinc_hit, 
                                    rtol=3e-12, atol=1e-12)
                # Unstable (forward)
                sol_up = solve_ivp(fdyn, [0, k*T], X0up,
                                    args = (T,mu), method='LSODA',
                                    events=Poinc_hit, 
                                    rtol=3e-12, atol=1e-12)
                sol_um = solve_ivp(fdyn, [0, k*T], X0um, 
                                    args = (T,mu), method='LSODA',
                                    events=Poinc_hit, 
                                    rtol=3e-12, atol=1e-12)
                
            else:
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
            ret = (Xsp, Xsm), (Xup, Xum)


            # Append the solution to the list
            self.sols_s.append(ret[0])
            self.sols_u.append(ret[1])

            print(f"\rCurve {i+1}/{n_curves} computed")
        # End of loop

        
        return self.sols_s, self.sols_u
    
    # Surface hit detection
    def surface_hit(self, surface, tol = 1e-3):
        # Surface hit detection
        # tol: tolerance for the surface hit
        # surface: function that defines the surface
        # The surface is defined as surface(X) = 0
        # The surface is defined in the form of a function
        # that takes a 4x1 vector and returns a scalar
        # If the surface is hit, the function returns the 
        # index of the hit, otherwise it returns -1
        idx_s = -1
        idx_u = -1
        for i in range(len(self.sols_s)):
            # Stable manifold
            for j in range(len(self.sols_s[i][0][0,:])):
                if np.abs(surface(self.sols_s[i][0][:,j])) < tol:
                    idx_s = i
                    break
            # Unstable manifold
            for j in range(len(self.sols_u[i][0][0,:])):
                if np.abs(surface(self.sols_u[i][0][:,j])) < tol:
                    idx_u = i
                    break
        return idx_s, idx_u
        

def compute_manifold(orbit: LyapOrbit, n_curves = 10, verbose = False, k = 1, 
                     terminal = True):

    # Computes the stable and unstable manifolds of a given orbit

    # k: tau_f = k*T, where tau_f is the final time of the integration
    if k < 0 or k == 0:
        raise ValueError("k must be a positive real number")
    
    # Perturbation parameter
    de = 40/384400

    # Compute the periodic orbit
    X, PHI = orbit.propagate()

    # Compute the length of the orbit
    length = len(X[0,:])
    
    T = orbit.Yd[2]
    mu = orbit.mu
    sols_s = []
    sols_u = []

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


        # Stable (backwards)
        if terminal:
            # Terminal event set

            # Stable (backward)
            sol_sp = solve_ivp(fdyn, [0, -k*T], X0sp, 
                                args = (T,mu), method='LSODA', 
                                events=Poinc_hit, 
                                rtol=3e-12, atol=1e-12)
            
            sol_sm = solve_ivp(fdyn, [0, -k*T], X0sm, 
                                args = (T,mu), method='LSODA',
                                events=Poinc_hit, 
                                rtol=3e-12, atol=1e-12)
            # Unstable (forward)
            sol_up = solve_ivp(fdyn, [0, k*T], X0up,
                                args = (T,mu), method='LSODA',
                                events=Poinc_hit, 
                                rtol=3e-12, atol=1e-12)
            sol_um = solve_ivp(fdyn, [0, k*T], X0um, 
                                args = (T,mu), method='LSODA',
                                events=Poinc_hit, 
                                rtol=3e-12, atol=1e-12)
            
        else:
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
        ret = (Xsp, Xsm), (Xup, Xum)


        # Append the solution to the list
        sols_s.append(ret[0])
        sols_u.append(ret[1])

        print(f"\rCurve {i+1}/{n_curves} computed")
    # End of loop

    
    return sols_s, sols_u