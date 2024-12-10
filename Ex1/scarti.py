### NOT RELIABLE!!! ###
### THE NONLINEAR LYAPUNOV METHOD USED TO COMPUTE THE ORBIT HAS THE X COORDINATE AS A DESIGN VARIABLE ###
### SO THE BISECTION METHOD ON DX IS NOT RELIABLE, AS X IS NOT THEN ENFORCED BY THE LYAPUNOV METHOD ###

# Bisection method, find the orbit around L2 with the same Jacobi constant within Ctol = 1e-4
# Use as first guess the orbit around L1 with the same Jacobi constant
def bisect(C_target, L_family, C_family, XL, debug = False):
    # Find two orbits in L_family with C1 < C_target < C2 
    idx = 0
    print(f"Target C: {C_target}")
    # Compute A(XL) and its eigenvalues and eigenvectors, it is a strumentopolo
    # useful for later
    A_ = A(XL[:2].flatten(), mu)
    
    eigvals, eigvecs = spli.eig(A_)
    # Find the complex eigenvalue with positive imaginary part
    
    Ec1_idx = [idx for idx in range(len(eigvals)) if isinstance(eigvals[idx], complex) 
               and eigvals[idx].imag > eps and abs(eigvals[idx].real) < eps]
    Ec1 = eigvecs[:, Ec1_idx]
    Lc1 = eigvals[Ec1_idx]
    # Compute the Lyapunov frequency
    om_ly = Lc1.imag[0]

    for i in range(len(C_family)):
        if C_family[i] < C_target and C_family[i-1] > C_target:
            # idx is the lower than orbit, while idx-1 is the higher than orbit
            # Kinda spoiled by the fact that C is monotonic decreasing as we move along the family
            idx = i
            break
    
    C_a = C_family[idx-1]
    C_b = C_family[idx]
    print(f"C_a: {C_a}, C_b: {C_b}")

    x_a = L_family[idx-1][0][0]
    x_b = L_family[idx][0][0]
    # Distance from the Lagrange point
    x_a_L = abs(x_a - XL[0])
    x_b_L = abs(x_b - XL[0])

    # Initial guess
    alpha_guess = (x_a_L + x_b_L)/2/spli.norm(Ec1.real[0]) # So that the initial guess is on the real axis
                                                           # at (x_a + x_b)/2

    err = 1
    err_old = 1

    while err > Ctol:
        
        try:
            # Compute nonlin lyap orbit
            Yd, DF, F_X = nonlin_lyapunov_orbit(XL, mu, alpha_guess)
            # Compute Jacobi constant
            C_guess = Jacobi(np.array([Yd[0][0], 0]), np.array([0, Yd[1][0]]), mu)
        
            
            # Compute the full orbit and the distance from the Lagrange point
            X, PHI = compute_orbit_Yd(Yd, mu)
            x_c_L = abs(X[0,500] - XL[0])
            if C_guess < C_target: # C_target > C_guess, get closer to XL
                x_b_L = x_c_L
                C_b = C_guess
            else: # C_target < C_guess, get farther from XL
                x_a_L = x_c_L
                C_a = C_guess
            if debug:
                print(f"alpha: {alpha_guess}, C: {C_guess}, err: {abs(C_guess - C_target)}")
                print(f"x_a: {x_a_L}, x_b: {x_b_L}")
            print(C_guess)
        
            # Compute the error
            err = abs(C_guess - C_target)
            if err < Ctol:
                break
            
            if err > err_old:
                print(f"Not converging, err: {err}, err_old: {err_old}")
                break
            else:
                err_old = err
            
                # Compute the new alpha guess
                alpha_guess = (x_a_L + x_b_L)/2/spli.norm(Ec1.real[0])
  
        except KeyboardInterrupt:
            break
       
    
    return Yd, C_guess

                