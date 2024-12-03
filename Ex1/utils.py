import numpy as np
import scipy.linalg as spli
from potential import Jacobi
from lyapunov import shooting
from consts import eps


def cos_simil(vec1, vec2):
    # Compute the cosine of the two vectors 
    sol = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return sol

def closest_dir(vec_ref, vectors):
    # Find the closest vector in the list of vectors
    # to the reference vector
    cos_sim = -1
    for vec in vectors:
        cos_sim_new = cos_simil(vec_ref, vec)
        if cos_sim_new > cos_sim:
            cos_sim = cos_sim_new
            closest_vec = vec
    return closest_vec



def bisection(f, a, b):
    # Bisection algorithm to find the root of a function f
    # Input:
    # f: function to find the root of
    # a: left bound of the interval
    # b: right bound of the interval
    # Output:
    # x: root of the function
    # Note: The function assumes that the root is unique and that f(a)*f(b) < 0
    c = (a + b)/2
    while np.abs(f(c)) > eps:
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
        c = (a + b)/2
    return c
    

def bisection_Jacobi(Yd0, C_range, mu):
    # This function serves to find an orbit in the C_range
    # provided in the consts.py file
    # It should compute the orbits pairwise (L1 and L2) once
    # a satisfying solution is found.

    # The function should return the initial guess for the
    # PAC algorithm

    # Design variables Yd0 = [x0, v0, T]
    # Apply the bisection method to x0
    Yd0 = Yd0.flatten()
    x0 = Yd0[0]
    v0 = Yd0[1]
    T = Yd0[2]

    X = np.array([x0, 0, 0, v0])

    # Compute the Jacobi constant
    C = Jacobi(X[:2],X[2:], mu)
    print (f"Jacobi constant: {C}")
    print("Entering bisection loop...")

    # Bisection algorithm
    while C < C_range[0] or C > C_range[1]:
        # Bisection on x0
        if C < C_range[0]:
            x0 = x0*2
        else:
            x0 = x0/2
        # Shoot the orbit to find the lyapunov orbit
        Y_guess = np.array([x0, v0, T])
        F_X = np.ones((2,1))
        # Shooting loop
        print("Shooting...")
        Y_guess_old = Y_guess
        while np.linalg.norm(F_X) > 1e-12:

            # Shoot
            DX, DF, F_X = shooting(Y_guess, mu)
            # Update the guess
            Y_guess_old = Y_guess
            Y_guess = Y_guess + DX

            if spli.norm(Y_guess - Y_guess_old) < 1e-12:
                print("Shooting cant change the guess")
                break
        # Once we exit we recompute the Jacobi constant
        Y_guess = Y_guess.flatten()
        X = np.array([Y_guess[0], 0, 0, Y_guess[1]])
        C = Jacobi(X[:2],X[2:], mu)
        print(f"Jacobi constant: {C}")

    return Y_guess

    


