# In this file several functions are defined to compute the potential, its gradient and its Hessian.
# and the Jacobi constant. The Jacobi constant is defined as C = 2*U - (vx^2 + vy^2), where U is the potential

import numpy as np

# Potential function
def U(r: np.ndarray, mu) -> float:
    # r = [x, y]
    x, y = r

    r1 = ((x+mu)**2 + y**2)**0.5
    r2 = ((x-1+mu)**2 + y**2)**0.5

    return 0.5*(x**2 + y**2) + (1-mu)/r1 + mu/r2

# Here we use several functions to compute the partial derivatives, if only a single one is needed
# we can use the following functions:
def Ux(r: np.ndarray, mu) -> float:
    # r = [x, y]
    x, y = r

    r1 = ((x+mu)**2 + y**2)**0.5
    r2 = ((x-1+mu)**2 + y**2)**0.5

    return x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3

def Uy(r: np.ndarray, mu) -> float:
    # r = [x, y]
    x, y = r

    r1 = ((x+mu)**2 + y**2)**0.5
    r2 = ((x-1+mu)**2 + y**2)**0.5

    return y - (1-mu)*y/r1**3 - mu*y/r2**3

def Uxx(r: np.ndarray, mu) -> float:
    # r = [x, y]
    x, y = r

    r1 = ((x+mu)**2 + y**2)**0.5
    r2 = ((x-1+mu)**2 + y**2)**0.5

    return 1 - (1-mu)*(1/r1**3-3*(x+mu)**2/r1**5) - mu*(1/r2**3-3*(x-1+mu)**2/r2**5)

def Uyy(r: np.ndarray, mu) -> float:
    # r = [x, y]
    x, y = r

    r1 = ((x+mu)**2 + y**2)**0.5
    r2 = ((x-1+mu)**2 + y**2)**0.5

    return 1 - (1-mu)*(1/r1**3-3*y**2/r1**5) - mu*(1/r2**3-3*y**2/r2**5)

def Uxy(r: np.ndarray, mu) -> float:
    # r = [x, y]
    x, y = r

    r1 = ((x+mu)**2 + y**2)**0.5
    r2 = ((x-1+mu)**2 + y**2)**0.5

    return (1-mu)*(3*(x+mu)*y/r1**5) + mu*(3*(x-1+mu)*y/r2**5)

# If we need the gradient or the Hessian, we can use the following functions:

def dU(r: np.ndarray, mu) -> np.ndarray:
    # r = [x, y]
    x, y = r

    r1 = ((x+mu)**2 + y**2)**0.5
    r2 = ((x-1+mu)**2 + y**2)**0.5

    Ux = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    Uy =  y - (1-mu)*y/r1**3 - mu*y/r2**3

    return np.array([Ux, Uy])

def d2U(r: np.ndarray, mu) -> np.ndarray:
    # r = [x, y]
    x, y = r

    r1 = ((x+mu)**2 + y**2)**0.5
    r2 = ((x-1+mu)**2 + y**2)**0.5

    Uxx = 1 - (1-mu)*(1/r1**3-3*(x+mu)**2/r1**5) - mu*(1/r2**3-3*(x-1+mu)**2/r2**5)
    Uyy = 1 - (1-mu)*(1/r1**3-3*y**2/r1**5) - mu*(1/r2**3-3*y**2/r2**5)
    Uxy = (1-mu)*(3*(x+mu)*y/r1**5) + mu*(3*(x-1+mu)*y/r2**5)

    return np.array([[Uxx, Uxy], [Uxy, Uyy]])


def Jacobi(r: np.ndarray, v: np.ndarray,  mu) -> float:
    # v = [vx, vy]
    vx, vy = v
    # Compute potential at r
    U_ = U(r, mu)
    
    # return Jacobi constant C = 2*U - (vx^2 + vy^2)
    return 2*U_ - (vx**2 + vy**2)