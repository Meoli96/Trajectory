import numpy as np

# Earth mass (kg)
mE = 5.972e24
# Moon mass (kg)
mM = 7.342e22

# Gravitational constant (m^3/kg/s^2)
G = 6.67430e-11

# Earth-Moon distance (km)
dEM = 384400

# "Relative" gravitational parameter
mu = mM/(mE+mM)



# L1
xL1 = 0.8369
yL1 = 0 
XL1 = [xL1, yL1]

# L2
xL2 = 1.1556
yL2 = 0
XL2 = [xL2, yL2]

alpha0 = 50/384400 # initial guess for alpha

# This Jacobi range tells me conditions for the initial position (dictated by alpha0) via the potential function
# and the initial velocity (v0, u0 is always 0 - perpendicular to the x-axis) for the shooting method
# C = 2U(r) - (vx^2 + vy^2)
C_range = [3.1370, 3.1493] # Jacobi constant range for L1 and L2, at least two/three pairs of orbits
                           # within this range

# Take the linearized Lyapunov orbit as the initial guess
# Shooting method to find orbit with two crossing points perpendicular to the x-axis
# u0 = 0, we are looking for an orbit perpendicular to the x-axis
# -- Initial guess for the state vector comes from the linearized Lyapunov orbit --
# We then need to find the eigenvectors of A(xL)
# Pseudo code:
# 1. Compute the Jacobian A at xL
# 2. Compute the eigenvectors of A
# 3. Take the eigenvector corresponding to the complex eigenvalue(with positive imaginary part)
# 4. Normalize the eigenvector?
# Apply the linearized Lyapunov orbit as the initial guess
# Shooting method to find orbit with two crossing points perpendicular to the x-axis
# Once one of the orbit is found, apply PAC to find the other ones of the same family
# Plot 
# Enjoy!


