import numpy as np
import matplotlib.pyplot as plt

################ Parameters ################
Nrho = 8                     # number of phenotypes
drho = 1.0 / Nrho            # discrete phenotypic spacing (simple Riemann sum)

epsilon2 = 5.0               # controls interface width (tumor diffuse edge)
M = 1.0                      # motility
# D_rho: diffusion coefficient per phenotype bin (array length Nrho)
# we can set different values per phenotype to model more/less aggressive cells
D_rho = np.linspace(0.05, 0.5, Nrho)   

# time
dt = 0.05
tmax = 30.0

############# Initial conditions ###########
# Spatial grid
Lx=100
Ly=100
# Field initialized to all zeros (only medium)
phi = np.zeros((Lx,Ly))
# Phenotype bins (u): small random initial densities for each phenotype. u is 3D: (x, y, phenotype)
u = 0.2 * np.random.random((Lx, Ly, Nrho))   

# Addition from students: initial tumor core (otherwise simulation rapidly grows to random noise)
cx, cy = Lx//2, Ly//2
radius = 12.0      
edge_width = 15.0   # How gradual the edge is (larger -> smoother)
max_center_value = 0.95  # peak phi in center (keeps it from being exactly 1 everywhere)

for i in range(Lx):
    for j in range(Ly):
        r = np.sqrt((i-cx)**2 + (j-cy)**2)
        # a smooth plateau near the center, decaying to 0 over 'edge_width'
        phi[i, j] = max_center_value * np.exp(-((r)**2) / (2*(edge_width**2)))
        if r <= radius:
            phi[i,j] = max_center_value

# Lists to shift rows and columns by one in the 4 directions
sright = [(i+1)%Lx for i in range(Lx)]
sleft  = [(i-1)%Lx for i in range(Lx)]
sup    = [(j+1)%Ly for j in range(Ly)]
sdown  = [(j-1)%Ly for j in range(Ly)]



################ Plotting ################
plt.figure()
plt.pcolor(phi)
plt.colorbar()

################ Time loop ################
t = 0.0
step = 0
while t < tmax:
    # Laplacian for phi
    lap_phi = (phi[sright,:] + phi[sleft,:] + phi[:,sup] + phi[:,sdown] - 4*phi)

    # reaction term for phi 
    reaction_phi = 0.5*(1 - phi)*phi*(1 - 2*phi)

    # compute integral over phenotypes: I_rho(x,y) = sum_k u[:,:,k] * drho
    I_rho = np.sum(u, axis=2) * drho  

    # coupling term in phi eqn: phi^2 (1-phi) * integral
    coupling = phi**2 * (1 - phi) * I_rho

    # dphi/dt as given
    dphi_dt = -M * (reaction_phi + epsilon2 * lap_phi + coupling)

    # update phi (explicit)
    phi_new = phi + dt * dphi_dt

    # prepare u update: compute spatial Laplacian for each phenotype
    # Vectorize across phenotype dimension
    lap_u = np.zeros_like(u)   # shape (Lx, Ly, Nrho)
    for k in range(Nrho):
        uk = u[:, :, k]
        lap_u[:, :, k] = (uk[sright,:] + uk[sleft,:] + uk[:,sup] + uk[:,sdown] - 4*uk)

    # du/dt for each phenotype: dphi_dt + D_rho[k] * lap_u[:,:,k]
    # how we interpret this -> phenotype density expands proportionally to phi growth, not the raw dphi/dt
    du_dt = np.empty_like(u)
    for k in range(Nrho):
        du_dt[:, :, k] = u[:, :, k] * dphi_dt + D_rho[k] * lap_u[:, :, k]


    # explicit update for u
    u_new = u + dt * du_dt

    # enforce simple bounds (keep phi in [0,1], u non-negative)
    phi = np.clip(phi_new, 0.0, 1.0)
    u = np.maximum(u_new, 0.0)

    # time advance
    t += dt
    step += 1

    # Plot
    if (round(t/dt)%100==0):
        plt.figure()
        plt.pcolor(phi, vmin=0, vmax=1)
        plt.colorbar(ticks=[0,0.5,1])
        plt.pause(0.001)

plt.ioff()
plt.show()
