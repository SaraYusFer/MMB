"""
Changes from v2:
- Add competition between phenotypes (displacement)
- visualize nutrient distribution
- create gradient in  nutrient distribution to encourage formation of non spheric shapes
"""
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio
from IPython.display import Image
from tqdm import tqdm
from datetime import datetime

################ Parameters and variables ################
Nrho = 2                     # number of phenotypes
drho = 1.0 / Nrho            # discrete phenotypic spacing (simple Riemann sum)

epsilon2 = 5.0               # controls interface width (tumor diffuse edge)
M = 1.0                      # motility
alpha_eps = 0.1              # controls how strongly the phenotype field modifies the phase-field interfacial width.
# D_rho: diffusion coefficient per phenotype bin. We only use high and low aggresivity phenotypes
D_rho = np.array([0.05, 0.25])  

# nutrient parameters
D_c = 1.0        # nutrient diffusion
gamma_c = 0.5    # consumption coefficient
c_inf = 1.0      # background nutrient level
lambda_c = 0.01  # relaxation to background

# phenotype-specific rates
r = np.array([0.2, 0.6])   # proliferation rates: low, high
q = np.array([0.5, 1.0])   # nutrient uptake per phenotype
d = np.array([0.01, 0.01]) # death rates

# carrying capacity for total phenotype density
K_U = 1.0

# time
dt = 0.05
tmax = 3.0

# helpers
frames = []
nutrient_frames=[]
rho_overlay_frames = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


############# Initial conditions ###########
# Spatial grid
Lx=100
Ly=100
# Field initialized to all zeros (only medium)
phi = np.zeros((Lx,Ly))
# Phenotype bins (u): small random initial densities for each phenotype. u is 3D: (x, y, phenotype)
u = 0.2 * np.random.random((Lx, Ly, Nrho))   

# Addition from students: initial tumor core
cx, cy = Lx//2, Ly//2
radius = 12.0      
edge_width = 15.0   # How gradual the edge is (larger -> smoother)
max_center_value = 0.95  # peak phi in center (keeps it from being exactly 1 everywhere)

for i in range(Lx):
    for j in range(Ly):
        dist = np.sqrt((i-cx)**2 + (j-cy)**2)
        # a smooth plateau near the center, decaying to 0 over 'edge_width'
        phi[i, j] = max_center_value * np.exp(-((dist)**2) / (2*(edge_width**2)))
        if dist <= radius:
            phi[i,j] = max_center_value

# Addition from students: nutrient field
# initialize nutrient field c with nutrient gradient (asymmetrical on one axis)
c = c_inf * (np.random.rand(Lx, Ly))
# Reduce nutrient inside initial tumor core to mimic consumption
c[phi > 0.5] = 0.8 * c_inf

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

print('Running simulation')
for t in tqdm(np.arange(0, tmax, dt)):
    # compute central neighbor fields
    phi_r = phi[sright, :]
    phi_l = phi[sleft, :]
    phi_u = phi[:, sup]
    phi_d = phi[:, sdown]

    # compute I_rho and phenotype totals
    U_tot = np.sum(u, axis=2) * drho     # total local density (shape Lx,Ly)
    # avoid divide by zero
    U_tot_safe = U_tot + 1e-12

    # normalized aggressive fraction (use phenotype 1 as aggressive)
    frac_aggr = u[:, :, 1] / U_tot_safe   # shape Lx,Ly; may be large if U_tot small
    frac_aggr = np.nan_to_num(frac_aggr, 0.0)

    # compute epsilon2(x,y) from phenotype metric (use U_tot normalized)
    # normalize U_tot to [0,1] by dividing by K_U
    m = np.clip(U_tot / K_U, 0.0, 1.0)
    epsilon2_xy = epsilon2 * (1.0 + alpha_eps * m)
    # clip to safe bounds
    epsilon2_xy = np.clip(epsilon2_xy, 0.1 * epsilon2, 3.0 * epsilon2)

    # Divergence form for variable-coefficient laplacian: div(eps * grad phi)
    eps = epsilon2_xy
    eps_r = 0.5 * (eps + eps[sright, :])
    eps_l = 0.5 * (eps + eps[sleft, :])
    eps_u = 0.5 * (eps + eps[:, sup])
    eps_d = 0.5 * (eps + eps[:, sdown])

    flux_r = eps_r * (phi_r - phi)
    flux_l = eps_l * (phi_l - phi)
    flux_u = eps_u * (phi_u - phi)
    flux_d = eps_d * (phi_d - phi)

    div_eps_grad_phi = flux_r + flux_l + flux_u + flux_d

    # reaction term
    reaction_phi = 0.5 * (1 - phi) * phi * (1 - 2 * phi)

    # phenotype-weighted local growth G(x)
    # nutrient-limited proliferation factor: Monod form
    Kc = 0.2   # half-saturation constant for nutrient
    cons = c / (c + Kc)
    # growth contributions from each phenotype (weighted by fraction in total U)
    G = np.zeros_like(phi)
    for k in range(Nrho):
        G += r[k] * cons * (u[:, :, k] / U_tot_safe)

    # phi growth localized near interface: multiply by phi*(1-phi)
    growth_term = G * (phi * (1 - phi))

    # dphi/dt: Allen-Cahn + growth 
    dphi_dt = -M * (reaction_phi - div_eps_grad_phi) + growth_term

    # Nutrient update: diffusion + consumption by tumor cells (localized by phi)
    # discrete laplacian for c
    c_lap = (c[sright, :] + c[sleft, :] + c[:, sup] + c[:, sdown] - 4.0 * c)
    uptake = gamma_c * (q[0] * u[:, :, 0] + q[1] * u[:, :, 1]) * phi
    dc_dt = D_c * c_lap - uptake - lambda_c * (c - c_inf)

    # Phenotype updates: diffusion + nutrient-driven logistic growth
    lap_u = np.zeros_like(u)
    for k in range(Nrho):
        uk = u[:, :, k]
        lap_u[:, :, k] = (uk[sright, :] + uk[sleft, :] + uk[:, sup] + uk[:, sdown] - 4.0 * uk)

    du_dt = np.zeros_like(u)
    # Define a bias array per phenotype (affects competition degree)
    bias = np.array([1.0, 1.5])
    for k in range(Nrho):
        # competition favours aggressive phenotype (k=1)
        competition = (1.0 - U_tot / K_U) * bias[k]
        growth = r[k] * (c / (c + Kc)) * u[:, :, k] * competition
        death = -d[k] * u[:, :, k]
        diffusion = D_rho[k] * lap_u[:, :, k]
        du_dt[:, :, k] = diffusion + growth + death

    # explicit updates (clip/guard)
    phi = np.clip(phi + dt * dphi_dt, 0.0, 1.0)
    u = np.maximum(u + dt * du_dt, 0.0)
    c = np.maximum(c + dt * dc_dt, 0.0)

    # time advance
    t += dt
    step += 1

    # Plot
    
    if (round(t/dt)%1==0):

        ####### Fields (phi)
        fig, ax = plt.subplots()
        col = ax.pcolor(phi, vmin=0, vmax=1)
        fig.colorbar(col)

        buf = io.BytesIO()
        fig.savefig(buf, format='png') 
        buf.seek(0)
        frames.append(imageio.imread(buf))

        plt.close(fig)


        ####### Phenotyes (rho) overlayed as mixed color
        rgb_img = np.zeros((phi.shape[0], phi.shape[1], 3))

        # Tumor pixels only
        tumor_indices = np.argwhere(phi > 0.1)
        for i, j in tumor_indices:
            low = u[i, j, 0]
            high = u[i, j, 1]
            total = low + high + 1e-12  # avoid divide by zero
            # normalized contributions
            low_frac = low / total
            high_frac = high / total
            # assign color: yellow = [1,1,0], red = [1,0,0]
            rgb_img[i, j, :] = low_frac * np.array([1, 1, 0]) + high_frac * np.array([1, 0, 0])


        fig, ax = plt.subplots()
        ax.imshow(rgb_img, origin='lower')
        ax.set_title("Phenotype Overlay")
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        rho_overlay_frames.append(imageio.imread(buf))
        plt.close(fig)

        ##### Nutrients
        # Normalize nutrient to [0,1]
        c_norm = np.clip(c / c_inf, 0, 1)

        # Initialize RGB image
        rgb_img = np.zeros((Lx, Ly, 3))

        # Map nutrient to blue channel
        rgb_img[:, :, 2] = c_norm   # high nutrients = more blue

        # Map tumor to red channel
        rgb_img[:, :, 0] = phi       # tumor density = red

        # Optional: green = phi * c_norm (interaction)
        rgb_img[:, :, 1] = phi * c_norm

        fig, ax = plt.subplots()
        ax.imshow(rgb_img, origin='lower')
        ax.set_title("Tumor (red) + Nutrient (blue) overlay")

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        nutrient_frames.append(imageio.imread(buf))
        plt.close(fig)

# Save GIFs
os.makedirs('./results/v3', exist_ok=True)

imageio.mimsave(f'results/v3/simulation_{timestamp}.gif', frames, fps=10)
imageio.mimsave(f'results/v3/simulation_rho_{timestamp}.gif', rho_overlay_frames, fps=10)
imageio.mimsave(f'results/v3/simulation_nutrients_{timestamp}.gif', nutrient_frames, fps=10)

plt.ioff()
