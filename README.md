### Simulating tumor growth with hybrid Phase Field - Fisher Kolmogorov model

This project contains the code for a simulation of tumoral growth using a hybrid model that combines Phase Field dynamics (commonly used for localized, well differentiated tumors) and Fisher Kolmogorov (often used in more spread, less localized tumors). We use the work by Jimenez Sanchez et. al (2021) and Lorenzo et. al (2016) on tumoral growth as reference.

The objective is to create a model that accurately simulates tumoral growth taking into account different phenotypes (more or less aggressive) within the tumor mass.

This project is developed as final project for the subject Multiscale Mathematical Biology, part of the MsC computer science at Leiden University.

### Version 1

The first version used the following equations:

$$
\frac{\partial \phi}{\partial t} = 
-M \left( 
\frac{1}{2} (1-\phi) \phi (1-2\phi) + 
\epsilon^2 \nabla^2 \phi + 
\phi^2 (1-\phi) \int u p \, dp 
\right)
$$

$$
\frac{\partial u}{\partial t} = 
\frac{\partial \phi}{\partial t} + D_\rho \nabla_\rho^2 u
$$

The results obtained point at two possible issues:
- Numerical instability: the Laplacian grows too fast into negative values, if the diffusion coefficient is too large it might be causing the numbers to explode
- Possible wrong modeling of *du/dt*: When plotting the values of u in the lattice, we see most of them quickly evolve to 0, all across the lattice, except for a few localized pixels in the middle. The lattice reaches non-zero values (very close to 0) shortly before collapsing to 0, even in areas where there is no tumor cells present. This suggest a possible error in the definition of *du/dt*.


### Version 2

We model **spatio-temporal growth of a tumor with two phenotypes** (low-aggressive and high-aggressive) in a 2D tissue, coupled to nutrient availability. The goal is to capture how local nutrients and competition shape tumor expansion and phenotypic composition.

**Key ideas implemented:**

- **Tumor field (`phi`)**  
  - Phase-field representation of tumor: `0 = medium`, `1 = tumor core`  
  - Allen-Cahn dynamics + phenotype-weighted growth:
    ```
    dphi/dt = -M * [0.5*(1-phi)*phi*(1-2*phi) - div(epsilon * grad(phi))] + G * phi * (1-phi)
    ```
  - Interface width modulated by local tumor density:
    ```
    epsilon(x,y) = epsilon2 * (1 + alpha_eps * U_tot/K_U)
    ```

- **Phenotype dynamics (`u_k`)**  
  - Two phenotypes: low-aggressive (k=0) and high-aggressive (k=1)  
  - Diffusion + nutrient-limited logistic growth + death:
    ```
    du_k/dt = D_k * lap(u_k) + r_k * (c/(c+Kc)) * u_k * (1 - U_tot/K_U) - d_k * u_k
    ```
  - Competition for space is implicit in `(1 - U_tot/K_U)`

- **Nutrients (`c`)**
  - Diffusion, consumption by tumor, and relaxation to background:
    ```
    dc/dt = D_c * lap(c) - phi * sum(q_k * u_k) - lambda_c * (c - c_inf)
    ```

- **Initial conditions**  
  - Tumor seeded as smooth circular core  
  - Small random densities for each phenotype

- **Visualization**
  - Tumor field: `pcolor(phi)`  
  - Phenotype overlay: pixel color = mixture of yellow (low-aggressive) + red (high-aggressive) proportional to local fractions


### Version 3

We added:
- Visualization of nutrient availability
- Gradient (asymmetrical) in nutrient distribution, to encourage protuberances in tumor growth
- Implemented displacement due to competition between phenotypes