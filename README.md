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
