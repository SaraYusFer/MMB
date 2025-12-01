### Simulating tumor growth with hybrid Phase Field - Fisher Kolmogorov model

This project contains the code for a simulation of tumoral growth using a hybrid model that combines Phase Field dynamics (commonly used for localized, well differentiated tumors) and Fisher Kolmogorov (often used in more spread, less localized tumors). We use the work by Jimenez Sanchez et. al (2021) and Lorenzo et. al (2016) on tumoral growth as reference.

The objective is to create a model that accurately simulates tumoral growth taking into account different phenotypes (more or less aggressive) within the tumor mass and varying nutrient concentrations in the environment.

This project is developed as final project for the subject Multiscale Mathematical Biology, part of the MsC computer science at Leiden University.

#### Summary

The first paper describes a model of tumor development that explores the behaviour (growth) of the tumor at different spatial concentrations of nutrients. They observe the tumors tend to develop non-spherical ('finger-like') shapes.

The second paper focuses on evolution and competition inside the tumor mass itself. It considers different phenotypes of tumor cells (more/less aggressive). The authors observe that the more aggressive types tend to move to the edges and displace the less aggressive types to the centre.

Our project aims to combine the both. We model:

- A tumor growing in a environment with changing levels of spatial nutrien concentration
- Composed of different phenotypes (different levels of aggressiveness)

We simulate the temporal and spatial evolution of the tumor, nutrients, and cell concentration per phenotype