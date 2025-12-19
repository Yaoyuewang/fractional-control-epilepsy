# Stabilizing Fractional Dynamical Networks Suppresses Epileptic Seizures

This repository contains the code used in the paper:

**Stabilizing Fractional Dynamical Networks Suppresses Epileptic Seizures**
Yaoyue Wang, Arian Ashourvan, Guilherme Ramos, Paul Bogdan, Emily A. Pereira

## Overview

This code implements a **fractional-order dynamical network modeling and control pipeline** for intracranial EEG (iEEG) data. The main components include:

* Estimation of **fractional-order exponents** using Haar wavelets
* Identification of **fractional-order dynamical network models** from multichannel iEEG
* Eigenvalue-based **stability analysis** of the estimated networks
* Design of a **sparse stabilizing state-feedback controller**
* In-silico reconstruction and evaluation of controlled signals

The pipeline is evaluated on real patient iEEG recordings across multiple seizures.

## Citation

If you use this code, please cite:

```
Wang, Y., Ashourvan, A., Ramos, G., Bogdan, P., & Pereira, E. A.
Stabilizing Fractional Dynamical Networks Suppresses Epileptic Seizures.
arXiv preprint, 2025.
```

## Data

Intracranial EEG data are obtained from the International Epilepsy Electrophysiology Portal (IEEG Portal).

## Notes

* Code is intended for research use
* Control results are validated in silico using estimated models
* See the paper for full methodological details
