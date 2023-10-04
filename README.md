# GromovMatcher

The GromovMatcher method is an optimal transport based algorithm for matching pairs of LC-MS untargeted metabolomic datasets.

This algorithm is part of our paper: "Optimal transport for automatic alignment of untargeted metabolomic data".


## Installation

To install GromovMatcher, clone the github repo
```
git clone git@github.com:sgstepaniants/GromovMatcher.git
```


## Datasets

GromovMatcher was validated on randomly generated splits of the EXPOsOMICs LC-MS dataset of cord blood samples from 500 newborns (see "Validation and comparison on ground-truth data" in paper). The dataset can be downloaded at

https://www.ebi.ac.uk/metabolights/MTBLS1684/files

at the bottom of the "Files" section under the name `metabolomics\_normalized\_data.xlsx'.

Our method was experimentally test on LC-MS data from three studies which were part of the European Prospective Investigation into Cancer and Nutrition (EPIC) cohort (see "Application to EPIC data" in paper). This data is not publicly available, but access requests can be submitted to the Steering Committee at

https://epic.iarc.fr/access/submit_appl_access.php


## Code Dependencies

### Unbalanced Gromov-Wasserstein
GromovMatcher uses the Unbalanced Gromov-Wasserstein (UGW) algorithm to perform its matchings. Our implementation of the UGW algorithm in `src/ugw_sinkhorn_solver.py` is an extension of the original code at https://github.com/thibsej/unbalanced_gromov_wasserstein.

### Existing Alignment Methods
On validation and experimental EPIC data, we compare GromovMatcher to two state-of-the-art alignment methods.

#### M2S
The code for the latest M2S toolbox is located in `experiments/M2S_experiments/M2S` taken from the github repository https://github.com/rjdossan/M2S.

#### MetabCombiner (mC)
The code for the latest metabCombiner toolbox is located in `experiments/mC_experiments/???` taken from the github repository https://github.com/hhabra/metabCombiner.


## GromovMatcher Algorithm
The GromovMatcher algorithm consists of two main steps:
1. Matching metabolic features with Unbalanced Gromov-Wasserstein in `src/ugw_sinkhorn_solver.py`
2. Filtering outlier matches through retention time drift estimation in `src/filtering.py`

For details of both these steps see the "Results" and "Methods" sections of our paper.

The entire GromovMatcher algorithm combining both these steps is defined in `src/GromovMatcher.py` under the `GM` function.

## GromovMatcher Tutorial
The application of GromovMatcher is detailed in the notebook `tutorial/GM_Example.ipynb`. Two toy datasets are provided as an example.
