# VISRET Statistical Analysis Code  

## Overview

This repository contains the statistical analysis code used in the study evaluating the reliability, validity, and clinical utility of the Visual Image Simple Recognition Test (VISRET). The analysis was conducted using Bayesian methods to assess VISRET’s ability to detect episodic memory impairment while minimizing the impact of aphasia.

## Repository Contents  

The repository includes the following scripts:  

- **`bayesian_logistic.py`**: Bayesian logistic regression analyzing the ability of VISRET to distinguish Alzheimer’s disease (AD) patients from healthy controls (HP).  
- **`bayesian_lr_1.py`**: Bayesian linear regression (1) used for establishing cutoff values and defining the memory score.  
- **`bayesian_lr_2.py`**: Bayesian linear regression (2) evaluating the impact of aphasia on VISRET total scores.  
- **`bayesian_lr_3.py`**: Bayesian linear regression (3) evaluating the impact of aphasia on VISRET false recognition counts.  

## Reproducibility

For full reproducibility:
- Exact versions of dependencies are listed in `requirements.txt`
- Statistical methods and results are detailed in our published manuscript

## Citation

If you use this code or model in your research, please cite our paper:

```text
[Our Paper Title]
[Our Authors]
[Journal Name, Year]
[DOI or URL]
```
*Citation details will be added after the publication of our paper.*


## Note

- The repository does not include raw patient data due to privacy concerns.
- The statistical models were implemented using Bayesian inference with PyMC.
- The scripts assume that data preprocessing (e.g., reading Excel files, selecting relevant cases) has been completed beforehand.
- If you use this code, please cite the corresponding paper.

## License

This code is released under the MIT License.
