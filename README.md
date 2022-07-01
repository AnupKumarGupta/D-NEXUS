# D-NEXUS: Defending Text Networks using Summarization
Codes and implementation files to be used as a reference to the paper: [**D-NEXUS: Defending Text Networks using Summarization**](https://doi.org/10.1016/j.elerap.2022.101171)

***Status - Published in Elsevier's Electronic Commerce Research and Applications*** 

## Overview
This repository contains the following files:

- **Attack_Data** - Contains data obtained by running `run_attack.sh` over the datasets for different models.
- **run_attack.sh** - For generating attack files using [textattack](https://github.com/QData/TextAttack) library.
- **Summarizer.py** -  For running proposed defense on attacked data.
- **Timer_Summ.py** - For inference time comparision with and without the application of the proposed defense.

Further details about the methodology may be directly referred to from the published study.

## Citation  
If you intend to use this work, kindly cite us as follows:  

```
@article{GUPTA2022DEFENDING,
title = {D-NEXUS: Defending Text Networks Using Summarization},
journal = {Electronic Commerce Research and Applications},
pages = {101171},
year = {2022},
issn = {1567-4223},
doi = {https://doi.org/10.1016/j.elerap.2022.101171},
url = {https://www.sciencedirect.com/science/article/pii/S1567422322000552},
author = {Anup Kumar Gupta and Aryan Rastogi and Vardhan Paliwal and Fyse Nassar and Puneet Gupta},
keywords = {Sentiment Analysis, Natural Language Processing, Adversarial Defenses, Transformers, Adversarial Attack, Language Summarization},
}
```
