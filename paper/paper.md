---
title: 'mTRFpy: A Python package for temporal response function analysis'
tags:
  - Python
  - electrophysiology
  - temporal response function
  - cognitive neuroscience
  - computational neuroscience
  - TRF
  
authors:
  - name: Ole Bialas
    orcid: 0000-0000-0000-0000
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
	corresponding: true # (This is how to denote the corresponding author)
  - name: Jin Dou
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Edmund C. Lalor
    affiliation: "1, 2"
affiliations:
 - name: Department of Biomedical Engineering, University of Rochester, USA
   index: 1
 - name: Department of Neuroscience, University of Rochester, USA
   index: 2
date: 21 March 2023
bibliography: paper.bib
---

# Summary
Traditionally, studies on the neural processing of speech involved the repetitive display of isolated tokens (e.g. phonemes, words, sentences) where the properties of interest were carefully controlled. Recently, more researchers started investigating brain responses to more naturalistic speech like audiobooks [@hamilton2022]. However, this approach demands statistical tools to account for the different sources of variance that naturally occur in speech. One of the most popular tools to model neural responses to naturalistic speech are multivariate temporal response functions (mTRFs). 

One of the most commonly used packages for computing mTRFs with regularized regression is the mTRF-toolbox [@crosse2016]. However, this toolbox is implemented in the proprietary MATLAB language, restricting accessibility for parts of the scientific community. To overcome this constraint, we present mTRFpy, a Python package which replicates and advances the functionality of the original mTRF-toolbox.

# Background
In a nutshell, the mTRF is a regularized linear regression between two continuous signals, computed across multiple time-delays or lags. This accounts for the fact that the relationship between stimulus and neural response is not instantaneous and that the signals are auto-correlated. 

mTRFs can be used as forward or encoding models to predict (multiple) univariate brain responses as the weighted sum of various acoustic and linguistic speech features while identifying their relative contributions [@diliberto2015, broderick2018]. In this case the model's weights have a clear physiological interpretation because they denote the expected change in neural response following a unit change in a given predictor [@haufe2014]. Thus, they can be understood as a generalization of the evoked response potential, obtained from averaging responses to repetitions of prototypical stimuli for continuous data.

mTRFs can also be used as backward or decoding models which reconstruct stimulus features from multivariate neural recordings, for example to identify the speaker an individual is attending to within a cocktail party scenario [@osullivan2015]. Because a decoder pools information across all neural sensors, it can leverage interactions between individual observations and their underlying generators. Thus their predictive power is usually higher compared to encoding models. However, because the neural signals are highly interrelated, the decoder will not only amplify relevant, but suppress irrelevant signals, making a physiological interpretation of the weights difficult [@haufe2014].

# Statement of need
Temporal response functions are powerful and versatile tool to study the neural processing of speech in it's natural complexity. They also allow researchers to conduct experiments that are engaging (e.g. listening to a conversation) while monitoring the comprehension of speech independently of classical behavioral tests. This makes mTRFs promising tools for clinical applications in infants or patients with schizophrenia, autism spectrum disorder or disorder of consciousness. We believe that mTRFs can be useful to a large clinical research community and hope that open and accessible software will facilitate their wider adoption.

We implement the same methods as the original MATLAB toolbox and use a sample data set to assert that mTRFpy produces the same results (within the limits of numerical accuracy). However, we use an object oriented design, where training, optimization and visualization are implemented as methods of a generic `TRF` class. Whats more, we added functions for permutation testing and model evaluation which were not included in the original mtrf-toolbox. Finally, we also included a method to conveniently export trained models to MNE which is the most common framework for analyzing MEG and EEG data in Python [@gramfort2013].

There is some overlap with eelbrain, another Python package focused on TRF analysis [@brodbeck2021]. However, while eelbrain estimates mTRFs with the iterative boosting algorithm, `mTRFpy` uses ridge regression and while the two methods produce similar results, there are nuanced differences [@kulasingham2022]. Whats more eelbrain provides a full analysis framework with custom data types while `mTRFpy` is more minimalist and operates on list and numpy arrays. This keeps the number of dependencies to a minimum and makes the package easily compatible with most analysis pipelines. We hope that this approach will foster synergies with the large machine learning and deep learning communities working in Python. 

# Overview and Example
`mTRFpy` provides a sample of EEG recordings during comprehension of naturalistic speech.Here, we use this data set to compute and visualize a forward TRF. Then we will estimate the model's accuracy as the Pearson's correlation between the actual and predicted EEG and compare the result against randomly permuted data.

```python
import numpy as np
from matplotlib import pyplot as plt
from mtrf.model import TRF, load_sample_data
from mtrf.stats import cross_validate, permutation_distribution

stim, resp, fs = load_sample_data()
stim, resp = np.array_split(stim, 10), np.array_split(resp, 10)
trf = TRF(direction=1, method='ridge')
tmin, tmax = 0, 0.4  # time window in seconds
regularization = 10  # ridge parameter
trf.train(stim, resp, fs, tmin, tmax, regularization)
r, _ = cross_validate(trf, stim, resp, fs, tmin, tmax, regularization)
r_perm, _ = permutation_distribution(
    trf, stim, resp, fs, tmin, tmax, regularization, n_permute=1000, k=5
    )
```

# References
