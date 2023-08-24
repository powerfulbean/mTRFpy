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
    orcid: 0000-0003-4472-7626
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
    corresponding: true # (This is how to denote the corresponding author)
  - name: Jin Dou
    orcid: 0009-0000-0539-5951
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Edmund C. Lalor
    orcid: 0000-0002-2498-6631
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
Traditionally, studies on the neural processing of speech involved the repetitive display of isolated tokens (e.g., phonemes, words, sentences) where the properties of interest were carefully controlled.
Recently, researchers have increasingly focused on investigating brain responses to more naturalistic speech like audiobooks [@hamilton2020].
However, this approach demands statistical tools to account for the different sources of variance that naturally occur in speech. 
Among the most popular tools to model neural responses to naturalistic speech are multivariate temporal response functions (mTRFs).

One of the most commonly used packages for computing mTRFs with regularized regression is the mTRF-toolbox [@crosse2016]. 
However, this toolbox is implemented in the proprietary MATLAB language, restricting accessibility for parts of the scientific community. 
To overcome this constraint, we present mTRFpy, a Python package which replicates and advances the functionality of the original mTRF-toolbox.

# Background
In a nutshell, the mTRF is a regularized linear regression between two continuous signals, computed across multiple time-delays or lags.
This accounts for the fact that the relationship between stimulus and neural response is not instantaneous and that the signals are auto-correlated.

mTRFs can be used as forward or encoding models to predict (multiple) univariate brain responses as the weighted sum of various acoustic and linguistic speech features while identifying their relative contributions [@diliberto2015, broderick2018].
In this case the model's weights have a clear physiological interpretation because they denote the expected change in neural response following a unit change in a given predictor [@haufe2014]. 
Thus, they can be understood as a generalization of the event potential, obtained from averaging responses to repetitions of prototypical stimuli for continuous data.

mTRFs can also be used as backward or decoding models which reconstruct stimulus features from multivariate neural recordings, for example to identify the speaker an individual is attending to within a cocktail party scenario [@osullivan2015]. 
Because a decoder pools information across all neural sensors, it can leverage interactions between individual observations and their underlying generators. 
Thus their predictive power is usually higher compared to encoding models. 
However, because the neural signals are highly interrelated, the decoder will not only amplify relevant, but suppress irrelevant signals, making a physiological interpretation of the weights difficult [@haufe2014].

# Statement of need
The temporal response function is a powerful and versatile tool to study the neural processing of speech in its natural complexity.
They also allow researchers to conduct experiments that are engaging (e.g., listening to a conversation) while monitoring the comprehension of speech independently of classical behavioral tests.
This makes mTRFs promising tools for clinical applications in infants or patients with schizophrenia, autism spectrum disorder or disorder of consciousness.
We believe that mTRFs can be useful to a large clinical research community and hope that open and accessible software will facilitate their wider adoption.

We implement the same methods as the original MATLAB toolbox and use a sample data set to demonstrate that mTRFpy produces the same results (within the limits of numerical accuracy).
However, we use an object oriented design, where training, optimization and visualization are implemented as methods of a generic `TRF` class.
What is more, we added functions for permutation testing and model evaluation which were not included in the original mTRF-toolbox.
Finally, we also included a method to conveniently export trained models to MNE-Python which is the most common framework for analyzing MEG and EEG data in Python [@gramfort2013].

There is some overlap with other Python packages focused on the neural processing of naturalistic speech such as eelbrain [@brodbeck2021] and naplib [@mischler2023]. 
However, while these packages provide a whole analysis framework, `mTRFpy` is more minimalist in its implementation. 
This makes `mTRFpy` easy to learn, keeps dependencies to a minimum and makes the package easy to integrate into a broad variety of analysis pipelines.

# Overview and Example
`mTRFpy` provides a sample of EEG recordings during comprehension of naturalistic speech. 
Here, we use this data set to compute and visualize a forward TRF. 
Then we estimate the model's accuracy as the Pearson's correlation between the actual and predicted EEG and compare the result against randomly permuted data.

```python
import numpy as np
from matplotlib import pyplot as plt
from mtrf.model import TRF, load_sample_data
from mtrf.stats import cross_validate, permutation_distribution
stim, resp, fs = load_sample_data(n_segments=5)
trf = TRF(direction=1, method='ridge')
tmin, tmax = 0, 0.4  # time window in seconds
regularization = 6000  # ridge parameter
trf.train(stim, resp, fs, tmin, tmax, regularization)
r, _ = cross_validate(trf, stim, resp, fs, tmin, tmax, regularization)
r_perm, _ = permutation_distribution(
    trf, stim, resp, fs, tmin, tmax, regularization, n_permute=10000, k=-1)
p = sum(r_perm>r)/len(r_perm)
fig, ax = plt.subplots(1, 2, figsize=(7, 4))
trf.plot(channel='avg', axes=ax[0], show=False, kind='image')
ax[1].hist(r_perm, bins=100)
ax[1].axvline(r, 0, 1, color='black', linestyle='--')
ax[1].set(ylabel='Number of permutations', xlabel='Correlation [r]')
ax[1].text(0.04, 250, f'p={p.round(2)}')
```
![Left panel shows the TRFs weights, averaged across channels, for each spectral band where bright yellow indicates high and dark blue indicates low weights. The histrogram on the right shows the distribution of correlation coefficients obtained by random permutation. The dashed line marks the actually observed value.](example.png)

# References
