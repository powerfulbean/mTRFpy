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
Temporal response functions are powerful and versatile tool to study the neural processing of speech in it's natural complexity. They also allow researchers to conduct experiments that are engaging (e.g. listening to a conversation) while monitoring the comprehension of speech independently of classical behavioral tests. This makes mTRFs promising tools for clinical applications in infants or patients with schizophrenia, autism spectrum disorder or disorder of consciousness. We believe that mTRFs can be useful to a large clinical research community and hope that open and accessible software will facilitate their wider adoption. Thus, our online documentation contains detailed tutorials to guide new users in computing and interpreting mTRFs.

- commonalities and differences between matlab and python TRFm. Potential synergies with ML community
- distinction from eelbrain


# Overview and Example
Compute and plot forward TRF and compare the correlation to the permutation distribution, plot the result


# References
