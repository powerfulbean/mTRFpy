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
    affiliation: 2
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

Traditionally, studies on the neural processing of speech involved the repetitive display of isolated tokens (e.g. phonemes, words, sentences) where the properties of interest were carefully controlled. Recently, more researchers started investigating brain responses to more naturalistic speech like audiobooks. However, this de

The brain is such a complex system that is hard to be fully understood. 
A better understanding of how the human brain works will benefit both 
health care and artificial intelligence. The sensory electrophysiology 
in human is a field of exploring how the human brain processes and represents 
the information from the outside world. The researchers in this field 
normally present participants with stimuli and analyze how their brain 
responses to them by collecting their neural responses. One technique to 
model the mapping between stimuli and brain responses is the Temporal Response Function 
which treats the brain as a linear time-invariant system.

# Statement of need

`mTRFpy` is a pure Numpy based Python Package for fitting the Temporal 
Response Function. It adapts and extends the matlab mTRF-toolbox [@crosse2016multivariate] which has been 
widely used in the field of sensory electrophysiology for many years. 
Python enables the flexible interaction of the TRF method with various
of open-source Python libraries that are actively maintained, and exposing 
the TRF method to a larger society of users who are familiar with the mTRF-toolbox 
but may want to transfer from Matlab to Python. The API for `mTRFpy` provides 
object-oriented organization of functionality for fitting and applying TRFs 
including training, predicting, and interpretation of TRF, plus cross-validation 
and hyper-parameter searching. The core design philosophy of `mTRFpy` is flat and transparent, 
which makes it robust to be extended and easy to be deployed by the users. 
In `mTRFpy`, all the data being involved is purely represented by Numpy array and python list 
without wrapped with customized classes. `mTRFpy` also provides interface to the widely-used 
lower-level brain signal analysis library MNE [@gramfort2013meg] without additional wrapping of 
any functions in it, to make it less confusing when integrating both libraries together.
`mTRFpy` also provides test case to make sure it generates almost the same results as the Matlab version 
with high precision.

`mTRFpy` was designed to be used by scientists and engineers who need analyze 
brain responses and also students taking courses involving brain signal analysis. 
The matlab mTRF-toolbox has been used in a lot of scientific publications [@broderick2018electrophysiological; @keshishian2023joint]. 
Given the `mTRFpy` implements and extends the core functionality of mTRF-toolbox, 
the seamless integration of `mTRFpy` with other python libraries will contribute 
to more efficient brain signal analysis and finally more exciting new findings 
about how the brain processes and represents the outside real world.

# Acknowledgements

We acknowledge the support from the Lalor Lab where this library is developed.

# References
