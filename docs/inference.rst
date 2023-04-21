Statistical inference
=====================

The section on regularization showed how to obtain an unbiased estimate of the models accuracy. The estimated correlation between the predicted and actual response was 0.018. To test whether this value is significant we can use permutation testing. The `permutation_distribution()` function from the `stats` module randomizes the data so that each stimulus is paired with a random response. Then, it computes a trf model and estimates its accuracy. This process is repeated (usually 10000 times or more) to obtain the permutation distribution which reflects the expected accuracy if there is no causal relationship between stimulus and response. The p-value of the actual observation is given by the probability of obtaining that or a larger value under the permutation distribution. Thus, a low p-value means that it is unlikely to observe that prediction accuracy if the relationship between stimulus and response was random. In the below example 

::
    
    import numpy as np
    from matplotlib import pyplot as plt
    from mtrf.model import TRF, load_sample_data
    from mtrf.stats import permutation_distribution
    trf = TRF()  # use forward model
    stimulus, response, fs = load_sample_data() # data will be downloaded
    stimulus = np.array_split(stimulus, 10)
    response = np.array_split(response, 10)
    for i in range(len(stimulus)):
        stimulus[i] = (stimulus[i] - stimulus[i].mean(axis=0))/stimulus[i].std(axis=0)
        response[i] = (response[i]- response[i].mean(axis=0))/response[i].std(axis=0)
    tmin, tmax = 0, 0.4  # range of time lags
    regularization = np.logspace(-1, 6, 10)
    r_obs, mse_obs = trf.fit(
        stimulus, response, fs, tmin, tmax, regularization, k=-1
        )
    r_obs = r_obs[np.argmin(mse_obs)]  # pick the best model
    r_perm, mse_perm = permutation_distribution(
        trf, stimulus, response, fs, tmin, tmax, trf.regularization, n_permute=10000
        )
    p = sum(r_perm>=r_obs)/len(r_perm)
    plt.hist(r_perm, bins=200)
    plt.axvline(x=r_obs, ymin=0, ymax=1, color='black', linestyle='--')
    plt.xlabel('Correlation [r]')
    plt.hist(r_perm, bins=200)
    plt.axvline(x=r_obs, ymin=0, ymax=1, color='black', linestyle='--')
    plt.xlabel('Correlation [r]')
    plt.ylabel('Number of models')
    plt.annotate(f'p={p.round(2)}', (0.06, 175))
    plt.show()
    plt.ylabel('Number of models')
    plt.annotate(f'p={p.round(2)}', (0.06, 175))
    plt.show()

.. image:: images/perm.png

The p-value of the observed is 0.41, meaning that we would not reject the null hypothesis that there is no relationship between stimulus and response. This is not surprising given that EEG has a poor signal-to-noise ratio and we are only using two minutes of data.

