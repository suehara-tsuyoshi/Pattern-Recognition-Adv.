## Instructions

Please use the following command to run algorithms.

#### EM Algorithm

`python ML_est/em_gmm.py x.csv z.csv params.dat`

#### VB Algorithm

`python Bayes_est/vb_gmm.py x.csv z.csv params.dat`

I describe contents of the outputs *z.csv* and *params.dat* here.

#### z.csv

This file stores the posterior probabilities of latent variables.
Each row corresponds to observed data and each column corresponds to a class.

#### params.dat

This file stores (variational) parameters obtained through each algorithm.
Here, I describe the final value of each parameter, paired with its parameter name.

## Discussions

What is the appropriate number of classes K 
