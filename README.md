# l96_rnn
Using Probabilistic Machine Learning to Better Model Temporal Patterns in Parameterizations: a case study with the Lorenz 96 model.
https://arxiv.org/abs/2203.14814

## Parent directory ##

`requirements.txt` gives the required packages. Everything here is configured to run without needing a GPU. Of course, if you have a GPU things will be quicker. You can install requirements using a conda env as below:
1. `conda create --prefix ./envs python==3.8.5`
2. `conda activate ./envs`
3. `pip install -r requirements.txt`

In all notebooks, paths need to be updated based on where data is and where you want things to be saved.

## create_l96_data ##

This folder is used for preparing all the training and evaluation data from the "truth" two-level L96 model.

- `create_l96_data.ipynb` creates the data. 
- If you can't run a full 50,000 MTU simulation run (e.g. due to OOM issues), you can run consecutive chunks and then merge them using `merge_evaluation_datasets.ipynb`.
- `weather_analysis_truth_data.ipynb` is used to create the data used for weather analysis.

## saved_models ##

Here are all the models, both their training and how they are used to create the results.

### RNN ###

`rnn_training.ipynb` is for training the model and `rnn_results.ipynb` is for generating data and calculating hold-out likelihoods. `rnn_diagnostics.ipynb` contains an example of how likelihood is used to diagnose what can be improved in the RNN model.

### Polynomial ###

`Christensen_polynomial_parameterisation.ipynb` trains the polynomial model and is used to calculate hold-out likelihood.

`generate_data.py` is used to create long simulation runs, whilst `generate_weather_data.py` is used to create weather data. 

You would run them like this `python generate_data.py 20 "path/to/inital/data"`, where F=20 for this example.

### GAN ###

`gan_training.py` is used to train the GAN. `generate_data.py` is used to create long simulation runs, whilst `generate_weather_data.py` is used to create weather data. They can be run in a similar manner to their analogues in the polynomial folder.

The importance sampler is trained in `importance_sampler_for_gan_training.py` and the AR1 parameters are then calculated in `gan_training_ar_parameters.ipynb`. The results (i.e. hold-out likelihood) are calculated in `importance_sampler_for_gan_results.py`

## analysis ##

If due to OOM issues you've needed to create separate chunks of simulation data, `merge_simulation_datasets.ipynb` is there to merge it.

The notebooks in `analysis_notebook` are used to create the plots and resulst shown in the results of the paper.

