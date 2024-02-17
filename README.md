# Individual Research Module (IM): Architecture Search for a Lightweight rs-fMRI CNN for Brain Age Prediction

_This work fulfills the requirements for the "Individual Research Module" (15 ECTS) of the MSc Cognitive Systems at the University of Potsdam. It was conducted in the form of an internship at the [Machine learning in Clinical Neuroimaging Lab](https://psychiatrie-psychotherapie.charite.de/en/research/translation_and_neurotechnology/machine_learning/) ([@ritterlab](https://github.com/ritterlab)) at the Charité Universitätsmedizin Berlin._

In this project, I developed a lightweight one-dimensional convolutional neural network (1D-CNN) to predict a subject's brain age using the mean activation timeseries of independent components of rs-fMRI recordings of the UK Biobank (UKBB).
Two data modalities were tested against each other: 25 vs. 100 independent components, as provided by the UKBB.

The project consisted of three stages: 
In the first step, I investigated whether this lightweight 1D-CNN approach using ICA timeseries is promising by means of a series of manual explorations (--> [ManualSearch](src/ManualSearch)). 
Finding potential, I created a `variable1DCNN` model class which allows flexible modifications for both broader architectural settings as well as typical hyperparameters. Using this model class, I performed a big hyperparameter sweep for both data modalities in order to find the best possible settings to train well-performing 1D-CNN models (--> [HyperparameterSweep](src/HyperparameterSweep/)).
In the last step, I used the uncovered architectural settings and hyperparameters that lead to the best validation results during the sweep to train one `variable1DCNN` model for each data modality and analysed their performance (--> [FinalModels](src/FinalModels/)).

## Structure
- **data:** doesn't contain the data itself but supplementary files
- **figures:** figures created for my report (includes dataset age distribution, model training development, BAG illustrations)
- **src:** all things code; further divided into ManualSearch, HyperparameterSearch, FinalModels and my ukbb_package
- **tracking:** models' tracked training (from manual exploration + the final models; sweep tracked using W&B)

More detailed information can be found in the directories themselves.

## Requirements
You should be able to create a working environment using the provided environment files.

e.g. with conda: `conda create -n ENVNAME --file cuda02_env_export_explicit.txt` or `conda env create -n ENVNAME --file cuda02_env_export_compatible.yml`

You might need to add this repository's src directory to your package/module paths so you can use ukbb_package properly. E.g.:
`conda develop /path/to/repo/IM/src/`
