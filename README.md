## SpecMod - A Toolbox for Processing and Modeling Seismic Spectra

SpecMod is very much still in development and, as such, contains bugs which may introduce inaccuracies into spectral modeling. Therefore, it should be used with caution and not without a basic understanding of processing seismic spectra.

## Usage Instructions

This code currently has a good number of dependencies and is written for python 3.7.

Crucial dependencies:

    1. ObsPy v1.1.0 - for processing time series data
    2. mtspec v0.3.2 - for calculating spectra
    3. lmfit v1.0.0 - for modeling spectra
    4. uncertainties v3.1.2 - for calculating the model uncertainties
    5. (optional) emcee v3.0.1 - for Markov-Chain Monte Carlo search
    6. sub-dependencies of all above

My suggestion is that you should use anaconda/miniconda create a new conda environment.

 $ conda create -n SpecMod python=3.7

Then install all dependencies listed in the requirements.txt file.
MAKE SURE YOU POINT TO THE CORRECT FILE! It is under the SpecMod directory.

 $ conda install -c conda-forge  --file requirements.txt

Anaconda will install all of the relevant dependencies. If this step fails, try
to just install 1-4(or 5) and anaconda should be able to manage all of the sub-dependencies.
