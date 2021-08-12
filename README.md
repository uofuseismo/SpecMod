SpecMod - A Python-Based Toolbox for Processing and Modeling Seismic Spectra
===============================
SpecMod was designed following method of spectral modeling described in Edwards et al. (2010).

Benjamin Edwards, Bettina Allmann, Donat Fäh, John Clinton, Automatic computation of moment magnitudes for small earthquakes and the scaling of local to moment magnitude, Geophysical Journal International, Volume 183, Issue 1, October 2010, Pages 407–420, https://doi.org/10.1111/j.1365-246X.2010.04743.x


SpecMod is still an early protype and may contain bugs which could introduce inaccuracies into spectral modeling. Therefore, it should be used with caution and not without a basic understanding of digital signal processing and data modeling. It is possible the code will
go through major redesigns in later versions, so this is a concept module. 

## Usage Instructions

This code is written for python 3.7 but is usable in versions 3.6, 3.7 and 3.8.

Crucial dependencies:

    1. ObsPy v1.1.0 - for processing time series data
    2. mtspec v0.3.2 - for calculating spectra
    3. lmfit v1.0.0 - for modeling spectra
    4. numdifftools v0.9.39  - for calculating the model uncertainties
    5. (optional) emcee v3.0.1 - for Markov-Chain Monte Carlo search
    6. sub-dependencies of all above

My suggestion is that you should use anaconda/miniconda create a new conda environment.

 $ conda create -n SpecMod python=3.X

Then install all dependencies listed above. Conda should install the sub-dependencies for you.
Theoretically, the code should work with the latest versions of the dependencies, unless they made a big change.
If there is an issue please create an issue and mention the versions of the listed modules you are using.
