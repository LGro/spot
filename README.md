# Spike Pattern Optimal Transport

The Python 3 module `spot` implements the Spike Pattern Optimal Transport Dissimilarity described in Grossberger, L., Battaglia, F. and Vinck, M. (2018). Unsupervised clustering of temporal patterns in high-dimensional neuronal ensembles using a novel dissimilarity measure. *PLoS Comput. Biol.*


## Setup

The dependencies can be installed by running `./env_setup.sh <ENV_NAME>` with the optional argument specifying the target environment (which must be source-able).

To setup the module, run `python setup.py install`.

A jupyter notebook is available in `notebooks/`, along with a demo dataset, showing an example workflow for the SPOTDisClust method.

### Linux (Anaconda)
1) `conda create -n spot python=3`
1) `./env_setup.sh spot`
1) `source activate spot`
1) `python setup.py install`


### Windows (Anaconda)
1) `conda create -n spot python=3`
1) `conda activate spot`
1) `conda install python=3.6.5`
1) `conda install -c conda-forge hdbscan=0.8.13=py36_0`
1) `conda install numba`
1) `conda install ipykernel`
1) `conda install matplotlib`
1) `python setup.py install`


## Loading data from a MATLAB workspace

For your convenience, the script `scripts/from_matlab.py -i example_data/matlab_workspace_example.mat -o example_data/matlab_` converts spike data from a MATLAB (.mat) workspace file, specified by `-i`, to the format required by SPOTDis and saves the resulting files with the file name prefix, specified by `-o`.
For this script to work, make sure that it contains the following variables (or follow the `example_data/generate_matlab_workspace_example.m`, which generated the MATLAB example):
 - `neuron_spike_times` cell array with one entry per neuron containing a vector of spike times
 - `trial_start_times` vector containing start times for all trials (inclusive)
 - `trial_end_times` vector containing end times for all trials (exclusive)
