import numpy as np
from scipy import io
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in-file', help='MATLAB (.mat) file containing the data')
parser.add_argument('-o', '--out-file-prefix', help='Prefix for the newly created numpy files')
args = parser.parse_args()


print('MATLAB data conversion starting ...')

print('Attempting to convert:', args.in_file)

matlab_workspace = io.loadmat(args.in_file)


if 'neuron_spike_times' not in matlab_workspace.keys():
    print('ERROR: No variable "neuron_spike_times" found in the given .mat file.')

if 'trial_start_times' not in matlab_workspace.keys():
    print('ERROR: No variable "trial_start_times" found in the given .mat file.')

if 'trial_end_times' not in matlab_workspace.keys():
    print('ERROR: No variable "trial_end_times" found in the given .mat file.')


neuron_spike_times = matlab_workspace['neuron_spike_times'].flatten()
trial_start_times = matlab_workspace['trial_start_times'].flatten()
trial_end_times = matlab_workspace['trial_end_times'].flatten()


n_neurons = len(neuron_spike_times)

print('Found spike times for {} neurons'.format(n_neurons))


assert len(trial_start_times) == len(trial_end_times), 'As many trial start as end times required'

n_trials = len(trial_start_times)


spike_times = np.array([], dtype=float)
ii_spike_times = np.zeros((n_trials, n_neurons, 2), dtype=int)
i_spike_times = 0
for i_trial, trial_start_time in enumerate(trial_start_times):
    trial_end_time = trial_end_times[i_trial]
    trial_duration = trial_end_time - trial_start_time

    if trial_duration <= 0:
        print('INFO: Skipping trial {} because its duration is zero.'.format(i_trial))
        continue

    for i_neuron in range(len(neuron_spike_times)):
        trial_neuron_spike_times = neuron_spike_times[i_neuron].flatten()
        if len(trial_neuron_spike_times) == 0:
            continue

        trial_neuron_spikes_mask = np.logical_and(
            trial_neuron_spike_times >= trial_start_time,
            trial_neuron_spike_times < trial_end_time)

        trial_neuron_spike_times = trial_neuron_spike_times[trial_neuron_spikes_mask]
        n_trial_neuron_spikes = len(trial_neuron_spike_times)

        ii_spike_times[i_trial, i_neuron] = [i_spike_times, i_spike_times+n_trial_neuron_spikes]

        spike_times = np.append(spike_times, trial_neuron_spike_times)
        i_spike_times += n_trial_neuron_spikes


spike_times_file = args.out_file_prefix+'spike_times.npy'
ii_spike_times_file = args.out_file_prefix+'ii_spike_times.npy'

print('Saving converted data in', spike_times_file, 'and', ii_spike_times_file)

np.save(spike_times_file, spike_times)
np.save(ii_spike_times_file, ii_spike_times)
