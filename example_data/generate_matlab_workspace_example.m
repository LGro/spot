% Spike times cell array, each entry corresponds to one neuron
neuron_spike_times = {
    [0.544, 1.231, 4.341],
    [2.0, 2.1],
    [0.1, 0.1234, 1.2, 1.24]};

% Trial time limits
trial_start_times = [0.2, 1.5];
trial_end_times = [1.5, 3.0];

% Save mat file
save('matlab_workspace_example.mat', ...
    'neuron_spike_times', 'trial_start_times', 'trial_end_times');
