import torch

# Parameters (as in your encoder)
obs_min, obs_max = -3.0, 3.0
pop_dim = 20  # or whatever your config uses
num_steps = 5

spacing = torch.linspace(0, 1, pop_dim)
means = obs_min + spacing * (obs_max - obs_min)  # shape [pop_dim]
delta_mean = means[1] - means[0]
stds = torch.full_like(means, delta_mean / 2.0)  # shape [pop_dim]

# Range to test
x = torch.linspace(obs_min, obs_max, 500)

# For each x, compute population activity
pop_activity = torch.exp(-0.5 * ((x.unsqueeze(1) - means) / stds) ** 2)  # [500, pop_dim]

# For each x, sum population activity (total current to all neurons)
total_current = pop_activity.sum(dim=1)  # [500]

# Simulate IF neuron for each x, for num_steps
def simulate_spikes(current, steps, threshold=1.0):
    mem = torch.zeros_like(current)
    spikes = torch.zeros_like(current)
    for _ in range(steps):
        mem = mem + current
        spk = (mem >= threshold).float()
        spikes += spk
        mem = mem - spk * threshold  # reset by subtraction
    return spikes

spikes = simulate_spikes(total_current, num_steps)
spike_rate = spikes / num_steps

min_rate = spike_rate.min().item()
max_rate = spike_rate.max().item()

print(f"Minimum spike rate over {num_steps} steps: {min_rate:.3f}")
print(f"Maximum spike rate over {num_steps} steps: {max_rate:.3f}")
