import numpy as np
import matplotlib.pyplot as plt

#Parameters (examples)
Q = 10000       # Total shares to execute
T = 6.0         # Time horizon (e.g., 6 hours)
N = 12          # Number of time slices
sigma = 0.01    # Price volatility (per hour)
gamma = 0.00001 # Permanent impact coefficient
eta = 0.0001    # Temporary impact coefficient

tau = T / N     # Time per slice

# Risk aversion levels to compare
lambda_values = [0.0, 0.5, 2.0, 10.0]
labels = ['Risk-Neutral', 'Low Risk', 'Moderate Risk', 'High Risk']
colors = ['#26804a', "#2e9c5c", '#56b67d', '#8dd1a8']


t_points = np.arange(N + 1) * tau
results = []

for lam in lambda_values:
  # Compute urgency parameter
  if lam < 1e-10:
      kappa = 0.0
  else:
      kappa = np.sqrt(lam * sigma ** 2 / eta)

  # Generate trajectory
  if kappa < 1e-10:
      traj = Q * (T - t_points) / T
  else:
      traj = Q * np.sinh(kappa * (T - t_points)) / np.sinh(kappa * T)

  # Convert to trading schedule
  schedule = -np.diff(traj)

  # Calculate expected cost
  perm_cost = 0.0
  cumulative = 0.0
  for i in range(len(schedule)):
      perm_cost += gamma * schedule[i] * cumulative
      cumulative += schedule[i]
  temp_cost = eta * np.sum(schedule ** 2 / tau)
  exp_cost = perm_cost + temp_cost

  # Calculate variance
  var = sigma ** 2 * tau * np.sum(traj[1:] ** 2)

  results.append({
      'lambda': lam,
      'kappa': kappa,
      'trajectory': traj,
      'schedule': schedule,
      'expected_cost': exp_cost,
      'variance': var,
      'std_dev': np.sqrt(var)
  })

#plotting
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left plot: Trajectories (shares remaining over time)
ax1 = axes[0]
for i, r in enumerate(results):
  ax1.plot(t_points, r['trajectory'], color=colors[i],
           linewidth=2.5, label=labels[i])

ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Shares Remaining')
ax1.set_title('Optimal Execution Trajectories')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(0, T)
ax1.set_ylim(0, Q * 1.05)
ax1.grid(True, alpha=0.3)

# Right plot: Cost-Risk Trade-off
ax2 = axes[1]
exp_costs = [r['expected_cost'] for r in results]
std_devs = [r['std_dev'] for r in results]

for i in range(len(results)):
  ax2.scatter(std_devs[i], exp_costs[i], color=colors[i], s=120,
              zorder=5, edgecolor='white', linewidth=2)

ax2.plot(std_devs, exp_costs, 'k--', alpha=0.4, linewidth=1.5)
ax2.set_xlabel('Cost Std Dev (dollars)')
ax2.set_ylabel('Expected Cost (dollars)')
ax2.set_title('Cost-Risk Trade-off')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("=" * 65)
print("ALMGREN-CHRISS EXECUTION COMPARISON")
print("=" * 65)
print(f"{'Lambda':<8} {'Kappa':<8} {'E[Cost]':<12} {'Std Dev':<12} {'First Slice':<12}")
print("-" * 65)
for r in results:
  print(f"{r['lambda']:<8.1f} {r['kappa']:<8.3f} {r['expected_cost']:<12.2f} {r['std_dev']:<12.2f} {r['schedule'][0]:<12.1f}")
print("=" * 65)