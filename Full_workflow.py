import numpy as np
import matplotlib.pyplot as plt


#Variables to be changed for each execution (below is an example scenario)
print("Signal: Momentum model predicts +2% move")
print("Alpha half-life: 2 hours")
print("Order: Buy 50,000 shares")
print("Time horizon: 4 hours")
print("Current price: 75.00 dollars")
print()

# Parameters for execution
Q = 50000           # Shares to buy
T = 4.0             # Hours
N = 8               # 30-minute intervals
sigma = 0.015       # volatility
gamma = 0.000005    # Permanent impact
eta = 0.00008       # Temporary impact
decision_price = 75.00

# Before the trade- Choose lambda based on alpha decay
print("Before the trade analysis")
print("-" * 40)

alpha_halflife = 2.0
target_kappa = 1.0 / alpha_halflife
suggested_lambda = (target_kappa ** 2) * eta / (sigma ** 2)

print(f"Alpha half-life: {alpha_halflife} hours")
print(f"Suggested lambda: {suggested_lambda:.3f}")
print(f"Target kappa: {target_kappa:.3f}")

# Use suggested lambda
lambda_risk = suggested_lambda
tau = T / N

kappa = np.sqrt(lambda_risk * sigma ** 2 / eta)
t_points = np.arange(N + 1) * tau
trajectory = Q * np.sinh(kappa * (T - t_points)) / np.sinh(kappa * T)
schedule = -np.diff(trajectory)

# Estimate costs
perm_cost = 0.0
cumulative = 0.0
for i in range(len(schedule)):
  perm_cost += gamma * schedule[i] * cumulative
  cumulative += schedule[i]
temp_cost = eta * np.sum(schedule ** 2 / tau)
expected_cost = perm_cost + temp_cost
variance = sigma ** 2 * tau * np.sum(trajectory[1:] ** 2)
std_dev = np.sqrt(variance)

print(f"Expected impact cost: {expected_cost:.2f} dollars")
print(f"Cost std dev: {std_dev:.2f} dollars")
print(f"Expected cost per share: {expected_cost/Q:.4f} dollars")
print()


# Simulation with more realistic example

print("STEP 2: EXECUTION")

# Add some randomness
np.random.seed(42)
price_path = [decision_price]
execution_prices = []
execution_shares = []
actual_trajectory = [Q]

current_price = decision_price
for i in range(N):
  shares_this_slice = schedule[i]

  # Permanent impact
  current_price += gamma * shares_this_slice

  # Random noise
  current_price += sigma * np.sqrt(tau) * np.random.randn() * current_price / 100

  # Temporary impact
  fill_price = current_price + eta * (shares_this_slice / tau)

  execution_prices.append(fill_price)
  execution_shares.append(shares_this_slice)
  actual_trajectory.append(actual_trajectory[-1] - shares_this_slice)
  price_path.append(current_price)

execution_prices = np.array(execution_prices)
execution_shares = np.array(execution_shares)

# Print execution log
print(f"{'Slice':<8} {'Shares':<12} {'Price':<12} {'Cumulative':<12}")
print("-" * 44)
cum_shares = 0
for i in range(N):
  cum_shares += execution_shares[i]
  print(f"{i+1:<8} {execution_shares[i]:<12.0f} {execution_prices[i]:<12.4f} {cum_shares:<12.0f}")
print()

#After-trade analysis
print("After the trade analysis")

total_shares = np.sum(execution_shares)
vwap = np.sum(execution_prices * execution_shares) / total_shares
is_per_share = vwap - decision_price
is_total = is_per_share * total_shares

print(f"Decision price: {decision_price:.4f} dollars")
print(f"VWAP: {vwap:.4f} dollars")
print(f"Implementation shortfall: {is_per_share:.4f} dollars/share")
print(f"Total IS: {is_total:.2f} dollars")
print(f"IS vs expected: {is_total:.2f} vs {expected_cost:.2f} (expected)")
print()

#Plotting and visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Trajectory
ax1 = axes[0]
ax1.plot(t_points, trajectory, 'b-', linewidth=2, label='Planned')
ax1.plot(t_points, actual_trajectory, 'g--', linewidth=2, label='Actual')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Shares Remaining')
ax1.set_title('Execution Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Price path
ax2 = axes[1]
ax2.plot(t_points, price_path, 'k-', linewidth=2)
ax2.axhline(y=decision_price, color='r', linestyle='--', label='Decision Price')
ax2.axhline(y=vwap, color='g', linestyle='--', label='VWAP')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Price (dollars)')
ax2.set_title('Price Path During Execution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Shares per slice
ax3 = axes[2]
slice_times = t_points[:-1] + tau/2
ax3.bar(slice_times, execution_shares, width=tau*0.8, color='#26804a', alpha=0.8)
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('Shares')
ax3.set_title('Execution Schedule')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"Strategy: Almgren-Chriss with lambda={lambda_risk:.3f}")
print(f"Result: Paid {is_per_share:.4f} extra per share")
print(f"Total cost: {is_total:.2f} dollars for {int(total_shares)} shares")
if abs(is_total) < expected_cost * 1.5:
  print("Assessment: Execution within expected range")
else:
  print("Assessment: Execution cost higher than expected - review parameters")