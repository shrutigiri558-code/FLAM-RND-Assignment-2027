import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree


# 1. Read ing the  CSV file
data = pd.read_csv("xy_data.csv")

observed = data[["x", "y"]].to_numpy()

# 2. Generate the curve

def generate_curve(theta, M, X, n=1500):
    theta = np.deg2rad(theta)

    t = np.linspace(6, 60, n)

    term = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = (
        t * np.cos(theta)
        - term * np.sin(theta)
        + X
    )
    y = (
        42
        + t * np.sin(theta)
        + term * np.cos(theta)
    )

    return np.column_stack((x, y))

# 3. Calculate L1 error

def calculate_l1_error(params):

    theta, M, X = params

    predicted = generate_curve(theta, M, X)

    # Build KD-tree for predicted curve
    tree = cKDTree(predicted)

    # Find closest predicted point using L1 distance
    distances, _ = tree.query(
        observed,
        p=1
    )

    # Total L1 error
    return np.sum(distances)

# 4. Parameter limits from assignment

bounds = [
    (0, 50),       # theta
    (-0.05, 0.05), # M
    (0, 100)       # X
]

# 5. Find the best parameters-

result = differential_evolution(
    calculate_l1_error,
    bounds,
    seed=42,
    popsize=15,
    maxiter=300,
    tol=1e-8,
    polish=True,
    workers=1
)

# 6. Display final result

theta, M, X = result.x

print()
print("Final Parameters")
print("----------------")
print(f"Theta = {theta:.6f} degrees")
print(f"M     = {M:.8f}")
print(f"X     = {X:.6f}")
print()
print(f"L1 Error = {result.fun:.6f}")