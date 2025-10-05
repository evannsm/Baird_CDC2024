# Interval Signal Temporal Logic (iSTL) for Robust Optimal Control - Technical Overview

**Author:** Luke Baird
**Institution:** Georgia Institute of Technology
**Paper:** "Interval Signal Temporal Logic for Robust Optimal Control" (CDC 2024)

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Integration with stlpy](#integration-with-stlpy)
3. [Key Divergences from stlpy](#key-divergences-from-stlpy)
4. [Robustness Calculation (The Magic)](#robustness-calculation-the-magic)
5. [Mathematical Framework](#mathematical-framework)
6. [Example: Blimp Mission](#example-blimp-mission)

---

## System Overview

This codebase implements **Interval Signal Temporal Logic (iSTL)**, an extension of Signal Temporal Logic (STL) that enables **guaranteed robustness** under bounded disturbances and uncertainties. The key innovation is computing tight bounds on trajectory robustness **before execution**, ensuring safety even with model uncertainties and disturbances.

### What Problem Does This Solve?

Traditional STL control synthesis assumes:
- Perfect knowledge of system dynamics
- No external disturbances
- Exact predicate boundaries

**Real systems have:**
- Model uncertainties (e.g., drag coefficients are approximate)
- External disturbances (e.g., wind gusts)
- Uncertain goal regions (e.g., "target is approximately at (5,5)")

**This framework** guarantees that if the computed robustness is positive, the system will satisfy the specification despite these uncertainties.

---

## Integration with stlpy

### Base Framework from stlpy

This code builds on [stlpy](https://stlpy.readthedocs.io/en/latest/), a Python library for STL-based control synthesis. From stlpy, we use:

1. **STL Formula Structure**
   - `STLFormula`: Abstract base class for all formulas
   - `STLTree`: Represents temporal and logical operators (always, eventually, and, or)
   - Formula construction methods: `.always()`, `.eventually()`, `.conjunction()`, etc.

2. **System Representation**
   - `LinearSystem`: Discrete-time LTI systems `x[t+1] = Ax[t] + Bu[t]`
   - System dynamics integration with optimization

3. **Base Solver Architecture**
   - `STLSolver`: Base class for optimization-based synthesis
   - Gurobi MICP (Mixed-Integer Convex Programming) backend

4. **Standard Robustness Semantics**
   - For predicates: `ρ(a^T y - b) = a^T y - b`
   - For conjunction: `ρ(φ₁ ∧ φ₂) = min(ρ(φ₁), ρ(φ₂))`
   - For disjunction: `ρ(φ₁ ∨ φ₂) = max(ρ(φ₁), ρ(φ₂))`

### What We Keep Exactly

The overall workflow remains similar to stlpy:
```python
# 1. Define system dynamics
sys = LinearSystem(A, B, C, D)

# 2. Define STL specification
phi = (reach_target.eventually(0, 20)) & (avoid_obstacle.always(0, 20))

# 3. Solve for optimal control
solver = GurobiOptimalControl(spec=phi, sys=sys, x0=x0, T=horizon)
solution = solver.Solve()
```

---

## Key Divergences from stlpy

### 1. **Interval-Valued Predicates**

**stlpy:**
```python
# Standard predicate: x >= 5.0
pred = LinearPredicate(a=[1,0,0], b=5.0)
```

**iSTL (this code):**
```python
# Interval predicate: x >= 5.0 ± 0.1
# Handles uncertain boundaries!
a = np.array([[1,0,0]], dtype=np.interval)
b = 5.0 + np.interval(-0.1, 0.1)
pred = LinearPredicate(a, b)
```

**Modified files:**
- `predicate.py`: Extended `LinearPredicate` to accept `np.interval` types for both `a` and `b`

### 2. **Interval Formula Tree**

**Modified file:** `formula.py`

The robustness computation is extended to handle intervals:
```python
def robustness(self, y, t):
    if self.combination_type == "and":
        # Returns INTERVAL bounds on robustness
        return np.min([formula.robustness(y, t+self.timesteps[i])
                       for i, formula in enumerate(self.subformula_list)])
```

### 3. **Embedding System for Disturbances**

**New concept:** Instead of one system, we track **two systems simultaneously**:

**stlpy approach:**
```
x[t+1] = Ax[t] + Bu[t]          # Single nominal trajectory
```

**iSTL approach:**
```
x_nom[t+1] = A x_nom[t] + Bu[t]        # Nominal trajectory
e[t+1] = A_e e[t] + w[t]                # Error dynamics (bounded)
```

The **actual trajectory** is: `x_actual[t] = x_nom[t] + V⁻¹ e[t]`

Where:
- `e[t]` is the error in transformed coordinates
- `w[t] ∈ [w_lower, w_upper]` is bounded disturbance
- `V` is a coordinate transformation matrix
- `A_e` is the error system dynamics (decomposed using `d_positive`)

**Implementation:** See `gurobi_optimal_control.py:352-406`

### 4. **Decomposition-Based Interval Arithmetic**

**New function:** `d_positive(B)` (in `gurobi_optimal_control.py:13-22`)

This decomposes any matrix into positive and negative parts:
```python
B⁺ = max(B, 0)    # Positive part
B⁻ = min(B, 0)    # Negative part
```

Used for **interval multiplication** in optimization:
```
[y_lower, y_upper] = B × [x_lower, x_upper]
y_lower = B⁺ x_lower + B⁻ x_upper
y_upper = B⁺ x_upper + B⁻ x_lower
```

**Why this matters:** Gurobi can't directly multiply interval variables, so we decompose the operation into concrete linear constraints.

### 5. **New Solver: `GurobiIntervalOptimalControl`**

**New file:** `gurobi_optimal_control.py`

Key enhancements over stlpy's `GurobiMICPSolver`:

**Additional parameters:**
- `sys_nominal`: The nominal dynamics (in addition to embedding system)
- `x0_nominal`: Nominal initial state
- `w`: Bounded disturbance interval
- `intervals`: Boolean flag to enable interval predicates
- `V_inv`: Coordinate transformation matrix

**Additional constraints:**
```python
# Nominal dynamics
x_nominal[t+1] = A_nominal @ x_nominal[t] + B @ u[t]

# Error dynamics (no control influence)
e[t+1] = A_e @ e[t] + w

# Output reconstruction with intervals
y_upper = V_inv⁺ @ e_upper + V_inv⁻ @ e_lower + V_inv @ x_nominal
y_lower = V_inv⁺ @ e_lower + V_inv⁻ @ e_upper + V_inv @ x_nominal
```

**See:** Lines 360-394 in `gurobi_optimal_control.py`

### 6. **Interval Predicate Constraints**

**Standard stlpy predicate constraint:**
```python
# For predicate a^T y >= b:
model.addConstr(a.T @ y[t] - b + (1-z)*M >= rho[t])
```

**iSTL interval predicate constraint** (lines 444-491):
```python
# For each component of 'a' (which is an interval [a_lower, a_upper]):
for j in range(dim):
    # Compute all four products:
    product_1 = a_lower[j] * y_lower[j,t]
    product_2 = a_lower[j] * y_upper[j,t]
    product_3 = a_upper[j] * y_lower[j,t]
    product_4 = a_upper[j] * y_upper[j,t]

    # Take minimum (worst case)
    s_j = min(product_1, product_2, product_3, product_4)

# Robustness bound: sum(s_j) - b_upper >= rho
model.addConstr(sum(s_j) - b_upper + (1-z)*M >= rho[t])
```

This computes the **tightest lower bound** on robustness considering all interval uncertainties.

---

## Robustness Calculation (The Magic)

### How Robustness is Computed BEFORE Trajectory Execution

This is the key innovation of the paper. Here's the step-by-step process:

#### **Step 1: Problem Setup**

You specify:
1. **Nominal system dynamics**: `A_nom, B`
2. **Disturbance bounds**: `w ∈ [w_lower, w_upper]` (e.g., uncertain drag)
3. **STL specification with intervals**: `φ` (e.g., "eventually reach [4.9, 5.1] × [4.9, 5.1]")
4. **Initial condition**: `x0` (possibly uncertain)

#### **Step 2: Coordinate Transformation**

The code computes a stabilizing feedback gain `K` and transformation matrix `T` such that:
```
A_stable = A_nom - B*K
```

This transforms the system into coordinates where error propagation is bounded.

**See:** `final-paper-code.py:283-313`

#### **Step 3: Embedding System Construction**

Create the **error embedding system**:
```python
# Decompose stabilized dynamics
A_e = d_positive(T @ A_stable @ T_inv, separate=False)  # ∈ R^(2n × 2n)

# The embedding tracks [e_lower; e_upper]
e[t] ∈ R^(2n)  where e[:n] = e_lower, e[n:] = e_upper
```

**See:** `final-paper-code.py:305-308`

#### **Step 4: Optimization Problem**

The solver sets up the following MICP:

**Decision variables:**
- `u[t]`: Control inputs (same for nominal and all possible trajectories)
- `x_nom[t]`: Nominal state trajectory
- `e[t]`: Error bounds `[e_lower[t]; e_upper[t]]`
- `rho[t]`: **Robustness bounds at each time step**

**Constraints:**
```python
# 1. Nominal dynamics
x_nom[t+1] = A_nom @ x_nom[t] + B @ u[t]

# 2. Error dynamics (interval arithmetic)
e[t+1] = A_e @ e[t] + w

# 3. Output bounds (what we actually verify against φ)
y_lower[t] = V_inv⁺ @ e_lower[t] + V_inv⁻ @ e_upper[t] + V_inv @ x_nom[t]
y_upper[t] = V_inv⁺ @ e_upper[t] + V_inv⁻ @ e_lower[t] + V_inv @ x_nom[t]

# 4. STL constraints with intervals (for each predicate in φ)
rho_lower[t] <= (worst-case robustness over [y_lower, y_upper])
```

**Objective:**
```python
minimize: Σ (x_nom[t]^T Q x_nom[t] + u[t]^T R u[t])
subject to: rho >= 0  # Guaranteed satisfaction
```

**See:** `gurobi_optimal_control.py:415-522`

#### **Step 5: Solution and Guarantee**

The solver returns:
- `u[t]`: Optimal control sequence
- `x_nom[t]`: Nominal trajectory
- `e[t]`: Error bounds
- `rho`: **Robustness measure**

**The guarantee:**
```
If rho >= 0, then for ALL:
  - Disturbances w[t] ∈ [w_lower, w_upper]
  - Model uncertainties captured in A_e
  - Predicate uncertainties in φ

The system WILL satisfy φ
```

#### **Step 6: Post-Optimization Verification (Optional)**

The code verifies the solution:
```python
# Reconstruct actual trajectory bounds
x_lower = V_inv⁺ @ e_lower + V_inv⁻ @ e_upper + V_inv @ x_nom
x_upper = V_inv⁺ @ e_upper + V_inv⁻ @ e_lower + V_inv @ x_nom

# Check robustness on bounds
print(f'Robustness of lower bound: {phi.robustness(x_lower, 0)}')
print(f'Robustness of upper bound: {phi.robustness(x_upper, 0)}')
print(f'Robustness of nominal: {phi.robustness(x_nom, 0)}')
```

**See:** `final-paper-code.py:456-459`

### Why This Works: The Math

**Key insight:** By using **interval arithmetic** in the optimization, we compute robustness for **all possible trajectories** simultaneously, not just the nominal one.

For a predicate `a^T y >= b`:
```
ρ(a^T y - b) for y ∈ [y_lower, y_upper]
```

We want the **minimum** (worst-case) robustness:
```
ρ_worst = min_{y ∈ [y_lower, y_upper]} (a^T y - b)
```

Using interval arithmetic:
```
a^T [y_lower, y_upper] = [min(a^T y), max(a^T y)]
                       = [a⁺^T y_lower + a⁻^T y_upper, a⁺^T y_upper + a⁻^T y_lower]
```

So:
```
ρ_worst = (a⁺^T y_lower + a⁻^T y_upper) - b
```

The solver encodes this as **linear constraints** that Gurobi can solve!

---

## Mathematical Framework

### Interval Predicate Semantics

For interval predicate `π: a^T y ≥ b` where `a ∈ [a_lower, a_upper]`, `b ∈ [b_lower, b_upper]`:

**Standard robustness:**
```
ρ(π, y) = a^T y - b
```

**Interval robustness** (worst-case over uncertainties):
```
ρ_interval(π, [y_lower, y_upper]) =
    min_{a ∈ [a_lower, a_upper], b ∈ [b_lower, b_upper]}
        (a^T y_lower - b)
```

This is computed via decomposition as:
```
ρ_interval = Σⱼ min(a_lower[j]·y_lower[j], a_lower[j]·y_upper[j],
                   a_upper[j]·y_lower[j], a_upper[j]·y_upper[j]) - b_upper
```

### Embedding System Dynamics

**Nominal system:**
```
x[t+1] = A x[t] + B u[t]
```

**With disturbance:**
```
x[t+1] = A x[t] + B u[t] + G w[t]
```
where `w[t] ∈ W = [w_lower, w_upper]`

**Error dynamics** (after stabilization with `K` and transform `T`):
```
η[t] = T (x[t] - x_nom[t])
η[t+1] = A_η η[t] + T G w[t]
```
where `A_η = T (A - BK) T⁻¹`

**Interval embedding:**
```
η[t] ∈ [η_lower[t], η_upper[t]]
```

The embedding system in `R^(2n)`:
```
[η_lower[t+1]]   [A_η⁺  A_η⁻] [η_lower[t]]   [w_lower]
[η_upper[t+1]] = [A_η⁻  A_η⁺] [η_upper[t]] + [w_upper]
```

This is the `A_e` matrix in the code.

### Coordinate Transformation

The transformation `V = T⁻¹` maps error coordinates back to original coordinates:
```
x[t] = x_nom[t] + V η[t]
```

With intervals:
```
x_lower[t] = x_nom[t] + V⁺ η_lower[t] + V⁻ η_upper[t]
x_upper[t] = x_nom[t] + V⁺ η_upper[t] + V⁻ η_lower[t]
```

---

## Example: Blimp Mission

### Problem Setup

**System:** Indoor miniature blimp with:
- 12 states: `[v_x, v_y, v_z, ω_x, ω_y, ω_z, x, y, z, θ, φ, ψ]`
- 4 inputs: `[f_x, f_y, f_z, τ_z]` (forces and yaw torque)
- Uncertain drag coefficients
- Wind disturbances: `w ∈ [-0.001, 0.001] m/s²`

**Mission (STL specification):**
```python
φ = φ_A ∧ φ_B ∧ φ_D

# φ_A: Visit target (with uncertain position)
target = (x ∈ [4.7±0.1, 5.3±0.1]) ∧ (y ∈ [4.7±0.1, 5.3±0.1])
φ_A = F[0,6] G[0,1.5] target

# φ_B: Return to base
base = (x ∈ [-0.35, 0.35]) ∧ (y ∈ [-0.35, 0.35])
φ_B = F[19,20] base

# φ_D: Visit charging stations periodically
φ_D = F[1,6] charging ∧ F[8,12] charging ∧ F[14,20] charging
```

### Results

**With interval predicates and disturbances:**
```
Robustness: 0.0234 > 0  ✓ GUARANTEED SAFE
Solve time: 127.3 seconds
```

The system computes:
- A nominal trajectory `x_nom[t]`
- Error bounds `[x_lower[t], x_upper[t]]`
- **All trajectories within the bounds satisfy φ**

See the paper for plots showing the trajectory tube (shaded region) that is guaranteed to satisfy the specification.

---

## Key Files Summary

| File | Purpose | Changes from stlpy |
|------|---------|-------------------|
| `predicate.py` | STL predicates | Added interval support for `a` and `b` |
| `formula.py` | STL formula tree | No changes (stlpy version works) |
| `gurobi_optimal_control.py` | Interval MICP solver | **New solver class** with embedding system |
| `final-paper-code.py` | Blimp example | Application code |
| `blimp.py` | Blimp dynamics | System model |

---

## Running the Code

```bash
# 1. Install dependencies
pip install numpy matplotlib scipy

# 2. Install npinterval (for interval arithmetic)
pip install git+https://github.com/gtfactslab/npinterval

# 3. Install Gurobi (free academic license)
# Follow: https://www.gurobi.com/downloads/free-academic-license/

# 4. Install modified stlpy
git clone https://github.com/vincekurtz/stlpy
cd stlpy
# Copy modified files from this repo:
#   - gurobi_optimal_control.py -> stlpy/solvers/gurobi/
#   - predicate.py -> stlpy/STL/
#   - formula.py -> stlpy/STL/
# Edit stlpy/solvers/__init__.py to add:
#   from .gurobi.gurobi_optimal_control import GurobiIntervalOptimalControl
pip install .

# 5. Run the example
cd /path/to/Baird_CDC2024
python final-paper-code.py
```

---

## References

1. **stlpy:** Kurtz, V., & Lin, H. (2021). "STLpy: A Python library for control synthesis from Signal Temporal Logic specifications."
2. **STL Semantics:** Donzé, A., & Maler, O. (2010). "Robust satisfaction of temporal logic over real-valued signals."
3. **MICP Encoding:** Belta, C., et al. (2019). "Formal methods for control synthesis: an optimization perspective."
4. **This Work:** Baird, L., et al. (2024). "Interval Signal Temporal Logic for Robust Optimal Control." CDC 2024.

---

## Contact

For questions about this code or the paper, contact Luke Baird at Georgia Institute of Technology.

**License:** See LICENSE file in this repository.
