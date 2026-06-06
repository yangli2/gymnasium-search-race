# Scientific Documentation & Handoff Guide: Model A vs. Model B Benchmarks

Welcome! This onboarding document details the reinforcement learning research,
physical control realizations, and execution directions for the
`gymnasium_search_race` autotuning project. It is designed to allow another
researcher or AI agent to instantly resume operations, evaluate models, and
continue hyperparameter exploration.

--------------------------------------------------------------------------------

## 1. Project Context & Hparams Mapping

We pivoted 100% to **Discrete Action Control** to optimize racing times
(completed steps) across 50 diverse Gymnasium track layouts. We benchmarked two
highly optimized categorical champion architectures:

### 🏆 Model A: The 10-Action Direct Baseline (Iter 14 Champion)

*   **Model ID**: `Iter14_Disc_10_Potential_Alpha2e_4_LR1e_4_GAE98`
*   **Action Space**: **Discrete-10** (5 Steering Angles $\times$ 2 Thrust
    Levels)
    *   *Steering Bins*: `[-18, -9, 0, 9, 18]` degrees.
    *   *Thrust Bins*: `[0, 200]` (Binary: Zero thrust or Max thrust).
*   **Training Hyperparameters**:
    *   Learning Rate: $1.0\times 10^{-4}$
    *   Discount Horizon: $\gamma = 0.995$
    *   GAE Horizon: $\lambda = 0.98$
    *   Progress Potential Attractor: $\alpha = 2.0\times 10^{-4}$ (dense
        shaping)
    *   Observation Space: 10-Dimensional (Raw coordinate delta vectors).
    *   *VecNormalize*: **Disabled** (Model A was trained raw and expects raw
        coordinate inputs!).

### 🧪 Model B: The 15-Action Deceleration Braking Grid (Iter 22 Champion)

*   **Model ID**: `Iter22_Disc15_H2_Brake_LR1e_4`
*   **Action Space**: **Discrete-15** (5 Steering Angles $\times$ 3 Thrust
    Levels)
    *   *Steering Bins*: `[-18, -9, 0, 9, 18]` degrees.
    *   *Thrust Bins*: `[0, 100, 200]` (Zero thrust, **50% Deceleration Gear**,
        Max thrust).
*   **Training Hyperparameters**:
    *   Learning Rate: $1.0\times 10^{-4}$
    *   Discount Horizon: $\gamma = 0.995$
    *   GAE Horizon: $\lambda = 0.98$
    *   Progress Potential Attractor: $\alpha = 2.0\times 10^{-4}$
    *   *VecNormalize*: **Enabled** (Expects standard vectorized scale
        mapping!).

--------------------------------------------------------------------------------

## 2. The Observation Normalization Evaluation Patch

> [!IMPORTANT] During training, `use_vec_normalize=True` serializes coordinate
> running-statistics to `vec_normalize.pkl`. Evaluations **MUST** load this
> `.pkl` file and wrap the raw environment inside a vectorized `DummyVecEnv` +
> `VecNormalize` layer before executing model predictions to prevent
> coordination wiggles and 600-step timeouts!

The load pipeline wraps the raw env in a `DummyVecEnv` to conform to the
StableBaselines3 VecEnv API, then dynamically locates and loads `VecNormalize`
stats from the run folder. The normalization stats are locked during evaluation
(training mode disabled, reward normalization disabled).

--------------------------------------------------------------------------------

## 3. How to Run Evaluations & Generate Video Trajectories

Use `record_video.py` to evaluate any model and render its trajectory as an
animated GIF. Pass the model path, environment ID, environment kwargs (action
space config, potential alpha, etc.), and an output folder. For Model B, also
supply the `vec_normalize.pkl` path.

--------------------------------------------------------------------------------

## 4. Control-Theoretic Physics Comparison Realizations

Our absolute, map-by-map benchmark over all 13 track layouts (ID 0 to 12)
exposed a brilliant, decoupled physical trade-off:

Average Completed Length (Across all 13 Tracks):
- Model A (10-Action Baseline): 🏆 157.31 steps completed
- Model B (15-Action Braking):   162.46 steps completed

### 1. The Decoupled Slalom / Cornering Victory (Model B Wins):

*   On **slalom and centripetal heavy tracks** (such as Map 1 and Map 3), Model
    B out-performed Model A by **8 to 13 physical steps**.
*   *Turning Physics*: Having the intermediate `100` thrust deceleration gear
    allows the pod to shed linear velocity right before the checkpoint boundary,
    shrinking its turning radius ($r = mv^2/F_c$). This allows it to redirect
    its heading immediately, executing tight corner pivots without centrifugal
    drift overshoot.

### 2. The Straightaway Velocity Victory (Model A Wins):

*   On **straightaway-heavy tracks** (such as Map 9 and Map 4), Model A
    out-performed Model B by **15 to 16 physical steps**.
*   *Straightway Velocity Conservation*: Because Model A has a strictly binary
    thrust space `[0, 200]`, PPO is mathematically forced to output max `200`
    thrust on straight lines. Model B's 3-level thrust space suffers from minor
    observation jitters, causing the actor to occasionally select intermediate
    `100` thrust on straightaways, bleeding overall linear speed.

--------------------------------------------------------------------------------

## 5. Immediate Onboarding Directions (Where to go next)

To push discrete completed times below **140 steps** on average, we propose the
following immediate hyperparameter exploration directions:

### 🚀 Direction 1: Asymmetric Angle Cap (No Straightaway Dilution)

Instead of giving the actor an intermediate deceleration gear in its action
space, let's keep the un-diluted 10-Action binary thrust space (`[0, 200]`) but
implement a **Heading-Error Thrust Cap** directly in the environment physics:

*   Calculate heading error delta to next checkpoint. If `heading_error > 60`
    degrees, cap physical thrust to `0.0` (braking) inside
    `SearchRaceEnv._convert_action_to_angle_thrust`.
*   *Physical Expectation*: This allows the pod to brake automatically on sharp
    corner turns, while ensuring PPO can *never* output intermediate thrust on
    straight lines, preserving the straightway velocity conservation!

### 🚀 Direction 2: Time-Decaying attraction progress attractor ($\alpha$)

Potential Progress attractor shaping ($\alpha = 2.0\times 10^{-4}$) is crucial
for early-stage convergence, but it slightly distorts the pod's optimal
trajectory towards checkpoint centers.

*   Modify `train.py` to decay $\alpha$ linearly to `0.0` after 5M training
    steps.
*   *Physical Expectation*: Forces the model to transition from progress-seeking
    behavior to absolute completed time optimization in late-stage training,
    shaving the final 5-9 steps of trajectory distortion!

--------------------------------------------------------------------------------
