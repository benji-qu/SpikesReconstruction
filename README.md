# Neural Spike Recording — Reverse-Engineering & Evaluation

## Repository Structure

```text
├── data/
│   ├── input/
│   │   └── spike_data.npz          # Reference extracellular recording (10,000 samples, 1 kHz)
│   └── output/
│       └── team_10_reconstruction.npz  # Standardised 6-array signal decomposition archive
├── src/
│   ├── __init__.py
│   └── reconstruct.py              # Signal detection, variable projection, and decomposition pipeline
├── eval/
│   ├── __init__.py
│   └── evaluate_team.py            # First-principles inter-team evaluation module & CLI
├── notebooks/
│   ├── reconstruct.ipynb           # Step-by-step reverse-engineering & decomposition notebook
│   └── test_reconstruct.ipynb      # Interactive verification and evaluator diagnostic notebook
├── evaluate_team.py                # Root evaluator entry-point (re-exports eval/evaluate_team.py)
├── requirements.txt                # Locked Python package dependencies
└── README.md                       # Model specification, assumptions, and methodology
```

> **Reconstruction Archive Download:** The generated reconstruction file (`team_10_reconstruction.npz`) can be downloaded here:  
> **[Download](https://drive.google.com/file/d/1adbsaYpGTRSjllL370LSOstXNn86pZIc/view?usp=sharing)** `team_10_reconstruction.npz` **[(Google Drive)](https://drive.google.com/file/d/1adbsaYpGTRSjllL370LSOstXNn86pZIc/view?usp=sharing)** _(place in_ `data/output/team_10_reconstruction.npz` _or regenerate via_ `python -m src.reconstruct`_)_.

---

## 1. Inferred Model and Parameters

The recorded extracellular signal `y(t)` (`raw_data`, sampled at `fs = 1000 Hz` over `T = 10.0 s`, `N = 10,000` samples) is modelled as the additive superposition of three components:

```text
y(t) = drift(t) + spike_signal(t) + ε(t)
```

1. **Slow Baseline Drift (**`drift`**):** A harmonic oscillation with constant DC offset:
   ```text
    drift(t) = C + A_drift · sin(2π · f_drift · t + φ_drift)
   ```
2. **Action Potential Waveform Train (**`spike_signal`**):** A sum of causal, single-sample-onset exponentially decaying spikes triggered at discrete onset indices `n_k` (`t_k = n_k / fs`, for `k = 1, ..., K`):

   ```text
    spike_signal(t) = Σ_{k=1..K} A_spike · exp(-(t - t_k) / τ) · I(t ≥ t_k)
   ```

   where event arrivals follow a homogeneous Poisson process with rate `λ` and a short absolute refractory dead-zone (`τ_ref ≈ 5 ms`).

3. **Background Noise (**`noise`**):** Zero-mean additive white Gaussian noise (AWGN):
   ```text
    ε(t) ~ N(0, σ²)   (independent and identically distributed)
   ```

### Estimated Parameters from `spike_data.npz`

| Component       | Parameter                   |        Symbol         |         Estimated Value          |
| :-------------- | :-------------------------- | :-------------------: | :------------------------------: |
| **Drift**       | DC Offset                   |          `C`          |           `+0.0031 V`            |
| **Drift**       | Sinusoidal Amplitude        |       `A_drift`       |            `0.6466 V`            |
| **Drift**       | Frequency (Period)          | `f_drift` (`T_drift`) |     `0.0908 Hz` (`11.02 s`)      |
| **Drift**       | Initial Phase               |       `φ_drift`       |           `0.2923 rad`           |
| **Spikes**      | Detected Event Count        |          `K`          |           `42 spikes`            |
| **Spikes**      | Mean Firing Rate            |          `λ`          |            `4.20 Hz`             |
| **Spikes**      | Peak Onset Jump             |       `A_spike`       |            `5.0164 V`            |
| **Spikes**      | Exponential Decay Constant  |          `τ`          |           `2.9828 ms`            |
| **Noise**       | Residual Standard Deviation |          `σ`          |            `0.2984 V`            |
| **Fit Quality** | Structured Signal R² / RMSE |     `R²` / `RMSE`     | `0.8349` (`83.49%`) / `0.2984 V` |

---

## 2. Assumptions

1. **Additive Linear Superposition:** The baseline drift, action potential train, and background thermal/electronic noise sum linearly without non-linear saturation.
2. **Causal Instantaneous Depolarisation & AR(1) Exponential Decay:** Each spike jumps upwards by `A_spike` within a single `1 ms` sample at its onset index `n_k` and decays exponentially with time constant `τ` (`~3 ms`), satisfying the discrete-time AR(1) recurrence `s[n] - exp(-Δt / τ) · s[n - 1] = Σ_k A_spike · δ[n - n_k]`.
3. **Stereotyped Single-Unit Amplitude:** Individual spike-to-spike peak variations around the fitted exponential kernel are accounted for by the additive background noise (`σ ≈ 0.2984 V`), so a constant unit amplitude `A_spike` prevents overfitting noise on individual peaks.
4. **Stationary White Gaussian Noise:** Background fluctuations are temporally uncorrelated (`lag-1 ACF ≈ -0.020`, within the 95% white-noise confidence band `±1.96 / √N = ±0.0196`) and normally distributed (`N(0, σ²)`).
5. **Structured Signal Reconstruction:** In accordance with the specification (_"The reconstruction should contain the estimated structured signal (drift + spikes), not a newly generated noise realization"_), `reconstruction = drift + spike_signal`, whilst `noise` stores a white Gaussian noise realisation synthesised with `seed = 42`.

---

## 3. Detection and Reconstruction Method

The decomposition pipeline (`src/reconstruct.py`) separates the signal components in three stages without discarding data via ad-hoc spike-masking windows:

1. **Stage 1 — Innovation-Based Spike Onset Detection (**`detect_spike_onsets`**):**
   - A rough initial baseline is estimated via a 501-sample quadratic Savitzky–Golay filter (`savgol_filter`) and subtracted to obtain the detrended signal `y_tilde[n]`.
   - The first-order sample-to-sample innovation `Δy_tilde[n] = y_tilde[n] - y_tilde[n - 1]` is computed, and its robust noise scale is estimated via the Median Absolute Deviation (`σ_Δ = MAD(Δy_tilde) / 0.6745 ≈ 0.422 V`).
   - Because action potential onsets produce an instantaneous `+5.0 V` jump (`~12 · σ_Δ`), thresholding at `Δy_tilde[n] > 6 · σ_Δ` (`2.53 V`) with a `5 ms` refractory dead-zone isolates all `42` exact spike starting indices (`spike_indices`).
2. **Stage 2 — Alternating Variable Projection (**`standardise_and_decompose_recording`**):**
   - Rather than masking out windows around spikes (which discards 8–10% of the recording), the estimation alternates between updating `spike_signal` and `drift` across the full 10,000-sample trace for 8 iterations:
     - **Spike Update (Profile Least Squares):** Given `y_no_drift = raw_data - drift`, for any candidate decay constant `τ`, the unit spike train `u_τ[n] = Σ_{k=1..42} exp(-(t[n] - t[n_k]) / τ) · I(n ≥ n_k)` admits a closed-form optimal amplitude:

       ```text
       A(τ) = <y_no_drift, u_τ> / ||u_τ||²
       ```

       The profile residual sum of squares is minimised over `τ ∈ [0.5 ms, 15 ms]` via bounded Brent optimisation (`minimise_scalar`).

     - **Drift Update (Harmonic Variable Projection):** Given `y_clean = raw_data - spike_signal`, `f_drift` is initialised from the FFT peak and refined via non-linear least squares whilst projecting linearly onto the harmonic basis `[1, sin(2πft), cos(2πft)]` to recover `C`, `A_drift`, and `φ_drift`.

3. **Stage 3 — Residual Noise Estimation & Standardised Export:**
   - The structured signal is assembled as `reconstruction = drift + spike_signal`, and the empirical residual `r = raw_data - reconstruction` yields `σ = std(r) = 0.2984 V`.
   - Setting `np.random.seed(42)`, a Gaussian realisation `noise = np.random.normal(0.0, sigma_noise, 10000)` is generated, and all 6 required arrays (`t`, `drift`, `spike_signal`, `noise`, `reconstruction`, `spike_indices`) are written to `data/output/team_10_reconstruction.npz`.

---

## 4. Usage and Execution

### 1. Environment Setup

Create and activate a virtual environment, then install the required dependencies from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Signal Decomposition & Reconstruction

Run the reconstruction module to decompose `data/input/spike_data.npz` and export `data/output/team_10_reconstruction.npz`:

```bash
python -m src.reconstruct
```

### 3. Inter-Team Evaluation (CLI & Python Interface)

Evaluate the default reconstruction archive or one or more candidate `.npz` files from the command line:

```bash
# Evaluate the default output archive (data/output/team_10_reconstruction.npz)
python -m eval.evaluate_team

# Evaluate one or more candidate archives against the reference recording
python -m eval.evaluate_team data/input/spike_data.npz data/output/team_10_reconstruction.npz
```

This generates per-candidate JSON reports, 4-panel diagnostic overlay figures (`*_overlay.png`), and `leaderboard_summary.csv` inside `data/output/evaluation_results/`.

To call the evaluator programmatically in Python:

```python
from eval.evaluate_team import evaluate

result = evaluate(
    reference_file="data/input/spike_data.npz",
    candidate_file="data/output/team_10_reconstruction.npz",
)
print("Final Score:", result["final score"])
```

### 4. Interactive Notebooks

Open the notebooks under `notebooks/` using the `.venv` Python kernel:

- `notebooks/reconstruct.ipynb`: Step-by-step walkthrough of exploratory inspection, spike detection, alternating variable projection, and residual diagnostics.
- `notebooks/test_reconstruct.ipynb`: Verification suite testing schema integrity, reconstruction accuracy, and evaluator sensitivity across synthetic control perturbations.
