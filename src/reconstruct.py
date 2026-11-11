from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize_scalar as minimise_scalar
from scipy.signal import savgol_filter


def detect_spike_onsets(
    y: np.ndarray,
    savgol_window: int = 501,
    savgol_poly: int = 2,
    mad_k: float = 6.0,
    refractory_samples: int = 5,
) -> np.ndarray:
    """Detect exact spike onset sample indices via first-order voltage innovations."""
    rough_drift = savgol_filter(y, window_length=savgol_window, polyorder=savgol_poly)
    detrended = y - rough_drift

    # First-order innovation highlights instantaneous single-sample depolarisations
    innovation = np.diff(detrended, prepend=detrended[0])
    robust_sigma = np.median(np.abs(innovation - np.median(innovation))) / 0.6745
    candidates = np.flatnonzero(innovation > mad_k * robust_sigma)

    # Enforce biophysical refractory dead-zone, keeping the sharpest onset jump
    spike_indices = []
    for idx in candidates:
        if not spike_indices or (idx - spike_indices[-1]) >= refractory_samples:
            spike_indices.append(int(idx))
        elif innovation[idx] > innovation[spike_indices[-1]]:
            spike_indices[-1] = int(idx)

    return np.array(spike_indices, dtype=np.int64)


def build_unit_spike_train(
    n_samples: int,
    dt: float,
    spike_indices: np.ndarray,
    tau: float,
    kernel_duration: float = 0.035,
) -> np.ndarray:
    """Construct a unit-amplitude causal exponential spike train for a given decay tau."""
    u = np.zeros(n_samples, dtype=np.float64)
    kernel_len = int(round(kernel_duration / dt))
    lag_t = np.arange(kernel_len, dtype=np.float64) * dt
    kernel = np.exp(-lag_t / tau)

    for idx in spike_indices:
        m = min(kernel_len, n_samples - idx)
        if m > 0:
            u[idx : idx + m] += kernel[:m]
    return u


def fit_harmonic_drift(t: np.ndarray, y_clean: np.ndarray, dt: float) -> tuple[float, float, float, float, np.ndarray]:
    """Fit C + A_drift * sin(2*pi*f*t + phi) via variable projection over frequency f."""
    n_samples = len(t)
    y_centred = y_clean - np.mean(y_clean)
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    fft_mag = np.abs(np.fft.rfft(y_centred))
    f_init = float(freqs[1 + np.argmax(fft_mag[1:])])

    def residual_for_freq(f_arr: np.ndarray) -> np.ndarray:
        f = float(f_arr[0])
        basis = np.column_stack([np.ones(n_samples), np.sin(2.0 * np.pi * f * t), np.cos(2.0 * np.pi * f * t)])
        beta, *_ = np.linalg.lstsq(basis, y_clean, rcond=None)
        return y_clean - basis @ beta

    opt = least_squares(residual_for_freq, x0=[f_init], bounds=([0.01], [1.0]))
    f_opt = float(opt.x[0])

    basis = np.column_stack([np.ones(n_samples), np.sin(2.0 * np.pi * f_opt * t), np.cos(2.0 * np.pi * f_opt * t)])
    offset, b_sin, b_cos = np.linalg.lstsq(basis, y_clean, rcond=None)[0]
    amp_drift = float(np.hypot(b_sin, b_cos))
    phase_drift = float(np.arctan2(b_cos, b_sin))
    drift_signal = basis @ np.array([offset, b_sin, b_cos])

    return float(offset), amp_drift, f_opt, phase_drift, drift_signal


def standardise_and_decompose_recording(
    input_path: str = "data/input/spike_data.npz",
    output_path: str = "data/output/team_10_reconstruction.npz",
    n_alternating_iters: int = 8,
    seed: int = 42,
) -> dict:
    """Run full decomposition and save the standardised reconstruction archive."""
    with np.load(input_path) as data:
        t = np.asarray(data["t"], dtype=np.float64).squeeze()
        raw_data = np.asarray(data["raw_data"], dtype=np.float64).squeeze()

    n_samples = len(t)
    dt = float(np.median(np.diff(t)))
    duration = n_samples * dt

    # Step 1: Detect spike starting indices
    spike_indices = detect_spike_onsets(raw_data)

    # Step 2: Alternating Variable Projection for Drift & Spike Waveform
    drift_est = savgol_filter(raw_data, window_length=501, polyorder=2)
    amp_spike = 0.0
    tau_spike = 0.003
    spike_signal = np.zeros(n_samples, dtype=np.float64)

    for _ in range(n_alternating_iters):
        y_no_drift = raw_data - drift_est

        def profile_ssr(tau: float) -> float:
            u = build_unit_spike_train(n_samples, dt, spike_indices, tau)
            a = float(np.dot(y_no_drift, u) / np.dot(u, u))
            res = y_no_drift - a * u
            return float(np.dot(res, res))

        opt_tau = minimise_scalar(profile_ssr, bounds=(0.0005, 0.015), method="bounded")
        tau_spike = float(opt_tau.x)
        unit_train = build_unit_spike_train(n_samples, dt, spike_indices, tau_spike)
        amp_spike = float(np.dot(y_no_drift, unit_train) / np.dot(unit_train, unit_train))
        spike_signal = amp_spike * unit_train

        offset_drift, amp_drift, freq_drift, phase_drift, drift_est = fit_harmonic_drift(
            t, raw_data - spike_signal, dt
        )

    # Step 3: Structured reconstruction (drift + spikes) and noise estimation
    reconstruction = drift_est + spike_signal
    empirical_residual = raw_data - reconstruction
    sigma_noise = float(np.std(empirical_residual, ddof=1))

    # Synthesise standardised noise realisation with seed = 42
    np.random.seed(seed)
    noise = np.random.normal(loc=0.0, scale=sigma_noise, size=n_samples).astype(np.float64)

    # Save standardised submission archive
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        t=t.astype(np.float64),
        drift=drift_est.astype(np.float64),
        spike_signal=spike_signal.astype(np.float64),
        noise=noise.astype(np.float64),
        reconstruction=reconstruction.astype(np.float64),
        spike_indices=spike_indices.astype(np.int64),
    )

    r2 = float(1.0 - np.var(empirical_residual) / np.var(raw_data))
    rmse = float(np.sqrt(np.mean(empirical_residual**2)))

    return {
        "n_spikes": int(len(spike_indices)),
        "firing_rate_hz": float(len(spike_indices) / duration),
        "drift_offset": offset_drift,
        "drift_amp": amp_drift,
        "drift_freq_hz": freq_drift,
        "drift_phase_rad": phase_drift,
        "spike_amp": amp_spike,
        "spike_tau_ms": tau_spike * 1000.0,
        "noise_sigma": sigma_noise,
        "residual_mean": float(np.mean(empirical_residual)),
        "rmse": rmse,
        "r2": r2,
    }


decompose_recording = standardise_and_decompose_recording


if __name__ == "__main__":
    summary = standardise_and_decompose_recording()
    print("=== Signal Reconstruction Summary ===")
    for k, v in summary.items():
        print(f"  {k:<18}: {v:.6f}" if isinstance(v, float) else f"  {k:<18}: {v}")
    print("Saved standardised archive to: data/output/team_10_reconstruction.npz")
