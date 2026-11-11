import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


REQUIRED_KEYS = ["t", "drift", "spike_signal", "noise", "reconstruction", "spike_indices"]


def _extract_model_free_reference_benchmarks(t_ref: np.ndarray, raw_ref: np.ndarray) -> dict:
    """
    Extract model-free benchmarks directly from `raw_data` without relying on any
    fitted parametric model, preventing self-bias during inter-team evaluation.
    """
    n_samples = len(raw_ref)
    dt = float(np.median(np.diff(t_ref)))
    fs = 1.0 / dt

    # 1. Model-free spike onset detection from 1st-order voltage jumps
    d1 = np.diff(raw_ref, prepend=raw_ref[0])
    sigma_d1 = float(np.median(np.abs(d1 - np.median(d1))) / 0.6745)
    raw_jumps = np.flatnonzero(d1 > 6.0 * sigma_d1)

    ref_onsets = []
    for idx in raw_jumps:
        if not ref_onsets or (idx - ref_onsets[-1]) >= 5:
            ref_onsets.append(int(idx))
        elif d1[idx] > d1[ref_onsets[-1]]:
            ref_onsets[-1] = int(idx)
    ref_onsets = np.array(ref_onsets, dtype=np.int64)

    # 2. Quiet baseline mask (excluding [-3, +22] ms around each jump) and active spike mask
    quiet_mask = np.ones(n_samples, dtype=bool)
    spike_window_mask = np.zeros(n_samples, dtype=bool)
    for idx in ref_onsets:
        quiet_mask[max(0, idx - 3) : min(n_samples, idx + 23)] = False
        spike_window_mask[idx : min(n_samples, idx + 16)] = True

    # 3. Model-free noise floor sigma from 2nd-order differences on quiet samples:
    # For white noise e[n] ~ N(0, sigma^2), Var(e[n+1] - 2e[n] + e[n-1]) = 6 * sigma^2,
    # whilst slow drift has negligible 2nd difference (~2e-7 V).
    d2_full = np.diff(raw_ref, n=2)
    quiet_d2_mask = quiet_mask[:-2] & quiet_mask[1:-1] & quiet_mask[2:]
    d2_quiet = d2_full[quiet_d2_mask]
    sigma_floor = float(np.sqrt(np.mean(d2_quiet**2) / 6.0))

    total_var = float(np.var(raw_ref))
    target_r2 = float(max(0.01, 1.0 - (sigma_floor**2) / total_var))

    return {
        "n_samples": n_samples,
        "dt": dt,
        "fs": fs,
        "sigma_floor": sigma_floor,
        "target_r2": target_r2,
        "ref_onsets": ref_onsets,
        "quiet_mask": quiet_mask,
        "spike_window_mask": spike_window_mask,
    }


def evaluate(reference_file: str, candidate_file: str) -> dict:
    """
    Inputs:
        reference_file: path to original spike_data.npz
        candidate_file: path to another team's reconstruction

    Output:
        dictionary containing:
        - validity checks - opt
        - numerical evaluation metrics - opt
        - Overlay figure - opt
        - final score - Must
    """
    validity_checks = {
        "file_exists": False,
        "loadable": False,
        "required_keys_present": False,
        "shapes_valid": False,
        "no_nans_or_infs": False,
        "time_vector_matches_reference": False,
        "component_additivity_consistent": False,
        "reconstruction_convention": "unknown",
        "passed": False,
        "notes": [],
    }
    metrics = {}
    fig = None

    # ------------------------------------------------------------------
    # 1. Load Reference Data & Model-Free Benchmarks
    # ------------------------------------------------------------------
    if not os.path.exists(reference_file):
        validity_checks["notes"].append(f"Reference file not found: {reference_file}")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": None,
            "final score": 0.0,
        }

    with np.load(reference_file, allow_pickle=True) as ref_archive:
        t_ref = np.asarray(ref_archive["t"], dtype=np.float64).squeeze()
        raw_ref = np.asarray(ref_archive["raw_data"], dtype=np.float64).squeeze()

    bench = _extract_model_free_reference_benchmarks(t_ref, raw_ref)
    n_ref = bench["n_samples"]

    # ------------------------------------------------------------------
    # 2. Candidate Validity & Schema Checks (10 pts max)
    # ------------------------------------------------------------------
    if not os.path.exists(candidate_file):
        validity_checks["notes"].append(f"Candidate file not found: {candidate_file}")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": None,
            "final score": 0.0,
        }
    validity_checks["file_exists"] = True

    try:
        cand_archive = np.load(candidate_file, allow_pickle=True)
        cand_keys = list(cand_archive.files) if hasattr(cand_archive, "files") else list(cand_archive.keys())
        validity_checks["loadable"] = True
    except Exception as exc:
        validity_checks["notes"].append(f"Failed to load candidate archive: {exc}")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": None,
            "final score": 0.0,
        }

    missing_keys = [k for k in REQUIRED_KEYS if k not in cand_keys]
    if missing_keys:
        validity_checks["notes"].append(f"Missing required keys: {missing_keys}")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": None,
            "final score": 0.0,
        }
    validity_checks["required_keys_present"] = True

    t_cand = np.asarray(cand_archive["t"], dtype=np.float64).squeeze()
    drift_cand = np.asarray(cand_archive["drift"], dtype=np.float64).squeeze()
    spike_cand = np.asarray(cand_archive["spike_signal"], dtype=np.float64).squeeze()
    noise_cand = np.asarray(cand_archive["noise"], dtype=np.float64).squeeze()
    recon_cand = np.asarray(cand_archive["reconstruction"], dtype=np.float64).squeeze()
    indices_cand = np.atleast_1d(np.asarray(cand_archive["spike_indices"]).squeeze())

    shapes_ok = (
        t_cand.shape == (n_ref,)
        and drift_cand.shape == (n_ref,)
        and spike_cand.shape == (n_ref,)
        and noise_cand.shape == (n_ref,)
        and recon_cand.shape == (n_ref,)
        and indices_cand.ndim == 1
    )
    if not shapes_ok:
        validity_checks["notes"].append("One or more arrays do not match the expected (10000,) / 1-D shapes.")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": None,
            "final score": 0.0,
        }
    validity_checks["shapes_valid"] = True

    finite_ok = all(
        np.all(np.isfinite(arr))
        for arr in (t_cand, drift_cand, spike_cand, noise_cand, recon_cand, indices_cand)
    )
    if not finite_ok:
        validity_checks["notes"].append("Candidate arrays contain NaN or Inf values.")
        return {
            "validity checks": validity_checks,
            "numerical evaluation metrics": metrics,
            "Overlay figure": None,
            "final score": 0.0,
        }
    validity_checks["no_nans_or_infs"] = True

    # Check time vector alignment
    time_ok = bool(np.allclose(t_cand, t_ref, atol=1e-4))
    validity_checks["time_vector_matches_reference"] = time_ok
    if not time_ok:
        validity_checks["notes"].append(
            f"Candidate t range [{t_cand.min():.3f}, {t_cand.max():.3f}] differs from reference [0.000, 9.999]."
        )

    # Check reconstruction additivity convention:
    # Convention A (PDF note): reconstruction == drift + spike_signal
    # Convention B (PDF table): reconstruction == drift + spike_signal + noise
    struct_sum = drift_cand + spike_cand
    full_sum = drift_cand + spike_cand + noise_cand
    err_struct = float(np.max(np.abs(recon_cand - struct_sum)))
    err_full = float(np.max(np.abs(recon_cand - full_sum)))

    if err_struct <= 1e-3:
        validity_checks["component_additivity_consistent"] = True
        validity_checks["reconstruction_convention"] = "structured_only (drift + spike_signal)"
        structured_cand = recon_cand
    elif err_full <= 1e-3:
        validity_checks["component_additivity_consistent"] = True
        validity_checks["reconstruction_convention"] = "full_sum (drift + spike_signal + noise)"
        structured_cand = struct_sum
        validity_checks["notes"].append(
            "Candidate reconstruction includes noise; evaluating structured fit on drift + spike_signal."
        )
    else:
        validity_checks["component_additivity_consistent"] = False
        validity_checks["reconstruction_convention"] = "inconsistent"
        structured_cand = struct_sum
        validity_checks["notes"].append(
            f"reconstruction differs from both drift+spikes (max diff {err_struct:.3f}) and drift+spikes+noise ({err_full:.3f})."
        )

    validity_checks["passed"] = True
    if not validity_checks["notes"]:
        validity_checks["notes"].append("All structural and schema checks passed.")

    # Validity score (10 pts): 6 pts for core schema + 2 pts time vector + 2 pts additivity
    score_validity = (
        6.0
        + (2.0 if validity_checks["time_vector_matches_reference"] else 0.0)
        + (2.0 if validity_checks["component_additivity_consistent"] else 0.0)
    )

    # ------------------------------------------------------------------
    # 3. Pillar 2: Structured Signal Fit & Variance Explained (30 pts)
    # ------------------------------------------------------------------
    residual = raw_ref - structured_cand
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    r2 = float(1.0 - np.var(residual) / np.var(raw_ref))
    pearson_r = float(np.corrcoef(raw_ref, structured_cand)[0, 1]) if np.std(structured_cand) > 1e-8 else 0.0

    sigma_floor = bench["sigma_floor"]
    target_r2 = bench["target_r2"]

    r2_ratio = float(np.clip(r2 / target_r2, 0.0, 1.0))
    rel_rmse_gap = max(0.0, abs(rmse - sigma_floor) / sigma_floor - 0.025)
    rmse_match_factor = float(np.exp(-1.5 * rel_rmse_gap))

    score_fit = 30.0 * (0.65 * r2_ratio + 0.35 * rmse_match_factor)

    # ------------------------------------------------------------------
    # 4. Pillar 3: Spike Detection & Waveform Fidelity (25 pts)
    # ------------------------------------------------------------------
    ref_onsets = bench["ref_onsets"]
    cand_onsets = np.sort(indices_cand.astype(np.int64))
    cand_onsets = cand_onsets[(cand_onsets >= 0) & (cand_onsets < n_ref)]

    tol_samples = 3
    used_cand = np.zeros(len(cand_onsets), dtype=bool)
    timing_errors = []
    for r_idx in ref_onsets:
        if len(cand_onsets) == 0:
            break
        diffs = np.abs(cand_onsets - r_idx)
        diffs[used_cand] = 10**9
        best_pos = int(np.argmin(diffs))
        if diffs[best_pos] <= tol_samples:
            used_cand[best_pos] = True
            timing_errors.append(int(cand_onsets[best_pos] - r_idx))

    tp = len(timing_errors)
    precision = float(tp / len(cand_onsets)) if len(cand_onsets) > 0 else 0.0
    recall = float(tp / len(ref_onsets)) if len(ref_onsets) > 0 else 0.0
    f1_spikes = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_abs_timing_err_ms = float(np.mean(np.abs(timing_errors))) if timing_errors else float("nan")

    quiet_mask = bench["quiet_mask"]
    spike_window_mask = bench["spike_window_mask"]
    spike_quiet_rmse = float(np.sqrt(np.mean(spike_cand[quiet_mask] ** 2)))
    sparsity_score = 5.0 * float(np.exp(-spike_quiet_rmse / 0.05))

    emp_spike_target = (raw_ref - drift_cand)[spike_window_mask]
    cand_spike_active = spike_cand[spike_window_mask]
    if np.std(cand_spike_active) > 1e-6 and np.std(emp_spike_target) > 1e-6:
        wf_corr = float(np.corrcoef(emp_spike_target, cand_spike_active)[0, 1])
    else:
        wf_corr = 0.0
    wf_score = 5.0 * float(np.clip(wf_corr / 0.965, 0.0, 1.0))

    score_spikes = 15.0 * f1_spikes + sparsity_score + wf_score

    # ------------------------------------------------------------------
    # 5. Pillar 4: Baseline Drift Quality & Smoothness (20 pts)
    # ------------------------------------------------------------------
    raw_quiet = raw_ref[quiet_mask]
    drift_quiet = drift_cand[quiet_mask]
    if np.std(drift_quiet) > 1e-6 and np.std(raw_quiet) > 1e-6:
        drift_quiet_corr = float(np.corrcoef(raw_quiet, drift_quiet)[0, 1])
    else:
        drift_quiet_corr = 0.0

    expected_quiet_corr = float(np.sqrt(max(0.01, 1.0 - (sigma_floor**2) / np.var(raw_quiet))))
    drift_corr_ratio = float(np.clip(drift_quiet_corr / expected_quiet_corr, 0.0, 1.0))

    freqs = np.fft.rfftfreq(n_ref, d=bench["dt"])
    drift_centred = drift_cand - np.mean(drift_cand)
    power = np.abs(np.fft.rfft(drift_centred)) ** 2
    total_power = float(np.sum(power))
    low_freq_energy_ratio = float(np.sum(power[freqs <= 1.0]) / total_power) if total_power > 1e-12 else 0.0

    score_drift = 12.0 * drift_corr_ratio + 8.0 * low_freq_energy_ratio

    # ------------------------------------------------------------------
    # 6. Pillar 5: Residual Whiteness & Synthesised Noise Fidelity (15 pts)
    # ------------------------------------------------------------------
    res_mean = float(np.mean(residual))
    res_std = float(np.std(residual, ddof=1))
    res_lag1_acf = float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if res_std > 1e-8 else 1.0
    ks_res_stat, ks_res_p = stats.kstest(residual, "norm", args=(res_mean, max(res_std, 1e-8)))

    res_mean_factor = float(np.exp(-abs(res_mean) / 0.03))
    res_acf_factor = float(np.exp(-max(0.0, abs(res_lag1_acf) - 0.025) / 0.08))
    res_norm_factor = float(np.exp(-max(0.0, ks_res_stat - 0.015) / 0.04))
    score_residual_whiteness = 8.0 * (0.3 * res_mean_factor + 0.35 * res_acf_factor + 0.35 * res_norm_factor)

    noise_mean = float(np.mean(noise_cand))
    noise_std = float(np.std(noise_cand, ddof=1))
    noise_lag1_acf = float(np.corrcoef(noise_cand[:-1], noise_cand[1:])[0, 1]) if noise_std > 1e-8 else 1.0
    ks_noise_stat, ks_noise_p = stats.kstest(noise_cand, "norm", args=(noise_mean, max(noise_std, 1e-8)))

    copied_residual = bool(np.allclose(noise_cand, residual, atol=1e-3))
    noise_std_factor = float(np.exp(-max(0.0, abs(noise_std - sigma_floor) / sigma_floor - 0.025) * 3.0))
    noise_white_factor = float(np.exp(-max(0.0, abs(noise_lag1_acf) - 0.025) / 0.08))
    noise_norm_factor = float(np.exp(-max(0.0, ks_noise_stat - 0.015) / 0.04))
    score_noise_synth = 7.0 * (0.5 * noise_std_factor + 0.25 * noise_white_factor + 0.25 * noise_norm_factor)
    if copied_residual:
        score_noise_synth *= 0.5
        validity_checks["notes"].append("Candidate noise array is copied from residual rather than synthesised.")

    score_noise_total = score_residual_whiteness + score_noise_synth

    # ------------------------------------------------------------------
    # 7. Final Composite Score & Metrics Dictionary
    # ------------------------------------------------------------------
    final_score = float(
        np.clip(
            score_validity + score_fit + score_spikes + score_drift + score_noise_total,
            0.0,
            100.0,
        )
    )

    metrics = {
        "final_score_breakdown": {
            "validity_score_10": round(score_validity, 2),
            "structured_fit_score_30": round(score_fit, 2),
            "spike_detection_waveform_score_25": round(score_spikes, 2),
            "drift_quality_score_20": round(score_drift, 2),
            "residual_and_noise_score_15": round(score_noise_total, 2),
        },
        "reconstruction_rmse": round(rmse, 5),
        "reconstruction_mae": round(mae, 5),
        "reconstruction_r2": round(r2, 5),
        "theoretical_ceiling_r2": round(target_r2, 5),
        "pearson_r": round(pearson_r, 5),
        "model_free_noise_floor_sigma": round(sigma_floor, 5),
        "reference_spike_count": int(len(ref_onsets)),
        "candidate_spike_count": int(len(cand_onsets)),
        "spike_precision": round(precision, 4),
        "spike_recall": round(recall, 4),
        "spike_f1_score": round(f1_spikes, 4),
        "mean_abs_timing_error_ms": round(mean_abs_timing_err_ms, 4) if np.isfinite(mean_abs_timing_err_ms) else None,
        "spike_quiet_rmse": round(spike_quiet_rmse, 5),
        "spike_waveform_correlation": round(wf_corr, 4),
        "drift_quiet_correlation": round(drift_quiet_corr, 4),
        "drift_low_freq_energy_ratio": round(low_freq_energy_ratio, 5),
        "residual_mean": round(res_mean, 5),
        "residual_std": round(res_std, 5),
        "residual_lag1_acf": round(res_lag1_acf, 4),
        "residual_ks_pvalue": round(float(ks_res_p), 4),
        "synthesised_noise_mean": round(noise_mean, 5),
        "synthesised_noise_std": round(noise_std, 5),
        "synthesised_noise_lag1_acf": round(noise_lag1_acf, 4),
        "synthesised_noise_ks_pvalue": round(float(ks_noise_p), 4),
    }

    # ------------------------------------------------------------------
    # 8. Diagnostic Overlay Figure (4-Panel Publication Layout)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=140)
    cand_name = Path(candidate_file).stem

    # Panel A: Full 10 s Trace & Structured Reconstruction + Drift
    axes[0, 0].plot(t_ref, raw_ref, color="#7f8c8d", lw=0.45, alpha=0.65, label="Reference raw_data")
    axes[0, 0].plot(t_ref, structured_cand, color="#2980b9", lw=0.85, alpha=0.9, label="Candidate (drift + spikes)")
    axes[0, 0].plot(t_ref, drift_cand, color="#d35400", lw=1.8, label="Candidate drift")
    if len(cand_onsets) > 0:
        axes[0, 0].scatter(
            t_ref[cand_onsets],
            raw_ref[cand_onsets],
            color="#c0392b",
            s=14,
            zorder=5,
            label=f"Candidate spikes (N={len(cand_onsets)})",
        )
    axes[0, 0].set_title(f"{cand_name} — Full Trace Overlay (R² = {r2*100:.1f}%, RMSE = {rmse:.3f} V)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Voltage (V)")
    axes[0, 0].set_xlim(0, 10)
    axes[0, 0].legend(loc="upper right", fontsize=8)

    # Panel B: 400 ms Zoom Window (2.20 s to 2.60 s)
    zoom = (t_ref >= 2.20) & (t_ref <= 2.60)
    axes[0, 1].plot(t_ref[zoom], raw_ref[zoom], color="#34495e", lw=1.2, alpha=0.8, label="Reference raw_data")
    axes[0, 1].plot(t_ref[zoom], structured_cand[zoom], color="#e74c3c", lw=1.5, ls="--", label="Candidate structured signal")
    axes[0, 1].plot(t_ref[zoom], drift_cand[zoom], color="#d35400", lw=1.6, label="Candidate drift")
    axes[0, 1].set_title(f"400 ms Close-Up Zoom (Spike F1 = {f1_spikes:.3f}, WF r = {wf_corr:.3f})")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Voltage (V)")
    axes[0, 1].legend(loc="upper right", fontsize=8)

    # Panel C: Residual vs Synthesised Noise Distribution
    v_grid = np.linspace(-1.4, 1.4, 200)
    axes[1, 0].hist(residual, bins=50, density=True, alpha=0.55, color="#2980b9", label=f"Empirical residual (SD={res_std:.3f})")
    axes[1, 0].hist(
        noise_cand,
        bins=50,
        density=True,
        histtype="step",
        lw=1.6,
        color="#e67e22",
        label=f"Candidate noise (SD={noise_std:.3f})",
    )
    axes[1, 0].plot(
        v_grid,
        stats.norm.pdf(v_grid, 0.0, sigma_floor),
        "k--",
        lw=1.5,
        label=f"Ref N(0, {sigma_floor:.3f}²)",
    )
    axes[1, 0].set_title(f"Residual & Noise Distributions (KS p = {ks_res_p:.2f})")
    axes[1, 0].set_xlabel("Voltage (V)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    # Panel D: Sub-Score Breakdown Bar Chart
    sub_names = [
        "Validity\n(/10)",
        "Signal Fit\n(/30)",
        "Spikes\n(/25)",
        "Drift\n(/20)",
        "Noise/Resid\n(/15)",
    ]
    sub_vals = [score_validity, score_fit, score_spikes, score_drift, score_noise_total]
    sub_max = [10.0, 30.0, 25.0, 20.0, 15.0]
    pcts = [100.0 * v / m for v, m in zip(sub_vals, sub_max)]
    bars = axes[1, 1].bar(sub_names, pcts, color=["#27ae60", "#2980b9", "#8e44ad", "#d35400", "#16a085"], alpha=0.85)
    axes[1, 1].set_ylim(0, 115)
    axes[1, 1].axhline(100, color="grey", ls=":", lw=1.0)
    for bar, val, mx in zip(bars, sub_vals, sub_max):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.5,
            f"{val:.1f}/{mx:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[1, 1].set_title(f"Rubric Sub-Score Percentages — Final Score: {final_score:.2f} / 100")
    axes[1, 1].set_ylabel("Category Score (%)")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return {
        "validity checks": validity_checks,
        "numerical evaluation metrics": metrics,
        "Overlay figure": fig,
        "final score": round(final_score, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more candidate reconstruction .npz files against spike_data.npz."
    )
    parser.add_argument(
        "reference_file",
        nargs="?",
        default="data/input/spike_data.npz",
        help="Path to reference spike_data.npz",
    )
    parser.add_argument(
        "candidate_files",
        nargs="*",
        default=["data/output/team_10_reconstruction.npz"],
        help="One or more candidate reconstruction .npz files",
    )
    parser.add_argument(
        "--out-dir",
        default="data/output/evaluation_results",
        help="Directory to save per-team JSON reports, overlay PNGs, and summary CSV",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_rows = []
    for cand_path in args.candidate_files:
        res = evaluate(args.reference_file, cand_path)
        stem = Path(cand_path).stem

        fig = res.get("Overlay figure")
        fig_path = None
        if fig is not None:
            fig_path = out_dir / f"{stem}_overlay.png"
            fig.savefig(fig_path, dpi=160, bbox_inches="tight")
            plt.close(fig)

        report_payload = {
            "reference_file": str(args.reference_file),
            "candidate_file": str(cand_path),
            "final score": res["final score"],
            "validity checks": res["validity checks"],
            "numerical evaluation metrics": res["numerical evaluation metrics"],
            "overlay_figure_path": str(fig_path) if fig_path else None,
        }
        json_path = out_dir / f"{stem}_evaluation.json"
        json_path.write_text(json.dumps(report_payload, indent=2))

        m = res["numerical evaluation metrics"]
        b = m.get("final_score_breakdown", {})
        leaderboard_rows.append(
            {
                "candidate": stem,
                "final_score": res["final score"],
                "validity_10": b.get("validity_score_10", 0.0),
                "fit_30": b.get("structured_fit_score_30", 0.0),
                "spikes_25": b.get("spike_detection_waveform_score_25", 0.0),
                "drift_20": b.get("drift_quality_score_20", 0.0),
                "noise_15": b.get("residual_and_noise_score_15", 0.0),
                "rmse": m.get("reconstruction_rmse"),
                "r2": m.get("reconstruction_r2"),
                "spike_f1": m.get("spike_f1_score"),
            }
        )

    leaderboard_rows.sort(key=lambda r: -r["final_score"])
    csv_path = out_dir / "leaderboard_summary.csv"
    if leaderboard_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(leaderboard_rows[0].keys()))
            writer.writeheader()
            writer.writerows(leaderboard_rows)

    print("\n" + "=" * 92)
    print(f"{'Candidate':<38} {'Score':>7} {'Valid':>6} {'Fit':>6} {'Spikes':>7} {'Drift':>6} {'Noise':>6} {'RMSE':>7} {'R²':>7}")
    print("-" * 92)
    for row in leaderboard_rows:
        print(
            f"{row['candidate']:<38} {row['final_score']:>7.2f} "
            f"{row['validity_10']:>6.1f} {row['fit_30']:>6.1f} "
            f"{row['spikes_25']:>7.1f} {row['drift_20']:>6.1f} {row['noise_15']:>6.1f} "
            f"{(row['rmse'] or 0.0):>7.4f} {(row['r2'] or 0.0):>7.4f}"
        )
    print("=" * 92)
    print(f"Saved individual JSON result files and overlay figures to: {out_dir}/")


if __name__ == "__main__":
    main()
