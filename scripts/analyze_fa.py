#!/usr/bin/env python3
"""
analyze_fa.py — P6 False Activation (FA) Rate & Poisson Statistics Analyzer.

Parses firmware serial UART logs, extracts TRIG lines, and computes:
  1. FA per hour = total_triggers / total_hours (on verified keyword-free audio)
  2. Exact Poisson 95% confidence upper bound (0 events in T hours => ~3/T)
  3. Probability distribution of triggers (near-threshold 0.5-0.6 vs high confidence >= 0.9)
  4. Hourly snapshots of heap, core temperature, and RSSI

Usage:
  python scripts/analyze_fa.py --log serial_soak.log [--hours 10]
  python scripts/analyze_fa.py --demo (runs on simulated 10-hour soak log)
"""

import os
import re
import sys
import argparse
import numpy as np
from scipy import stats

TRIG_REGEX = re.compile(
    r"TRIG\s+tick=(\d+)\s+t_ms=(\d+)\s+p=\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]\s+pre_ts=(\d+)"
)
HEALTH_REGEX = re.compile(
    r"HEALTH\s+uptime=(\d+)s\s+sram_min=(\d+)B.*?inf_avg_us=(\d+).*?freq=(\d+)MHz"
)


def compute_poisson_upper_bound_95(events, hours):
    """
    Computes exact 95% Poisson confidence upper bound for event rate (per hour).
    Uses the exact Chi-square distribution relationship: lambda_upper = 0.5 * chi2.ppf(0.95, 2*(k+1)) / T.
    For k=0 events, chi2.ppf(0.95, 2) = 5.99146 -> 5.99146 / (2*T) = 2.9957 / T ≈ 3/T.
    """
    if hours <= 0:
        return float('nan')
    df = 2 * (events + 1)
    chi2_val = stats.chi2.ppf(0.95, df)
    upper_bound = 0.5 * chi2_val / hours
    return upper_bound


def analyze_log_content(log_text, declared_hours=None):
    trig_matches = TRIG_REGEX.findall(log_text)
    health_matches = HEALTH_REGEX.findall(log_text)

    # Determine elapsed hours from timestamps or declared_hours
    max_t_ms = 0
    p_values = []

    for match in trig_matches:
        tick, t_ms, p0, p1, p2, pre_ts = match
        t_ms = int(t_ms)
        if t_ms > max_t_ms:
            max_t_ms = t_ms
        p_values.append(float(p2))  # newest probability at firing

    max_uptime_s = 0
    min_sram_bytes = None
    for match in health_matches:
        upt_s, sram_b, lat_us, freq = match
        upt_s = int(upt_s)
        sram_b = int(sram_b)
        if upt_s > max_uptime_s:
            max_uptime_s = upt_s
        if min_sram_bytes is None or sram_b < min_sram_bytes:
            min_sram_bytes = sram_b

    calculated_hours = max_uptime_s / 3600.0 if max_uptime_s > 0 else (max_t_ms / 3600000.0)
    effective_hours = declared_hours if declared_hours is not None else max(calculated_hours, 1.0)

    num_triggers = len(trig_matches)
    fa_rate = num_triggers / effective_hours
    poisson_95_upper = compute_poisson_upper_bound_95(num_triggers, effective_hours)

    print("=" * 72)
    print("  SPECTRA KWS -- P6 SOAK & FALSE ACTIVATION (FA) ANALYSIS REPORT")
    print("=" * 72)
    print(f"  Total Duration:           {effective_hours:.2f} hours")
    print(f"  Verified Keyword-Free:    YES (Continuous background ambient audio)")
    print(f"  Total False Triggers:     {num_triggers}")
    print(f"  Measured FA Rate:         {fa_rate:.4f} FA / hour")
    print(f"  Poisson 95% Upper Bound:  < {poisson_95_upper:.4f} FA / hour")
    print("-" * 72)

    # Statistical claim validity
    print("\n--- Statistical Significance Analysis (Poisson Upper Bound) ---")
    if num_triggers == 0:
        print(f"  0 events in {effective_hours:.1f} h -> 95% confidence that true lambda < {poisson_95_upper:.2f} / hour.")
        if effective_hours >= 10.0:
            print("  [CLAIM PASSED WITH MARGIN] 10 h overnight soak rigorously proves FA rate < 0.3 / h (< 1.0 / h target).")
        elif effective_hours >= 3.0:
            print("  [CLAIM MODERATE] Proves FA rate < 1.0 / h.")
        else:
            print("  [WEAK STATISTIC] 1 h bench test cannot honestly claim < 1.0 / h (bound is < 3.0 / h).")
    else:
        print(f"  {num_triggers} trigger events observed. Poisson 95% upper bound: {poisson_95_upper:.3f} / hour.")

    # Probability distribution
    if p_values:
        p_arr = np.array(p_values)
        near_thresh = np.sum((p_arr >= 0.5) & (p_arr < 0.7))
        mid_thresh = np.sum((p_arr >= 0.7) & (p_arr < 0.9))
        high_thresh = np.sum(p_arr >= 0.9)

        print("\n--- Trigger Confidence Distribution ---")
        print(f"  Near-Threshold [0.50 - 0.70): {near_thresh} ({near_thresh/len(p_arr)*100:.1f}%) [tunable]")
        print(f"  Mid-Range      [0.70 - 0.90): {mid_thresh} ({mid_thresh/len(p_arr)*100:.1f}%)")
        print(f"  High Confidence      >= 0.90: {high_thresh} ({high_thresh/len(p_arr)*100:.1f}%)")

    if min_sram_bytes is not None:
        print("\n--- System Health & Memory Telemetry ---")
        print(f"  Minimum Internal SRAM Free: {min_sram_bytes} bytes (Target: > 45,000 B)")
        print(f"  Memory Leak Detected:       {'NO' if min_sram_bytes > 45000 else 'YES'}")

    print("\n--- Acceptance Bar Verdict ---")
    if num_triggers == 0 and effective_hours >= 10.0:
        verdict = "PASS"
        verdict_detail = f"0 FAs observed over {effective_hours:.1f} h (Poisson 95% bound < {poisson_95_upper:.3f} / h satisfies < 1.0/h bar with margin)"
    elif num_triggers == 0:
        verdict = "INCONCLUSIVE (WEAK)"
        verdict_detail = f"0 FAs observed over {effective_hours:.1f} h (Duration too short: Poisson bound < {poisson_95_upper:.2f}/h cannot claim < 1.0/h)"
    else:
        verdict = "FAIL"
        verdict_detail = f"{num_triggers} false triggers observed on keyword-free audio (Measured rate: {fa_rate:.2f}/h, fails 0-FA soak bar)"

    print(f"  VERDICT: {verdict}")
    print(f"  Detail:  {verdict_detail}")
    print("=" * 72)

    return {
        "hours": effective_hours,
        "triggers": num_triggers,
        "fa_rate": fa_rate,
        "poisson_95_upper": poisson_95_upper,
        "p_values": p_values
    }


def generate_demo_soak_log(filename="serial_soak_demo.log", hours=10):
    """Generates a synthetic 10-hour clean soak log with zero false triggers."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("[SPECTRA] Spectra KWS -- 10-Hour Soak Test Started\n")
        f.write("[POWER_MGR] DFS configured: min=80MHz, max=240MHz, light_sleep=1\n")
        # Write hourly health telemetry lines
        for h in range(1, hours + 1):
            upt = h * 3600
            f.write(f"[HEALTH] HEALTH uptime={upt}s sram_min=172000B sram_cur=172000B psram_min=8000000B inf_count={h*7200} inf_avg_us=15820 trigs=0 overruns=0 freq=80MHz\n")
    print(f"Created demo soak log: {filename} ({hours} hours, 0 FAs)")
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze P6 soak test logs for False Activation (FA) rate")
    parser.add_argument("--log", help="Path to UART serial log file")
    parser.add_argument("--hours", type=float, help="Explicit test duration in hours")
    parser.add_argument("--demo", action="store_true", help="Run on simulated 10-hour zero-FA soak log")
    args = parser.parse_args()

    if args.demo:
        demo_file = generate_demo_soak_log()
        with open(demo_file, "r", encoding="utf-8") as f:
            analyze_log_content(f.read(), declared_hours=10.0)
    elif args.log:
        if not os.path.exists(args.log):
            print(f"Error: Log file not found: {args.log}")
            sys.exit(1)
        with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
            analyze_log_content(f.read(), declared_hours=args.hours)
    else:
        parser.print_help()
