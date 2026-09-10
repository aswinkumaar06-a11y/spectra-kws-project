#!/usr/bin/env python3
"""
analyze_fa.py — Algorithm FA-1 (Normative False-Accept Analyzer).

Canonical implementation matching Spectra Phase 6 Specification:
  1. Parses serial log files. Matches:
       - Standard format: TRIG tick=<int> p=<float>
       - Firmware format: TRIG tick=<int> t_ms=<int> p=[p0, p1, p2] pre_ts=<int>
     Tracks unparseable / malformed lines without crashing.
  2. Duration: T = (t_last - t_first) in hours from timestamps or tick deltas (500 ms/tick).
  3. Events: k = count of TRIG events in [t_first, t_last].
  4. Observed FA rate = k / T.
  5. One-sided 95% Poisson upper bound on rate:
       λ_upper = χ²⁻¹(0.95, df = 2(k+1)) / (2T)
     Special case k = 0 => λ_upper = 5.99146 / (2T) ≈ 3.0 / T.
  6. Verdict table:
       k = 0 and T >= 10 h => "PASS (strong): λ < 0.30/h, beats <1.0/h bar with margin"
       k = 0 and T >=  1 h => "PASS (weak): λ < 3.0/T/h"
       k > 0 and λ_upper > 1.0/h => "FAIL vs <1.0/h claim bar"
       else => "INCONCLUSIVE: extend soak duration"
  7. Confidence-bucket histogram:
       near-threshold 0.50–0.70 | mid 0.70–0.90 | high >= 0.90.
  8. Output: T, k, k/T, λ_upper, verdict, bucket table, unparseable-line count.
"""

import os
import re
import sys
import argparse
import numpy as np
from scipy import stats

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def safe_print(s):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", errors="replace").decode("ascii"))

# Regex for standard or firmware TRIG line
# Matches:
#   TRIG tick=18000 p=0.61
#   TRIG tick=14400 t_ms=7200000 p=[0.450, 0.520, 0.610] pre_ts=7199000
#   [TRIGGER] TRIG tick=...
TRIG_RE = re.compile(
    r"(?:\[TRIGGER\]\s+)?TRIG\s+tick=(\d+)(?:\s+t_ms=(\d+))?\s+p=(?:\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]|([0-9.]+))"
)

# Regex for heartbeat / health lines with timestamp or uptime
HEALTH_RE = re.compile(
    r"(?:\[HEALTH\]\s+)?HEALTH\s+uptime=(\d+)s"
)
TICK_LINE_RE = re.compile(
    r"(?:tick=(\d+)|uptime=(\d+)s|t_ms=(\d+))"
)


# Wall-clock timestamp regex: matches YYYY-MM-DD HH:MM:SS(.fff)? or HH:MM:SS(.fff)?
WALL_CLOCK_RE = re.compile(
    r"(?:^|\[)(\d{4}-\d{2}-\d{2}[ T])?(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)(?:\])?"
)


def parse_wall_clock_time(timestr, datestr=None):
    """Parses wall-clock timestamp string into total seconds."""
    parts = timestr.split(":")
    h = float(parts[0])
    m = float(parts[1])
    s = float(parts[2])
    day_offset = 0.0
    if datestr:
        # If full date is present, calculate date offset from day
        date_clean = datestr.strip(" T")
        try:
            from datetime import datetime
            dt = datetime.strptime(f"{date_clean} {timestr}", "%Y-%m-%d %H:%M:%S.%f" if "." in timestr else "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except Exception:
            pass
    return h * 3600.0 + m * 60.0 + s


def compute_poisson_upper_bound_95(events, hours):
    """
    Computes exact one-sided 95% Poisson confidence upper bound for event rate (per hour).
    λ_upper = χ²⁻¹(0.95, df = 2(k+1)) / (2T)
    For k=0, χ²⁻¹(0.95, 2) = 5.99146 => 5.99146 / (2T) ≈ 3.0 / T.
    """
    if hours <= 0:
        return float('nan')
    df = 2 * (events + 1)
    chi2_val = stats.chi2.ppf(0.95, df)
    upper_bound = 0.5 * chi2_val / hours
    return float(upper_bound)


def analyze_log_content(log_text, declared_hours=None):
    lines = log_text.splitlines()
    unparseable_count = 0

    first_tick_s = None
    last_tick_s = None
    first_wc_s = None
    last_wc_s = None

    trig_events = []

    for line_idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        # Check for wall-clock timestamps
        wc_match = WALL_CLOCK_RE.search(line)
        if wc_match:
            datestr, timestr = wc_match.groups()
            try:
                wc_s = parse_wall_clock_time(timestr, datestr)
                if first_wc_s is None:
                    first_wc_s = wc_s
                last_wc_s = wc_s
            except Exception:
                pass

        # Check for TRIG line
        trig_match = TRIG_RE.search(line)
        if trig_match:
            tick_str, t_ms_str, p0, p1, p2, p_single = trig_match.groups()
            tick = int(tick_str)
            p_val = float(p_single) if p_single is not None else float(p2)

            if t_ms_str is not None:
                t_s = int(t_ms_str) / 1000.0
            else:
                t_s = tick * 0.5  # 500 ms per tick

            if first_tick_s is None or t_s < first_tick_s:
                first_tick_s = t_s
            if last_tick_s is None or t_s > last_tick_s:
                last_tick_s = t_s

            trig_events.append({
                "line": line_idx + 1,
                "tick": tick,
                "time_s": t_s,
                "p": p_val
            })
            continue

        # Check for other recognized timestamp/heartbeat lines
        health_match = HEALTH_RE.search(line)
        if health_match:
            upt_s = float(health_match.group(1))
            if first_tick_s is None or 0.0 < first_tick_s:
                first_tick_s = 0.0
            if last_tick_s is None or upt_s > last_tick_s:
                last_tick_s = upt_s
            continue

        # Check for generic tick indicators
        tick_match = TICK_LINE_RE.search(line)
        if tick_match:
            tk, up, tm = tick_match.groups()
            if tk is not None:
                t_s = int(tk) * 0.5
            elif up is not None:
                t_s = float(up)
            else:
                t_s = float(tm) / 1000.0

            if first_tick_s is None or t_s < first_tick_s:
                first_tick_s = t_s
            if last_tick_s is None or t_s > last_tick_s:
                last_tick_s = t_s
            continue

        # Check if line contains known metadata, banners, or separators
        if (line.startswith("[SPECTRA]") or line.startswith("[POWER_MGR]") or 
            line.startswith("[BOOT]") or line.startswith("[SOAK]") or line.startswith("=")):
            continue

        # Line not recognized by grammar
        unparseable_count += 1

    # Duration T calculation
    duration_ticks_s = (last_tick_s - first_tick_s) if (first_tick_s is not None and last_tick_s is not None) else None
    duration_wc_s = (last_wc_s - first_wc_s) if (first_wc_s is not None and last_wc_s is not None) else None

    if declared_hours is not None:
        T = float(declared_hours)
    elif duration_wc_s is not None and duration_wc_s > 0:
        T = duration_wc_s / 3600.0
    elif duration_ticks_s is not None and duration_ticks_s > 0:
        T = duration_ticks_s / 3600.0
    else:
        raise ValueError("cannot bound duration (no valid timestamps or duration <= 0)")

    if T <= 0:
        raise ValueError("cannot bound duration (T <= 0)")

    k = len(trig_events)
    fa_rate = k / T
    lambda_upper = compute_poisson_upper_bound_95(k, T)

    # Verdict table per Step 6 of Algorithm FA-1
    if k == 0 and T >= 10.0:
        verdict = "PASS (strong): λ < 0.30/h, beats <1.0/h bar with margin"
        status_code = "PASS"
    elif k == 0 and T >= 1.0:
        verdict = f"PASS (weak): λ < {3.0 / T:.2f}/h"
        status_code = "PASS_WEAK"
    elif k > 0 and lambda_upper > 1.0:
        verdict = f"FAIL vs <1.0/h claim bar (measured {fa_rate:.2f}/h, λ_upper = {lambda_upper:.2f}/h)"
        status_code = "FAIL"
    else:
        verdict = f"INCONCLUSIVE: extend soak duration (measured {fa_rate:.2f}/h, λ_upper = {lambda_upper:.2f}/h)"
        status_code = "INCONCLUSIVE"

    # Confidence-bucket histogram per Step 7
    p_values = [e["p"] for e in trig_events]
    near_count = sum(1 for p in p_values if 0.50 <= p < 0.70)
    mid_count = sum(1 for p in p_values if 0.70 <= p < 0.90)
    high_count = sum(1 for p in p_values if p >= 0.90)

    safe_print("=" * 75)
    safe_print("  SPECTRA KWS -- ALGORITHM FA-1 FALSE-ACCEPT ANALYZER")
    safe_print("=" * 75)
    safe_print(f"  Duration (T):             {T:.2f} hours ({T*3600:.0f} seconds)")
    safe_print(f"  False-Accept Count (k):   {k} events")
    safe_print(f"  Observed FA Rate (k/T):   {fa_rate:.4f} FA / hour")
    safe_print(f"  Poisson 95% Upper (λ):    < {lambda_upper:.4f} / hour")
    safe_print(f"  Unparseable Line Count:   {unparseable_count}")
    safe_print("-" * 75)
    safe_print(f"  VERDICT:                  {verdict}")
    safe_print("-" * 75)
    safe_print("  Confidence-Bucket Histogram of TRIG Events:")
    safe_print(f"    - Near-Threshold [0.50, 0.70):  {near_count:3d}")
    safe_print(f"    - Mid-Range      [0.70, 0.90):  {mid_count:3d}")
    safe_print(f"    - High-Conf      [0.90, 1.00]:  {high_count:3d}")
    safe_print("=" * 75)

    return {
        "duration_hours": T,
        "duration_ticks_s": duration_ticks_s,
        "duration_wallclock_s": duration_wc_s,
        "k_events": k,
        "triggers": k,  # compat
        "fa_rate": fa_rate,
        "lambda_upper": lambda_upper,
        "poisson_95_upper": lambda_upper,  # compat
        "hours": T,  # compat
        "p_values": p_values,  # compat
        "verdict": verdict,
        "status_code": status_code,
        "unparseable_count": unparseable_count,
        "buckets": {
            "near": near_count,
            "mid": mid_count,
            "high": high_count
        }
    }



def main():
    parser = argparse.ArgumentParser(description="Algorithm FA-1 False-Accept Analyzer")
    parser.add_argument("--log", required=True, help="Path to serial log file")
    parser.add_argument("--hours", type=float, help="Explicit duration in hours (overrides timestamp math)")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Error: Log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    try:
        analyze_log_content(content, declared_hours=args.hours)
    except Exception as e:
        print(f"Error analyzing log: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
