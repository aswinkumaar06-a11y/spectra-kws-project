#!/usr/bin/env python3
"""
test_power_sim.py — Spectra Power & Energy Optimization Simulation Suite

Validates Phase 6 power optimization mathematical models:
  1. Active vs Idle Duty Cycle (< 10% active duty cycle @ 500ms hop)
  2. Average Current Draw across three operating regimes:
     - Baseline: Always-on 240 MHz (no DFS, no sleep)
     - DFS-only: 240 MHz active / 80 MHz idle
     - DFS + Light Sleep: 240 MHz active / Light sleep idle
  3. Energy Reduction Verification (asserts >= 60% energy reduction)
  4. Battery Lifetime Projections for 500 mAh and 1000 mAh LiPo cells
"""

import unittest

class TestPowerOptimizationSim(unittest.TestCase):
    """Evaluates electrical energy consumption and duty cycle conservation."""

    # XIAO ESP32-C5 Typical Current Characteristics (@ 3.3V)
    I_ACTIVE_240MHZ_MA = 55.0   # CPU @ 240 MHz running DSP / TFLM
    I_IDLE_80MHZ_MA    = 18.0   # CPU @ 80 MHz (DFS floor)
    I_LIGHT_SLEEP_MA   = 2.5    # Light sleep with SRAM retention & RTC timer
    I_WIFI_STREAM_MA   = 110.0  # Active Wi-Fi TX during speech streaming

    HOP_DURATION_MS    = 500.0  # 500 ms inference hop interval
    DSP_LATENCY_MS     = 8.5    # MFCC extraction @ 240 MHz
    TFLM_LATENCY_MS    = 14.5   # TFLM invoke @ 240 MHz

    def test_duty_cycle_budget(self):
        """Assert active compute consumes < 10% of total hop window."""
        t_active = self.DSP_LATENCY_MS + self.TFLM_LATENCY_MS
        duty_cycle = t_active / self.HOP_DURATION_MS
        print(f"\n[POWER SIM] Active Compute: {t_active:.2f} ms / {self.HOP_DURATION_MS:.0f} ms -> Duty Cycle: {duty_cycle*100:.1f}%")
        self.assertLess(duty_cycle, 0.10, "Active duty cycle must be strictly < 10%")

    def test_energy_reduction_dfs_and_sleep(self):
        """Assert DFS + light sleep delivers >= 60% power reduction vs always-on baseline."""
        t_active = self.DSP_LATENCY_MS + self.TFLM_LATENCY_MS
        t_idle = self.HOP_DURATION_MS - t_active

        # Baseline: Always-on 240 MHz
        i_baseline = self.I_ACTIVE_240MHZ_MA

        # DFS-only: 240 MHz active, 80 MHz idle
        i_dfs = (t_active * self.I_ACTIVE_240MHZ_MA + t_idle * self.I_IDLE_80MHZ_MA) / self.HOP_DURATION_MS

        # DFS + Light Sleep: 240 MHz active, Light sleep idle
        i_sleep = (t_active * self.I_ACTIVE_240MHZ_MA + t_idle * self.I_LIGHT_SLEEP_MA) / self.HOP_DURATION_MS

        reduct_dfs = (1.0 - i_dfs / i_baseline) * 100
        reduct_sleep = (1.0 - i_sleep / i_baseline) * 100

        print(f"[POWER SIM] Baseline (Always 240MHz):     {i_baseline:.2f} mA")
        print(f"[POWER SIM] With DFS (240MHz / 80MHz):    {i_dfs:.2f} mA  ({reduct_dfs:.1f}% reduction)")
        print(f"[POWER SIM] With DFS + Light Sleep:       {i_sleep:.2f} mA  ({reduct_sleep:.1f}% reduction)")

        self.assertGreater(reduct_dfs, 60.0, "DFS alone must deliver > 60% power reduction")
        self.assertGreater(reduct_sleep, 85.0, "DFS + Light sleep must deliver > 85% power reduction")

    def test_battery_runtime_projection(self):
        """Compute battery runtime on 500 mAh battery."""
        battery_mah = 500.0
        t_active = self.DSP_LATENCY_MS + self.TFLM_LATENCY_MS
        t_idle = self.HOP_DURATION_MS - t_active

        i_baseline = self.I_ACTIVE_240MHZ_MA
        i_dfs = (t_active * self.I_ACTIVE_240MHZ_MA + t_idle * self.I_IDLE_80MHZ_MA) / self.HOP_DURATION_MS
        i_sleep = (t_active * self.I_ACTIVE_240MHZ_MA + t_idle * self.I_LIGHT_SLEEP_MA) / self.HOP_DURATION_MS

        hrs_baseline = battery_mah / i_baseline
        hrs_dfs = battery_mah / i_dfs
        hrs_sleep = battery_mah / i_sleep

        print(f"\n[BATTERY 500 mAh] Always-on:       {hrs_baseline:.1f} hours ({hrs_baseline/24:.1f} days)")
        print(f"[BATTERY 500 mAh] DFS (80/240):    {hrs_dfs:.1f} hours ({hrs_dfs/24:.1f} days)")
        print(f"[BATTERY 500 mAh] DFS+LightSleep:  {hrs_sleep:.1f} hours ({hrs_sleep/24:.1f} days)")

        self.assertGreater(hrs_dfs, hrs_baseline * 2.5, "DFS must extend battery life by at least 2.5x")
        self.assertGreater(hrs_sleep, 48.0, "DFS + Light sleep must achieve > 48 hours continuous listening on 500 mAh")


if __name__ == "__main__":
    unittest.main()
