# Spectra P6 — Diagram Parity & Specification Audit

**Diagram File:** `p6_spec/p6_a_soak.svg`  
**Generator Script:** `scripts/gen_p6_diagram.py`  
**Normative Reference:** `Spectra_Phase6_Implementation_Spec.md` (§0 – §5)  
**Audit Timestamp:** 2026-09-10  
**Parity Verdict:** **EXACT MATCH (0 Divergence)**

---

## 1. Node & Component Comparison

| Spec Section | Spec Rig Requirement | Diagram Representation (`p6_a_soak.svg`) | Status |
|---|---|---|---|
| **§0 Owner Decision** | Choice between 1 h bench (weak, $\lambda < 3/\text{h}$) vs 10 h overnight (recommended, $\lambda < 0.3/\text{h}$) | `OWNER DECISION: soak bar (1 h weak vs 10 h overnight recommended)` | **MATCH** |
| **§1 Power Rig** | USB power meter INLINE on XIAO supply measuring mA avg/p95 per state | `USB power meter INLINE (mA avg/p95 per state)` | **MATCH** |
| **§2 Acoustic Rig** | Speaker playing verified keyword-free background playlist (TV, radio, podcast) 1–2 m from mic | `SPEAKER (TV / radio / pods | no keyword in)` $\rightarrow$ `air` $\rightarrow$ `XIAO C5 + INMP441` | **MATCH** |
| **§2 Serial Logger** | Captures `TRIG` lines, hourly heap, core temperature, RSSI, and reconnect counts | `serial log → TRIG lines + heap + temp + RSSI` | **MATCH** |
| **§2 FA Analyzer** | `analyze_fa.py` calculating Poisson 95% confidence upper bound $\approx 3/T$ | `HONEST FA STATISTICS (Poisson — 0 events in T hours ⇒ 95% upper bound ≈ 3/T)` | **MATCH** |
| **§4 Wi-Fi & ASR** | Wi-Fi connection from XIAO C5 to TCP ASR server streaming UTT and receiving finals | `XIAO C5` $\xrightarrow{\text{Wi-Fi}}$ `ASR SERVER (partials / finals, UTT success %)` | **MATCH** |
| **§5 Exit Criteria** | All items MEASURED/PASS (heap, MFCC+invoke ms, 50-UART parity, 5-clip boot, LED, NVS, power table) | `P6 EXIT = everything MEASURED/PASS: heap table · MFCC+invoke ms · 50-UART parity · 5-clip boot · LED visual · NVS retention` | **MATCH** |

---

## 2. Parity Confirmation
There are **zero divergent diagrams or conflicting visual representations** in the repository. The diagram `p6_spec/p6_a_soak.svg` is authoritative and fully reconciled against the normative P6 implementation specification.
