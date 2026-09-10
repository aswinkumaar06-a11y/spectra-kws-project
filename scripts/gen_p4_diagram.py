#!/usr/bin/env python3
"""
P4 spec diagram: Confidence Trigger State Machine, Streams A-D & LED Timing.
Generates p4_spec/p4_a_streams.svg
"""
import os

NAVY, NAVY2 = '#0B1F3A', '#163866'
ORANGE, TEAL, GREEN, RED, GREY, AMBER = '#FF6B1A', '#009E8A', '#0E7C3E', '#B91C1C', '#5E6B7D', '#B45309'
SLATE, BG, DARK, WHITE, LINE = '#0E7490', '#EAF0F7', '#1A1A1A', '#FFFFFF', '#C9D3E3'
FONT = 'Segoe UI, Arial, Helvetica, sans-serif'

def T(x, y, s, size=14, bold=False, color=DARK, anchor='middle'):
    b = ' font-weight="bold"' if bold else ''
    s = str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}"{b} fill="{color}" text-anchor="{anchor}">{s}</text>'

def R(x, y, w, h, fill, rx=8, stroke='none', sw=2):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke != 'none' else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}/>'

def AR(x1, y1, x2, y2, color=ORANGE, w=3, dashed=False):
    d = ' stroke-dasharray="7 5"' if dashed else ''
    m = {'#0B1F3A': 'aN', '#163866': 'aN', '#FF6B1A': 'aO', '#009E8A': 'aT',
         '#5E6B7D': 'aG', '#B91C1C': 'aR', '#0E7C3E': 'aGr', '#0E7490': 'aS'}[color]
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"{d} marker-end="url(#{m})"/>'

def box(x, y, w, h, title, sub, fill, ts=12.5, ss=10.5):
    return (R(x, y, w, h, fill, rx=12) +
            T(x + w / 2, y + h / 2 - 6, title, ts, True, WHITE) +
            T(x + w / 2, y + h / 2 + 14, sub, ss, False, WHITE))

DEFS = ('<defs>' + ''.join(
    f'<marker id="{i}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    for i, c in [('aN', NAVY), ('aO', ORANGE), ('aT', TEAL), ('aG', GREY), ('aR', RED),
                 ('aGr', GREEN), ('aS', SLATE)]) + '</defs>')

b = []
# Title
b.append(T(30, 36, 'P4 · Confidence Trigger & LED Timing — State Machine & Test Streams A–D', 20, True, NAVY, 'start'))

# State Machine Diagram
states = [
    ('LISTEN', 'Dark · p ≤ 0.5', NAVY2),
    ('CANDIDATE', 'Dark · Count 1..2', SLATE),
    ('TRIGGERED', 'FIRE! · LED ON', GREEN),
    ('COOLDOWN', 'LED ON · 4 ticks (2s)', AMBER),
]

xs, y0, w0, h0, gap = 50, 68, 220, 76, 70
for i, (t, s, f) in enumerate(states):
    x = xs + i * (w0 + gap)
    b.append(box(x, y0, w0, h0, t, s, f))
    if i < len(states) - 1:
        lbl = ['p > 0.5 (count=1)', 'p > 0.5 (count=3)', 'tick fire actions'][i]
        b.append(AR(x + w0 + 4, y0 + h0 / 2, x + w0 + gap - 4, y0 + h0 / 2, ORANGE, 3))
        b.append(T(x + w0 + gap / 2, y0 + h0 / 2 - 12, lbl, 10.5, True, DARK))

# Cooldown to LISTEN return arrow
b.append(AR(xs + 3 * (w0 + gap) + w0 / 2, y0 + h0, xs + w0 / 2, y0 + h0, TEAL, 3, dashed=True))
b.append(T(600, y0 + h0 + 20, 'ticks_left == 0 (LED OFF → Return to LISTEN fresh)', 11, True, TEAL))

# Test Streams Table Box
b.append(R(14, 200, 1172, 180, BG, rx=12, stroke=LINE))
b.append(T(34, 226, 'NORMATIVE TEST STREAMS (Discrete 500 ms Hops, P_THRESH = 0.50):', 13, True, NAVY, 'start'))

streams = [
    ("Stream A (Clean):", "[.10, .20, .60, .70, .80, .30, .10]", "Fire @ Tick 4", "LED ON@4, Cooldown covers ticks 5-6"),
    ("Stream B (Flicker):", "[.60, .40, .70, .40, .80, .90, .95]", "Fire @ Tick 6", "Expire drops below 0.5 twice, resets count to 0"),
    ("Stream C (Cooldown):", "[.90, .90, .90, .90, .90, .90, .90, .10]", "Fire @ Tick 2", "Ticks 3–6 suppressed during 4-tick cooldown despite p=0.9"),
    ("Stream D (Double):", "[.85, .90, .95, .10, .10, .10, .10, .80, .85, .90]", "Fire @ Ticks [2, 9]", "LED ON@2, OFF@7, ON@9; re-fire after fresh LISTEN"),
]

for idx, (name, vec, events, note) in enumerate(streams):
    sy = 254 + idx * 30
    b.append(T(34, sy, name, 11.5, True, DARK, 'start'))
    b.append(T(200, sy, vec, 11, False, SLATE, 'start'))
    b.append(T(620, sy, events, 11.5, True, GREEN, 'start'))
    b.append(T(760, sy, note, 11, False, DARK, 'start'))

# Fire Actions Box
b.append(R(14, 396, 1172, 126, BG, rx=12, stroke=LINE))
b.append(T(34, 422, 'FIRE ACTIONS & NORMATIVE PROTOCOL (Binding Rules):', 13, True, NAVY, 'start'))
b.append(T(34, 446, '1. LED Control: ON at fire tick (GPIO 27 active HIGH), persists through cooldown end, OFF otherwise', 12, False, DARK, 'start'))
b.append(T(34, 468, '2. Normative UART Format: TRIG tick=%u t_ms=%lu p=[%.3f,%.3f,%.3f] pre_ts=%lu (exact float format)', 12, False, DARK, 'start'))
b.append(T(34, 490, '3. NVS Persistence: trigger_count++ & last_trigger_ts stored in "spectra" namespace (honesty: raw triggers, no FA claim)', 12, False, DARK, 'start'))
b.append(T(34, 510, '4. Pre-Roll Snapshot: 16,000 INT16 samples (32 KB in internal SRAM) copied without stalling live capture', 12, False, GREEN, 'start'))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 536" width="100%">'
       f'<rect x="0" y="0" width="1200" height="536" fill="white"/>{DEFS}{"".join(b)}</svg>')

os.makedirs('p4_spec', exist_ok=True)
out_svg = os.path.join('p4_spec', 'p4_a_streams.svg')
with open(out_svg, 'w', encoding='utf-8') as f:
    f.write(svg)
print(f'Wrote {out_svg}')
