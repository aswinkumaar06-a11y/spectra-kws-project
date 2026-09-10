#!/usr/bin/env python3
"""P6 spec diagram: soak + power + FA rig with honest statistics."""
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

DEFS = ('<defs>' + ''.join(
    f'<marker id="{i}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    for i, c in [('aN', NAVY), ('aO', ORANGE), ('aT', TEAL), ('aG', GREY), ('aR', RED),
                 ('aGr', GREEN), ('aS', SLATE)]) + '</defs>')

b = [T(30, 34, 'P6 · Harden rig — soak, power, FA-rate (first all-hardware phase: no host half)', 20, True, NAVY, 'start')]
b.append(R(30, 60, 250, 120, NAVY2, rx=12) + T(155, 98, 'LAPTOP', 15, True, WHITE)
         + T(155, 122, 'playlist + logger', 12, False, WHITE) + T(155, 144, 'FA analyzer', 12, False, WHITE))
b.append(R(330, 60, 180, 120, SLATE, rx=12) + T(420, 98, 'SPEAKER', 15, True, WHITE)
         + T(420, 122, 'TV / radio / pods', 12, False, WHITE) + T(420, 144, 'no keyword in', 12, False, WHITE))
b.append(R(560, 60, 280, 120, TEAL, rx=12) + T(700, 92, 'XIAO C5 + INMP441', 15, True, WHITE)
         + T(700, 114, 'full pipeline + Wi-Fi', 12, False, WHITE) + T(700, 136, 'hourly heap · temp', 12, False, WHITE)
         + T(700, 158, 'RSSI · reconnects', 12, False, WHITE))
b.append(R(890, 60, 280, 120, GREEN, rx=12) + T(1030, 98, 'ASR SERVER', 15, True, WHITE)
         + T(1030, 122, 'partials / finals', 12, False, WHITE) + T(1030, 144, 'UTT success %', 12, False, WHITE))
b.append(AR(280, 120, 330, 120, NAVY2, 3) + T(305, 108, 'audio', 11, False, GREY))
b.append(AR(510, 120, 560, 120, NAVY2, 3) + T(535, 108, 'air', 11, False, GREY))
b.append(AR(840, 100, 890, 100, ORANGE, 3) + T(865, 88, 'Wi-Fi', 11, False, ORANGE))
b.append(AR(700, 180, 700, 208, GREY, 2.5, dashed=True))
b.append(R(480, 214, 440, 60, WHITE, rx=10, stroke=AMBER, sw=2)
         + T(700, 238, 'USB power meter INLINE (mA avg/p95 per state)', 12, True, AMBER)
         + T(700, 260, 'serial log → TRIG lines + heap + temp + RSSI', 12, False, DARK))
b.append(R(30, 214, 400, 60, WHITE, rx=10, stroke=LINE, sw=2)
         + T(230, 238, 'OWNER DECISION: soak bar', 12, True, NAVY)
         + T(230, 260, '1 h (weak) vs 10 h overnight (recommended)', 12, False, DARK))
b.append(R(30, 294, 1140, 130, BG, rx=12, stroke=LINE))
b.append(T(50, 322, 'HONEST FA STATISTICS (Poisson — 0 events in T hours ⇒ 95% upper bound ≈ 3/T):', 13, True, NAVY, 'start'))
b.append(T(50, 350, '· 0 FAs in 1 h ⇒ rate < ~3/h (WEAK — cannot honestly claim <1/h) · 0 FAs in 10 h ⇒ rate < ~0.3/h (CLAIM <1/h with margin)', 12.5, False, DARK, 'start'))
b.append(T(50, 376, '· every trigger on keyword-free audio = FA · recall: 20 owner-voice reps at 1 m / 3 m (small-n, labeled honestly)', 12.5, False, DARK, 'start'))
b.append(T(50, 402, '· VAD TH from measured noise floor (p99 × 4, ≥ 0.01) · failure matrix: Wi-Fi drop / server down / mic unplug / heap-low — each with expected behavior', 12, True, GREEN, 'start'))
b.append(R(30, 438, 1140, 62, GREEN, rx=12)
         + T(600, 464, 'P6 EXIT = everything MEASURED/PASS: heap table · MFCC+invoke ms · 50-UART parity · 5-clip boot · LED visual · NVS retention', 13, True, WHITE)
         + T(600, 488, 'P6 closes S3, T1-measurements, T3-measurements and the P4/P5 device halves — the show-me phase', 12, False, WHITE))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 514" width="100%">'
       f'<rect x="0" y="0" width="1200" height="514" fill="white"/>{DEFS}{"".join(b)}</svg>')

os.makedirs('p6_spec', exist_ok=True)
with open('p6_spec/p6_a_soak.svg', 'w', encoding='utf-8') as f:
    f.write(svg)
print('wrote p6_spec/p6_a_soak.svg')
