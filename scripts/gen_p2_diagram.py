#!/usr/bin/env python3
"""P2 spec diagram: mixed fixed/float MFCC data-flow with word widths + parity gate."""
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

def box(x, y, w, h, title, sub, fill, ts=13, ss=11):
    return (R(x, y, w, h, fill, rx=12) + T(x + w / 2, y + h / 2 - 6, title, ts, True, WHITE)
            + T(x + w / 2, y + h / 2 + 15, sub, ss, False, WHITE))

DEFS = ('<defs>' + ''.join(
    f'<marker id="{i}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    for i, c in [('aN', NAVY), ('aO', ORANGE), ('aT', TEAL), ('aG', GREY), ('aR', RED),
                 ('aGr', GREEN), ('aS', SLATE)]) + '</defs>')

b = [T(30, 34, 'P2 · MFCC data-flow — word widths, golden exports, parity gate (32 frames/hop)', 20, True, NAVY, 'start')]
stages = [('WINDOW', 'int16 ×16,000', NAVY2),
          ('GATE+NORM', '≥0.03 · ÷peak', NAVY2),
          ('HANN+FFT', 'int16 · /2 ×10', SLATE),
          ('POWER', 'int64 · re²+im²', SLATE),
          ('SPARSE MEL', 'f32 · 128', TEAL),
          ('LOG+CLIP', '+60.21 · top80', TEAL),
          ('DCT 40', 'f32 ortho', TEAL),
          ('QUANT', 'banker · int8', GREEN)]
xs, y0, w0, h0, gap = 14, 62, 128, 76, 22
for i, (t, s, f) in enumerate(stages):
    x = xs + i * (w0 + gap)
    b.append(box(x, y0, w0, h0, t, s, f))
    if i:
        b.append(AR(x - gap + 3, y0 + h0 / 2, x - 3, y0 + h0 / 2, ORANGE, 3))
b.append(T(600, 168, 'golden exports from Python (never recomputed in C):', 13, True, NAVY))
exp = ['Hann-1024 Q15', 'twiddle-512 Q15', 'mel 128×513 f32', 'DCT 40×128 f32']
for i, e in enumerate(exp):
    x = 90 + i * 270
    b.append(R(x, 180, 240, 40, WHITE, rx=10, stroke=AMBER, sw=2) + T(x + 120, 206, '★ ' + e, 12, True, AMBER))
b.append(R(14, 244, 1172, 150, BG, rx=12, stroke=LINE))
b.append(T(34, 272, 'SCALE DERIVATION (exact):', 13, True, NAVY, 'start'))
b.append(T(34, 298, '· FFT /2 per stage × 10 stages = ÷1024 → power ÷1024² → log-mel shifted −60.206 dB uniformly', 12.5, False, DARK, 'start'))
b.append(T(34, 322, '· compensation: logmel += +60.206 dB BEFORE top_db clip → c0 exact; c1..c39 untouched by construction', 12.5, False, DARK, 'start'))
b.append(T(34, 346, '· power in int64: 2×32767² = 2.147e9 overflows int32 — documented trap, not a guess', 12.5, False, DARK, 'start'))
b.append(T(34, 370, '· power spectrum |X|² (librosa power=2.0) · unnormalized FFT (no 1/N) · reflect-pad 512/side', 12.5, False, DARK, 'start'))
b.append(R(14, 408, 1172, 96, BG, rx=12, stroke=LINE))
b.append(T(34, 436, 'PARITY GATE (blocking): 50 golden clips over UART — max|Δ| ≤ 1e-3 · int8 bytes ≥ 99% · argmax gate moves to P3', 13, True, GREEN, 'start'))
b.append(T(34, 462, 'cycle budget: measured < 100 ms @240 MHz (soft) · memory: FFT scratch ≈8 KB internal · tables ≈290 KB flash', 12.5, False, DARK, 'start'))
b.append(T(34, 486, 'goldens record librosa/scipy/numpy versions — regenerate (never hand-edit) on env change', 12.5, False, DARK, 'start'))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 520" width="100%">'
       f'<rect x="0" y="0" width="1200" height="520" fill="white"/>{DEFS}{"".join(b)}</svg>')
os.makedirs('p2_spec', exist_ok=True)
with open('p2_spec/p2_a_precision.svg', 'w', encoding='utf-8') as f:
    f.write(svg)
print('wrote p2_spec/p2_a_precision.svg')
