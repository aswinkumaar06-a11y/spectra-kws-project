#!/usr/bin/env python3
"""P5 spec diagram: endpoint-VAD state machine + deterministic energy vectors."""
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

def AR(x1, y1, x2, y2, color=ORANGE, w=3):
    m = {'#0B1F3A': 'aN', '#163866': 'aN', '#FF6B1A': 'aO', '#009E8A': 'aT',
         '#5E6B7D': 'aG', '#B91C1C': 'aR', '#0E7C3E': 'aGr', '#0E7490': 'aS'}[color]
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" marker-end="url(#{m})"/>'

DEFS = ('<defs>' + ''.join(
    f'<marker id="{i}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    for i, c in [('aN', NAVY), ('aO', ORANGE), ('aT', TEAL), ('aG', GREY), ('aR', RED),
                 ('aGr', GREEN), ('aS', SLATE)]) + '</defs>')

b = [T(30, 34, 'P5 · Endpoint VAD — state machine + deterministic vectors (frame = 100 ms RMS)', 20, True, NAVY, 'start')]

# state machine row
states = [('IDLE', 'wait speech', NAVY2), ('SPEECH', 'in utterance', TEAL), ('END', 'endpoint!', GREEN)]
for i, (t, s, f) in enumerate(states):
    x = 200 + i * 300
    b.append(R(x, 56, 220, 64, f, rx=12) + T(x + 110, 82, t, 15, True, WHITE) + T(x + 110, 104, s, 12, False, WHITE))
    if i:
        b.append(AR(x - 80 + 4, 88, x - 4, 88, ORANGE, 3))

b.append(T(350, 78, 'E>TH ×2', 11, False, ORANGE) + T(650, 78, 'E<TH ×3', 11, False, ORANGE))
b.append(T(600, 140, 'blip (< 2 frames) never leaves IDLE · hangover absorbs ≤ 2-frame pauses · END needs ≥ 3-frame utterance', 12, False, GREY))

vectors = [
    ('V1 · burst', [.005, .008, .05, .09, .12, .10, .004, .003, .002, .001], 2, 8, 'START@2 END@8'),
    ('V2 · blip', [.005, .06, .004, .003, .002, .001], None, None, 'no start (single frame)'),
    ('V3 · pause', [.06, .07, .004, .005, .08, .09, .10, .003, .002, .001], 0, 9, 'START@0 END@9 · pause absorbed'),
]

y = 158
TH = 0.02
for name, vals, st, en, note in vectors:
    n = len(vals)
    x0, bw, gap, base = 210, min(64, 640 // n - 8), 8, y + 56
    b.append(T(200, y + 30, name, 13, True, NAVY, 'end'))
    vmax = max(vals)
    for i, v in enumerate(vals):
        x = x0 + i * (bw + gap)
        h = max(3, v / vmax * 50)
        b.append(R(x, base - h, bw, h, TEAL if v > TH else '#B9C4D4', rx=5))
        b.append(T(x + bw / 2, base + 15, f'{v:g}', 10, False, GREY))
        if i == st:
            b.append(T(x + bw / 2, base - h - 8, '▼ START', 11, True, GREEN))
        if i == en:
            b.append(T(x + bw / 2, base - h - 8, '■ END', 11, True, RED))
    th_y = base - TH / vmax * 50
    b.append(f'<line x1="{x0 - 10}" y1="{th_y}" x2="{x0 + n * (bw + gap)}" y2="{th_y}" stroke="{RED}" stroke-width="1.5" stroke-dasharray="6 4"/>')
    b.append(T(x0 + n * (bw + gap) + 8, th_y + 4, 'TH 0.02', 11, True, RED, 'start'))
    b.append(T(1190, y + 30, note, 11.5, False, AMBER, 'end'))
    y += 92

b.append(R(14, y + 4, 1172, 58, GREEN, rx=12)
         + T(600, y + 29, 'HOST GATE (blocking): START/END indices exact on V1–V3 · TH tunable via Kconfig (P6 calibrates from mic data)', 13, True, WHITE)
         + T(600, y + 50, 'TH default 0.02 RMS is a placeholder with honest provenance — logic proven here, value tuned on hardware', 12, False, WHITE))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {y + 76}" width="100%">'
       f'<rect x="0" y="0" width="1200" height="{y + 76}" fill="white"/>{DEFS}{"".join(b)}</svg>')

os.makedirs('p5_spec', exist_ok=True)
with open('p5_spec/p5_a_vad.svg', 'w', encoding='utf-8') as f:
    f.write(svg)
print('wrote p5_spec/p5_a_vad.svg')