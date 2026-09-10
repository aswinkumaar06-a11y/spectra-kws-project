#!/usr/bin/env python3
"""
P3 spec diagram: TFLM Inference, Minimal Resolver <5>, Memory Discipline & Parity Gate.
Generates p3_spec/p3_a_bringup.svg
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
b.append(T(30, 36, 'P3 · TFLM Inference Pipeline — Minimal Resolver <5>, Internal SRAM & Logits Parity', 20, True, NAVY, 'start'))

# Pipeline stages
stages = [
    ('MODEL FLASH', '39,552 B · TFL3', NAVY2),
    ('RESOLVER <5>', '5 unique opcodes', SLATE),
    ('ARENA .BSS', '80 KB · 16B align', SLATE),
    ('FEED INPUT', 'int8 [1,40,32,1]', TEAL),
    ('INVOKE TFLM', 'Cycle Counted', TEAL),
    ('DEQUANT', 'int8 → [p0, p1]', TEAL),
    ('PARITY GATE', 'Δ ≤ 0.02 · 100%', GREEN),
]

xs, y0, w0, h0, gap = 14, 66, 146, 76, 22
for i, (t, s, f) in enumerate(stages):
    x = xs + i * (w0 + gap)
    b.append(box(x, y0, w0, h0, t, s, f))
    if i:
        b.append(AR(x - gap + 3, y0 + h0 / 2, x - 3, y0 + h0 / 2, ORANGE, 3))

# Sub-badges: Minimal Resolver 5 ops
b.append(T(600, 172, 'Verified 5 Opcodes in spectra_model.tflite (10 layers total):', 13, True, NAVY))
ops = ['CONV_2D (4x)', 'DEPTHWISE_CONV_2D (3x)', 'MEAN (1x)', 'FULLY_CONNECTED (1x)', 'SOFTMAX (1x)']
for i, op in enumerate(ops):
    x = 14 + i * 236
    b.append(R(x, 186, 224, 38, WHITE, rx=8, stroke=TEAL, sw=2) + T(x + 112, 210, op, 11.5, True, TEAL))

# Memory discipline callout box
b.append(R(14, 244, 1172, 126, BG, rx=12, stroke=LINE))
b.append(T(34, 270, 'STRICT MEMORY DISCIPLINE (ESP32-C5 Internal SRAM < 256 KB):', 13, True, NAVY, 'start'))
b.append(T(34, 294, '• Tensor Arena (80 KB): Statically placed in internal SRAM .bss (__attribute__((aligned(16), section(".bss"))))', 12, False, DARK, 'start'))
b.append(T(34, 316, '• NEVER PSRAM: Inference tensors in PSRAM cause severe DMA stall & CPU cache thrashing on micro-controllers', 12, False, RED, 'start'))
b.append(T(34, 338, '• Internal SRAM Budget: Arena (80 KB) + MFCC scratch (~87 KB) + Tasks & Stacks (~16 KB) = ~183 KB (< 256 KB target)', 12, False, GREEN, 'start'))
b.append(T(34, 358, '• Recording Allocator: Evaluates used_bytes on first invoke; finalizes arena headroom to used + 10% + 2 KB guard', 12, False, DARK, 'start'))

# Mathematical Contract & Parity Gates Box
b.append(R(14, 384, 1172, 134, BG, rx=12, stroke=LINE))
b.append(T(34, 410, 'MATHEMATICAL CONTRACT & PARITY CRITERIA:', 13, True, NAVY, 'start'))
b.append(T(34, 434, '• Input Dequant:   q_in = clamp(round(feature / 3.9962144) + 60, -128, 127) contiguous NHWC format', 12, False, DARK, 'start'))
b.append(T(34, 456, '• Output Dequant:  p_neg = (q0 + 128) / 256.0f,  p_pos = (q1 + 128) / 256.0f  (scale = 1/256, zp = -128)', 12, False, DARK, 'start'))
b.append(T(34, 478, '• Zeros Smoke Anchor: Input zeros → q = [127, -128] → p_neg = 0.996094 (±0.01), p_pos = 0.000000 (< 0.01)', 12, False, DARK, 'start'))
b.append(T(34, 500, '• 5-Clip Boot Self-Test: Minimal vs Reference bit-identical (Δ = 0.0000) on 2 pos, 1 neg, 2 hard neg clips', 12, False, GREEN, 'start'))

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 536" width="100%">'
       f'<rect x="0" y="0" width="1200" height="536" fill="white"/>{DEFS}{"".join(b)}</svg>')

os.makedirs('p3_spec', exist_ok=True)
out_svg = os.path.join('p3_spec', 'p3_a_bringup.svg')
with open(out_svg, 'w', encoding='utf-8') as f:
    f.write(svg)
print(f'Wrote {out_svg}')
