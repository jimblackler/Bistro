#!/usr/bin/env python3
"""Combines screenshots into a labelled contact sheet. Usage: contact_sheet.py OUT.png IN1.png IN2.png ..."""
import os
import sys

from PIL import Image, ImageDraw

out, files = sys.argv[1], sys.argv[2:]
cols = 3
w, h = 800, 356
rows = (len(files) + cols - 1) // cols
sheet = Image.new('RGB', (cols * w, rows * h), (0, 0, 0))
d = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    im = Image.open(f).convert('RGB').resize((w, h))
    x, y = (i % cols) * w, (i // cols) * h
    sheet.paste(im, (x, y))
    d.rectangle([x, y, x + 110, y + 22], fill=(0, 0, 0))
    d.text((x + 4, y + 4), os.path.basename(f)[:-4], fill=(255, 255, 0))
sheet.save(out)
