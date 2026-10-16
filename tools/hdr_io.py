"""Minimal Radiance RGBE (.hdr) reader/writer helpers using numpy."""
import re

import numpy as np


def read_hdr(path):
    """Reads a Radiance .hdr file and returns a float32 array of shape (H, W, 3)."""
    with open(path, 'rb') as f:
        data = f.read()
    pos = 0
    # Header lines terminated by an empty line.
    while True:
        end = data.index(b'\n', pos)
        line = data[pos:end]
        pos = end + 1
        if line.strip() == b'':
            break
    end = data.index(b'\n', pos)
    res_line = data[pos:end].decode('ascii')
    pos = end + 1
    m = re.match(r'-Y (\d+) \+X (\d+)', res_line)
    if not m:
        raise ValueError('Unsupported HDR orientation: ' + res_line)
    height, width = int(m.group(1)), int(m.group(2))
    buf = np.frombuffer(data, dtype=np.uint8, offset=pos)
    out = np.empty((height, width, 4), dtype=np.uint8)
    p = 0
    for y in range(height):
        if buf[p] == 2 and buf[p + 1] == 2 and (int(buf[p + 2]) << 8 | int(buf[p + 3])) == width:
            p += 4
            for c in range(4):
                x = 0
                row = out[y, :, c]
                while x < width:
                    count = int(buf[p])
                    p += 1
                    if count > 128:
                        count -= 128
                        row[x:x + count] = buf[p]
                        p += 1
                    else:
                        row[x:x + count] = buf[p:p + count]
                        p += count
                    x += count
        else:
            # Flat (uncompressed) scanline.
            out[y] = buf[p:p + width * 4].reshape(width, 4)
            p += width * 4
    rgbe = out.astype(np.float32)
    e = out[:, :, 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)
    return rgbe[:, :, :3] * scale[:, :, None]
