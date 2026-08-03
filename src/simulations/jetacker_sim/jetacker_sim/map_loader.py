"""Parse a nav2_map_server-style map (YAML + binary PGM) into an occupancy grid.

Standalone (no rclpy dependency) so it can be unit-tested on its own.
"""
import os

import numpy as np
import yaml


def read_pgm(path):
    """Parse a binary (P5) PGM file. Returns a (h, w) uint8 numpy array."""
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        if magic != b'P5':
            raise ValueError(f'only binary PGM (P5) supported, got {magic!r}')
        tokens = []
        while len(tokens) < 3:
            line = f.readline()
            if line.startswith(b'#'):
                continue
            tokens += line.split()
        width, height, maxval = int(tokens[0]), int(tokens[1]), int(tokens[2])
        dtype = np.uint8 if maxval < 256 else np.uint16
        data = np.frombuffer(f.read(width * height * dtype().itemsize), dtype=dtype)
        return data.reshape((height, width))


def load_map(yaml_path):
    """Load a map_server-format map.

    Returns (occupied, resolution, origin_xy):
      occupied    -- (h, w) bool ndarray, True where a cell is occupied
      resolution  -- meters/pixel
      origin_xy   -- (x, y) world position of the pixel at row=h-1, col=0
                     (map yaw is assumed 0 -- true for map_01.yaml; not handled in general)
    """
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    image_path = meta['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(yaml_path), image_path)

    resolution = float(meta['resolution'])
    origin = (float(meta['origin'][0]), float(meta['origin'][1]))
    negate = int(meta.get('negate', 0))
    occupied_thresh = float(meta.get('occupied_thresh', 0.65))

    pixels = read_pgm(image_path)
    # map_server convention: pixel value 255 (white) = free, 0 (black) = occupied,
    # unless negate flips that. occ_prob in [0,1].
    if negate:
        occ_prob = pixels.astype(np.float64) / 255.0
    else:
        occ_prob = (255.0 - pixels.astype(np.float64)) / 255.0

    occupied = occ_prob > occupied_thresh
    return occupied, resolution, origin


def world_to_grid(x, y, origin, resolution, height):
    """World (x,y) meters -> (row, col) pixel indices. row 0 = top = max y."""
    col = int((x - origin[0]) / resolution)
    row = height - 1 - int((y - origin[1]) / resolution)
    return row, col
