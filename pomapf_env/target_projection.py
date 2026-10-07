import numpy as np


def get_square_target(x, y, tx, ty, obs_radius):
    full_size = obs_radius * 2 + 1
    result = np.zeros((full_size, full_size), dtype=np.float32)
    dx = int(round(float(x) - float(tx)))
    dy = int(round(float(y) - float(ty)))
    dx = min(dx, obs_radius) if dx >= 0 else max(dx, -obs_radius)
    dy = min(dy, obs_radius) if dy >= 0 else max(dy, -obs_radius)
    result[obs_radius - dx, obs_radius - dy] = 1.0
    return result
