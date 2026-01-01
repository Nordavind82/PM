#!/usr/bin/env python3
"""
Check y_axis spacing for full vs partial tiles.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pstm.config.models import OutputGridConfig, TilingConfig
from pstm.pipeline.tile_planner import TilePlanner

GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),
    'c2': (627094.02, 5106803.16),
    'c3': (631143.35, 5110261.43),
    'c4': (622862.92, 5119956.77),
}


def main():
    grid = OutputGridConfig.from_corners(
        corner1=GRID_CORNERS['c1'],
        corner2=GRID_CORNERS['c2'],
        corner3=GRID_CORNERS['c3'],
        corner4=GRID_CORNERS['c4'],
        t_min_ms=0.0,
        t_max_ms=2000.0,
        dx=25.0,
        dy=12.5,
        dt_ms=2.0,
    )

    tiling_config = TilingConfig(auto_tile_size=False, tile_nx=128, tile_ny=128)
    planner = TilePlanner(grid, tiling_config)
    tile_plan = planner.plan()

    # Find tiles
    for t in tile_plan.tiles:
        if t.x_start == 0:
            # Create y_axis as the executor does
            y_axis = np.linspace(t.y_min, t.y_max, t.ny)

            if t.ny > 1:
                dy = y_axis[1] - y_axis[0]
            else:
                dy = 12.5

            print(f"Tile y=[{t.y_start}:{t.y_end}] (ny={t.ny}):")
            print(f"  y_min={t.y_min:.2f}, y_max={t.y_max:.2f}")
            print(f"  y_axis[0]={y_axis[0]:.2f}, y_axis[-1]={y_axis[-1]:.2f}")
            print(f"  dy={dy:.4f} (expected=12.5)")
            print()


if __name__ == "__main__":
    main()
