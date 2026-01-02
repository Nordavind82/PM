"""Velocity model I/O functions."""

from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import zarr


def load_velocity_model(path: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load velocity model from zarr or parquet file.

    Returns:
        Tuple of (velocity_data, metadata)
        - velocity_data: numpy array (can be 1D, 2D, or 3D)
        - metadata: dict with coordinate info (t_coords, il_coords, xl_coords)
    """
    path = Path(path)
    metadata = {}

    try:
        if path.suffix == '.parquet':
            import pandas as pd
            df = pd.read_parquet(path)
            # Check for different column formats
            if 'time_ms' in df.columns and 'velocity_ms' in df.columns:
                # 1D velocity function: (time, velocity)
                metadata['t_coords'] = df['time_ms'].values
                return df['velocity_ms'].values, metadata
            elif 'IL' in df.columns and 'XL' in df.columns:
                # 3D velocity: pivot to cube
                if 'time_ms' in df.columns:
                    metadata['t_coords'] = df['time_ms'].unique()
                return df, metadata  # Return DataFrame for 3D handling
            else:
                return df.values, metadata

        elif path.suffix == '.zarr' or path.is_dir():
            z = zarr.open(str(path), mode='r')

            # Try to get velocity array
            if isinstance(z, zarr.Array):
                vel_data = z
            elif hasattr(z, 'keys'):
                # It's a Group - look for velocity arrays
                if 'velocity' in z:
                    vel_data = z['velocity']
                elif 'vrms' in z:
                    vel_data = z['vrms']
                elif 'vint' in z:
                    vel_data = z['vint']
                else:
                    # Try first array found
                    for key in z.keys():
                        if isinstance(z[key], zarr.Array):
                            vel_data = z[key]
                            break
                    else:
                        vel_data = z
            else:
                vel_data = z

            # Load metadata/coordinates from attrs
            if hasattr(vel_data, 'attrs'):
                attrs = dict(vel_data.attrs)

                # Try various attribute names for time coordinates
                for t_key in ['t_coords', 't_axis_ms', 't', 'time']:
                    if t_key in attrs:
                        metadata['t_coords'] = np.array(attrs[t_key])
                        break

                # Try various attribute names for IL coordinates
                for il_key in ['il_coords', 'x_axis', 'x', 'inline']:
                    if il_key in attrs:
                        metadata['il_coords'] = np.array(attrs[il_key])
                        break

                # Try various attribute names for XL coordinates
                for xl_key in ['xl_coords', 'y_axis', 'y', 'crossline']:
                    if xl_key in attrs:
                        metadata['xl_coords'] = np.array(attrs[xl_key])
                        break

                if 'dt_ms' in attrs:
                    metadata['dt_ms'] = float(attrs['dt_ms'])

            # Check for coordinate arrays in zarr Group (only if it's a Group)
            if hasattr(z, 'keys'):
                if 'time' in z:
                    metadata['t_coords'] = np.asarray(z['time'])
                if 'inline' in z:
                    metadata['il_coords'] = np.asarray(z['inline'])
                if 'crossline' in z:
                    metadata['xl_coords'] = np.asarray(z['crossline'])

            return np.asarray(vel_data), metadata

        return None, None

    except Exception as e:
        print(f"Error loading velocity: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_velocity_function(vel_model: np.ndarray, il: int, xl: int,
                               t_coords: np.ndarray,
                               il_coords: np.ndarray = None,
                               xl_coords: np.ndarray = None) -> np.ndarray:
    """
    Extract 1D velocity function at given IL/XL position.

    Args:
        vel_model: Velocity model (1D, 2D, or 3D)
        il: Inline coordinate (not index!)
        xl: Crossline coordinate (not index!)
        t_coords: Time coordinates
        il_coords: Array of IL coordinates in the velocity model
        xl_coords: Array of XL coordinates in the velocity model

    Returns:
        1D velocity array (n_time,)
    """
    if vel_model is None:
        return None

    if vel_model.ndim == 1:
        # 1D velocity - same for all positions
        return vel_model
    elif vel_model.ndim == 2:
        # 2D velocity (IL, time) or (time, velocity pairs)
        if vel_model.shape[0] == len(t_coords):
            return vel_model[:, 0] if vel_model.shape[1] > 1 else vel_model.flatten()
        else:
            # Assume (IL, time) - find nearest IL index
            if il_coords is not None:
                il_idx = np.argmin(np.abs(il_coords - il))
            else:
                il_idx = min(il, vel_model.shape[0] - 1)
            return vel_model[il_idx, :]
    elif vel_model.ndim == 3:
        # 3D velocity (IL, XL, time) - find nearest indices
        if il_coords is not None:
            il_idx = np.argmin(np.abs(il_coords - il))
        else:
            il_idx = min(il, vel_model.shape[0] - 1)

        if xl_coords is not None:
            xl_idx = np.argmin(np.abs(xl_coords - xl))
        else:
            xl_idx = min(xl, vel_model.shape[1] - 1)

        return vel_model[il_idx, xl_idx, :]

    return None
