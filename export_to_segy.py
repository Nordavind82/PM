#!/usr/bin/env python3
"""
Export migrated PSTM data to SEG-Y format.
"""

import argparse
import struct
from pathlib import Path
import numpy as np
import zarr


# Grid corners for coordinate calculation
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),  # Origin (IL=1, XL=1)
    'c2': (627094.02, 5106803.16),  # Inline end (IL=511, XL=1)
    'c3': (631143.35, 5110261.43),  # Far corner (IL=511, XL=427)
    'c4': (622862.92, 5119956.77),  # Crossline end (IL=1, XL=427)
}


def compute_cdp_coordinates(il: int, xl: int, nx: int = 511, ny: int = 427):
    """Compute CDP X, Y coordinates for a given IL, XL position."""
    # Normalized positions (0 to 1)
    il_norm = (il - 1) / (nx - 1)
    xl_norm = (xl - 1) / (ny - 1)

    # Bilinear interpolation
    c1 = np.array(GRID_CORNERS['c1'])
    c2 = np.array(GRID_CORNERS['c2'])
    c3 = np.array(GRID_CORNERS['c3'])
    c4 = np.array(GRID_CORNERS['c4'])

    # Interpolate along inline direction
    p_xl0 = c1 + il_norm * (c2 - c1)  # At XL=1
    p_xl1 = c4 + il_norm * (c3 - c4)  # At XL=max

    # Interpolate along crossline direction
    p = p_xl0 + xl_norm * (p_xl1 - p_xl0)

    return p[0], p[1]


def write_segy(data: np.ndarray, output_path: Path, dt_ms: float = 2.0):
    """
    Write 3D data to SEG-Y format.

    Args:
        data: 3D array (nx, ny, nt) - inline x crossline x time
        output_path: Output SEG-Y file path
        dt_ms: Sample interval in milliseconds
    """
    nx, ny, nt = data.shape
    dt_us = int(dt_ms * 1000)  # Convert to microseconds

    print(f"Writing SEG-Y: {nx} inlines x {ny} crosslines x {nt} samples")
    print(f"Sample interval: {dt_ms} ms")
    print(f"Output: {output_path}")

    with open(output_path, 'wb') as f:
        # Write 3200-byte textual header (EBCDIC)
        text_header = f"C01 PSTM Migration Output                                               \n"
        text_header += f"C02 Inlines: 1-{nx}, Crosslines: 1-{ny}                                  \n"
        text_header += f"C03 Samples: {nt}, Sample interval: {dt_ms}ms                            \n"
        text_header += f"C04 Data format: IEEE float (format code 5)                             \n"
        text_header += f"C05                                                                      \n"
        for i in range(6, 41):
            text_header += f"C{i:02d}                                                                      \n"

        # Pad to exactly 3200 bytes and convert to EBCDIC-like (just use ASCII for simplicity)
        text_bytes = text_header[:3200].ljust(3200).encode('ascii', errors='replace')
        f.write(text_bytes)

        # Write 400-byte binary header
        binary_header = bytearray(400)
        # Job ID (bytes 1-4)
        struct.pack_into('>i', binary_header, 0, 1)
        # Line number (bytes 5-8)
        struct.pack_into('>i', binary_header, 4, 1)
        # Reel number (bytes 9-12)
        struct.pack_into('>i', binary_header, 8, 1)
        # Traces per ensemble (bytes 13-14)
        struct.pack_into('>h', binary_header, 12, 1)
        # Aux traces per ensemble (bytes 15-16)
        struct.pack_into('>h', binary_header, 14, 0)
        # Sample interval in microseconds (bytes 17-18)
        struct.pack_into('>H', binary_header, 16, dt_us)
        # Original sample interval (bytes 19-20)
        struct.pack_into('>H', binary_header, 18, dt_us)
        # Samples per trace (bytes 21-22)
        struct.pack_into('>H', binary_header, 20, nt)
        # Original samples per trace (bytes 23-24)
        struct.pack_into('>H', binary_header, 22, nt)
        # Data format code (bytes 25-26): 5 = IEEE float
        struct.pack_into('>h', binary_header, 24, 5)
        # Ensemble fold (bytes 27-28)
        struct.pack_into('>h', binary_header, 26, 1)
        # Trace sorting code (bytes 29-30): 4 = horizontally stacked
        struct.pack_into('>h', binary_header, 28, 4)
        # Measurement system (bytes 55-56): 1 = meters
        struct.pack_into('>h', binary_header, 54, 1)
        # SEG-Y revision (bytes 301-302): 1 = rev 1
        struct.pack_into('>H', binary_header, 300, 256)  # 0x0100 = rev 1.0
        # Fixed trace length flag (bytes 303-304): 1 = fixed
        struct.pack_into('>h', binary_header, 302, 1)
        # Number of extended textual headers (bytes 305-306)
        struct.pack_into('>h', binary_header, 304, 0)

        f.write(binary_header)

        # Write traces
        total_traces = nx * ny
        trace_count = 0

        for il in range(1, nx + 1):
            for xl in range(1, ny + 1):
                # Get trace data
                trace_data = data[il - 1, xl - 1, :].astype(np.float32)

                # Compute CDP coordinates
                cdp_x, cdp_y = compute_cdp_coordinates(il, xl, nx, ny)

                # Create 240-byte trace header
                trace_header = bytearray(240)

                # Trace sequence number in line (bytes 1-4)
                struct.pack_into('>i', trace_header, 0, trace_count + 1)
                # Trace sequence number in file (bytes 5-8)
                struct.pack_into('>i', trace_header, 4, trace_count + 1)
                # Original field record number (bytes 9-12)
                struct.pack_into('>i', trace_header, 8, il)
                # Trace number within original field record (bytes 13-16)
                struct.pack_into('>i', trace_header, 12, xl)
                # CDP ensemble number (bytes 21-24)
                struct.pack_into('>i', trace_header, 20, trace_count + 1)
                # Trace number within CDP (bytes 25-28)
                struct.pack_into('>i', trace_header, 24, 1)
                # Trace ID code (bytes 29-30): 1 = seismic
                struct.pack_into('>h', trace_header, 28, 1)

                # CDP X (bytes 181-184)
                struct.pack_into('>i', trace_header, 180, int(cdp_x))
                # CDP Y (bytes 185-188)
                struct.pack_into('>i', trace_header, 184, int(cdp_y))

                # Inline number (bytes 189-192)
                struct.pack_into('>i', trace_header, 188, il)
                # Crossline number (bytes 193-196)
                struct.pack_into('>i', trace_header, 192, xl)

                # Coordinate scalar (bytes 71-72): 1 = no scaling
                struct.pack_into('>h', trace_header, 70, 1)

                # Number of samples (bytes 115-116)
                struct.pack_into('>H', trace_header, 114, nt)
                # Sample interval in microseconds (bytes 117-118)
                struct.pack_into('>H', trace_header, 116, dt_us)

                # Write trace header
                f.write(trace_header)

                # Write trace data (big-endian IEEE float)
                f.write(trace_data.astype('>f4').tobytes())

                trace_count += 1

                if trace_count % 10000 == 0:
                    print(f"  Written {trace_count}/{total_traces} traces ({100*trace_count/total_traces:.1f}%)")

        print(f"  Written {trace_count}/{total_traces} traces (100%)")

    print(f"SEG-Y export complete: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e9:.2f} GB")


def export_all_bins(input_dir: Path, output_dir: Path, dt_ms: float = 2.0):
    """Export all migrated bins to SEG-Y."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all migration bins
    bin_dirs = sorted(input_dir.glob("migration_bin_*"))
    print(f"Found {len(bin_dirs)} migration bins to export")

    for bin_dir in bin_dirs:
        zarr_path = bin_dir / "migrated_stack.zarr"
        if not zarr_path.exists():
            print(f"Skipping {bin_dir.name}: no migrated_stack.zarr")
            continue

        bin_num = bin_dir.name.replace("migration_bin_", "")
        output_path = output_dir / f"pstm_bin_{bin_num}.sgy"

        print(f"\n{'='*60}")
        print(f"Exporting {bin_dir.name}")
        print(f"{'='*60}")

        # Load data
        zarr_data = zarr.open(str(zarr_path), mode='r')
        data = np.array(zarr_data)

        # Export to SEG-Y
        write_segy(data, output_path, dt_ms)

    print(f"\n{'='*60}")
    print(f"All exports complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Export migrated PSTM data to SEG-Y")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/olegadamovich/SeismicData/PSTM_common_offset"),
        help="Input directory with migration bins",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/segy_export"),
        help="Output directory for SEG-Y files",
    )
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=2.0,
        help="Sample interval in milliseconds",
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=None,
        help="Export only specific bin number",
    )

    args = parser.parse_args()

    if args.bin is not None:
        # Export single bin
        bin_dir = args.input_dir / f"migration_bin_{args.bin:02d}"
        zarr_path = bin_dir / "migrated_stack.zarr"
        output_path = args.output_dir / f"pstm_bin_{args.bin:02d}.sgy"
        args.output_dir.mkdir(parents=True, exist_ok=True)

        zarr_data = zarr.open(str(zarr_path), mode='r')
        data = np.array(zarr_data)
        write_segy(data, output_path, args.dt_ms)
    else:
        # Export all bins
        export_all_bins(args.input_dir, args.output_dir, args.dt_ms)


if __name__ == "__main__":
    main()
