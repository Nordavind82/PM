"""Quality Control module for PSTM."""

from pstm.qc.analysis import (
    GeometryQCReport,
    VelocityQCReport,
    OutputQCReport,
    MigrationVerificationReport,
    compute_fold_map,
    analyze_offsets,
    analyze_azimuths,
    run_geometry_qc,
    run_velocity_qc,
    run_output_qc,
    extract_slice,
    verify_diffractor_focus,
    verify_flat_reflector_depth,
)
from pstm.qc.visualization import (
    check_matplotlib_available,
    plot_fold_map,
    plot_offset_histogram,
    plot_azimuth_rose,
    plot_velocity_profile,
    plot_velocity_slice,
    plot_image_slice,
    create_qc_report_figures,
)

__all__ = [
    # Reports
    "GeometryQCReport",
    "VelocityQCReport",
    "OutputQCReport",
    "MigrationVerificationReport",
    # Geometry QC
    "compute_fold_map",
    "analyze_offsets",
    "analyze_azimuths",
    "run_geometry_qc",
    # Velocity QC
    "run_velocity_qc",
    # Output QC
    "run_output_qc",
    "extract_slice",
    # Verification
    "verify_diffractor_focus",
    "verify_flat_reflector_depth",
    # Visualization
    "check_matplotlib_available",
    "plot_fold_map",
    "plot_offset_histogram",
    "plot_azimuth_rose",
    "plot_velocity_profile",
    "plot_velocity_slice",
    "plot_image_slice",
    "create_qc_report_figures",
]
