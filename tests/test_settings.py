"""Tests for PSTM settings module."""

import json
import tempfile
from pathlib import Path

import pytest

from pstm.settings import (
    ApplicationSettings,
    GridSettings,
    VelocitySettings,
    ApertureSettings,
    SettingsManager,
    get_settings,
    load_settings,
    save_settings,
    reset_settings,
    generate_default_settings_file,
    generate_toml_with_comments,
)


class TestGridSettings:
    def test_defaults(self):
        s = GridSettings()
        assert s.dx_m == 25.0
        assert s.dy_m == 25.0
        assert s.dt_ms == 2.0

    def test_custom_values(self):
        s = GridSettings(dx_m=50.0, dy_m=100.0, dt_ms=4.0)
        assert s.dx_m == 50.0
        assert s.dy_m == 100.0
        assert s.dt_ms == 4.0


class TestVelocitySettings:
    def test_defaults(self):
        s = VelocitySettings()
        assert s.min_velocity_ms == 500.0
        assert s.max_velocity_ms == 10000.0
        assert s.default_constant_velocity_ms == 2000.0

    def test_qc_range(self):
        s = VelocitySettings()
        assert s.qc_min_velocity_ms < s.qc_max_velocity_ms


class TestApertureSettings:
    def test_defaults(self):
        s = ApertureSettings()
        assert s.min_aperture_m < s.max_aperture_m
        assert s.max_dip_degrees == 45.0
        assert 0 < s.taper_fraction < 1


class TestApplicationSettings:
    def test_all_sections_present(self):
        s = ApplicationSettings()
        assert hasattr(s, 'grid')
        assert hasattr(s, 'velocity')
        assert hasattr(s, 'aperture')
        assert hasattr(s, 'kernel')
        assert hasattr(s, 'tiling')
        assert hasattr(s, 'checkpoint')
        assert hasattr(s, 'io')
        assert hasattr(s, 'qc')
        assert hasattr(s, 'cig')
        assert hasattr(s, 'profiling')
        assert hasattr(s, 'ui')
        assert hasattr(s, 'units')

    def test_to_dict(self):
        s = ApplicationSettings()
        d = s.to_dict()
        
        assert isinstance(d, dict)
        assert 'grid' in d
        assert 'velocity' in d
        assert d['grid']['dx_m'] == 25.0

    def test_from_dict(self):
        data = {
            'grid': {'dx_m': 50.0},
            'velocity': {'min_velocity_ms': 1000.0},
        }
        s = ApplicationSettings.from_dict(data)
        
        assert s.grid.dx_m == 50.0
        assert s.velocity.min_velocity_ms == 1000.0
        # Other values should be defaults
        assert s.grid.dy_m == 25.0


class TestSettingsManager:
    def test_singleton(self):
        m1 = SettingsManager()
        m2 = SettingsManager()
        assert m1 is m2

    def test_reset(self):
        m = SettingsManager()
        m.settings.grid.dx_m = 100.0
        m.reset()
        assert m.settings.grid.dx_m == 25.0

    def test_update_nested(self):
        m = SettingsManager()
        m.reset()
        m.update(**{'grid.dx_m': 75.0})
        assert m.settings.grid.dx_m == 75.0


class TestFileOperations:
    def test_save_and_load_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            
            # Modify and save
            s = get_settings()
            original_dx = s.grid.dx_m
            s.grid.dx_m = 123.0
            save_settings(path)
            
            # Reset and load
            reset_settings()
            assert get_settings().grid.dx_m == 25.0  # Default
            
            load_settings(path)
            assert get_settings().grid.dx_m == 123.0
            
            # Restore
            reset_settings()

    def test_generate_default_toml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.toml"
            generate_default_settings_file(path, 'toml')
            
            assert path.exists()
            content = path.read_text()
            
            # Check it has expected content
            assert '[grid]' in content
            assert 'dx_m' in content
            assert '[velocity]' in content

    def test_generate_default_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            generate_default_settings_file(path, 'json')
            
            assert path.exists()
            data = json.loads(path.read_text())
            
            assert 'grid' in data
            assert 'velocity' in data


class TestModuleLevelAccess:
    def test_get_settings(self):
        s = get_settings()
        assert isinstance(s, ApplicationSettings)

    def test_settings_singleton(self):
        reset_settings()
        s1 = get_settings()
        s2 = get_settings()
        
        # Both should reference same object
        s1.grid.dx_m = 999.0
        assert s2.grid.dx_m == 999.0
        
        reset_settings()


class TestIntegrationWithModules:
    """Test that settings are properly used by other modules."""
    
    def test_kernel_config_uses_settings(self):
        from pstm.kernels.base import KernelConfig
        
        # Modify settings
        s = get_settings()
        s.aperture.max_aperture_m = 7000.0
        
        # Create kernel config (should use settings)
        config = KernelConfig()
        assert config.max_aperture_m == 7000.0
        
        # Reset
        reset_settings()

    def test_tile_planner_uses_settings(self):
        from pstm.pipeline.tile_planner import TilePlanner
        from pstm.config.models import OutputGridConfig, TilingConfig
        
        # Modify settings
        s = get_settings()
        s.tiling.max_memory_gb = 16.0
        
        # Create planner without explicit max_memory_gb
        output_grid = OutputGridConfig(
            x_min=0, x_max=1000, y_min=0, y_max=1000,
            t_min_ms=0, t_max_ms=2000, dx=25, dy=25, dt_ms=2.0
        )
        tiling_config = TilingConfig()
        
        planner = TilePlanner(output_grid, tiling_config)
        assert planner.max_memory_gb == 16.0
        
        # Reset
        reset_settings()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
