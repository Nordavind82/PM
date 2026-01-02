"""Color palettes for seismic visualization."""

import numpy as np


def create_palette(name: str, n: int = 256) -> np.ndarray:
    """Create color palette as (n, 3) RGB array."""
    t = np.linspace(0, 1, n)

    if name == "gray":
        r = g = b = (t * 255).astype(np.uint8)
    elif name == "seismic":
        # Blue-White-Red (centered at white for +/- data)
        r = np.where(t < 0.5, t * 2 * 255, 255).astype(np.uint8)
        g = np.where(t < 0.5, t * 2 * 255, (1 - t) * 2 * 255).astype(np.uint8)
        b = np.where(t < 0.5, 255, (1 - t) * 2 * 255).astype(np.uint8)
    elif name == "rwb":
        # Red-White-Blue (reversed seismic)
        r = np.where(t < 0.5, 255, (1 - t) * 2 * 255).astype(np.uint8)
        g = np.where(t < 0.5, t * 2 * 255, (1 - t) * 2 * 255).astype(np.uint8)
        b = np.where(t < 0.5, t * 2 * 255, 255).astype(np.uint8)
    elif name == "viridis":
        # Dark purple -> blue -> green -> yellow (perceptually uniform)
        r = (np.clip(0.267 + 0.004*t + 1.2*t**2 - 0.8*t**3, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(0.004 + 1.0*t - 0.15*t**2, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(0.329 + 0.6*t - 0.6*t**2 - 0.2*t**3, 0, 1) * 255).astype(np.uint8)
    elif name == "bone":
        r = (np.clip(t * 0.75 + 0.25 * np.maximum(0, t - 0.75) * 4, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(t * 0.75 + 0.25 * np.clip((t - 0.25) * 4, 0, 1), 0, 1) * 255).astype(np.uint8)
        b = (np.clip(t * 0.75 + 0.25 * np.clip(t * 4, 0, 1), 0, 1) * 255).astype(np.uint8)
    elif name == "hot":
        # Black -> Red -> Yellow -> White (high contrast for positive data)
        r = np.clip(t * 3, 0, 1)
        g = np.clip(t * 3 - 1, 0, 1)
        b = np.clip(t * 3 - 2, 0, 1)
        r = (r * 255).astype(np.uint8)
        g = (g * 255).astype(np.uint8)
        b = (b * 255).astype(np.uint8)
    elif name == "plasma":
        # Dark blue -> purple -> orange -> yellow (perceptually uniform, high contrast)
        r = (np.clip(0.05 + 0.9*t + 0.5*t**2 - 0.45*t**3, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(0.03 + 0.1*t + 0.9*t**2 - 0.2*t**3, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(0.53 + 0.5*t - 1.5*t**2 + 0.5*t**3, 0, 1) * 255).astype(np.uint8)
    elif name == "inferno":
        # Black -> purple -> red -> yellow (high contrast)
        r = (np.clip(-0.03 + 1.5*t - 0.5*t**2, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(-0.03 + 0.8*t**2, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(0.02 + 1.2*t - 1.8*t**2 + 0.6*t**3, 0, 1) * 255).astype(np.uint8)
    elif name == "magma":
        # Black -> purple -> pink -> white
        r = (np.clip(0.0 + 0.8*t + 0.4*t**2 - 0.2*t**3, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(0.0 + 0.2*t + 0.8*t**2, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(0.02 + 0.9*t - 0.3*t**2 - 0.1*t**3, 0, 1) * 255).astype(np.uint8)
    elif name == "turbo":
        # Improved rainbow: dark blue -> cyan -> green -> yellow -> red
        # Better perceptual uniformity than jet
        r = (np.clip(0.14 - 0.15*t + 4.5*t**2 - 4.0*t**3 + 1.0*t**4, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(-0.03 + 2.5*t - 1.5*t**2, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(0.48 + 2.0*t - 5.5*t**2 + 4.0*t**3, 0, 1) * 255).astype(np.uint8)
    elif name == "jet":
        # Classic rainbow: blue -> cyan -> green -> yellow -> red
        r = np.clip(1.5 - np.abs(4*t - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4*t - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4*t - 1), 0, 1)
        r = (r * 255).astype(np.uint8)
        g = (g * 255).astype(np.uint8)
        b = (b * 255).astype(np.uint8)
    elif name == "coolwarm":
        # Blue -> white -> red (diverging, good for +/- data)
        r = np.where(t < 0.5, 0.23 + t * 1.54, 1.0)
        g = np.where(t < 0.5, 0.3 + t * 1.4, 1.0 - (t - 0.5) * 1.4)
        b = np.where(t < 0.5, 0.75 + t * 0.5, 1.0 - (t - 0.5) * 1.54)
        r = (np.clip(r, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(b, 0, 1) * 255).astype(np.uint8)
    elif name == "cividis":
        # Colorblind-friendly: blue -> yellow
        r = (np.clip(-0.05 + 0.65*t + 0.45*t**2, 0, 1) * 255).astype(np.uint8)
        g = (np.clip(0.13 + 0.75*t, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(0.33 + 0.15*t - 0.35*t**2, 0, 1) * 255).astype(np.uint8)
    else:
        r = g = b = (t * 255).astype(np.uint8)

    return np.stack([r, g, b], axis=1)


# All available palettes
PALETTES = {
    "Gray": create_palette("gray"),
    "Seismic (BWR)": create_palette("seismic"),
    "Seismic (RWB)": create_palette("rwb"),
    "Coolwarm": create_palette("coolwarm"),
    "Viridis": create_palette("viridis"),
    "Plasma": create_palette("plasma"),
    "Inferno": create_palette("inferno"),
    "Magma": create_palette("magma"),
    "Hot": create_palette("hot"),
    "Turbo": create_palette("turbo"),
    "Jet": create_palette("jet"),
    "Cividis": create_palette("cividis"),
    "Bone": create_palette("bone"),
}

# Palettes recommended for seismic amplitude (centered, +/- data)
SEISMIC_PALETTES = ["Gray", "Seismic (BWR)", "Seismic (RWB)", "Coolwarm", "Bone"]

# Palettes recommended for positive-only data (semblance, velocity, attributes)
POSITIVE_PALETTES = ["Hot", "Plasma", "Inferno", "Magma", "Viridis", "Turbo", "Jet", "Cividis"]
