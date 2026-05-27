"""Physical 2D field computation for the Grad-Shafranov task.

These utilities are shared between prepare_dataset.py (offline preprocessing)
and infer.py (online inference from raw freegs_input NPZ files).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

# FreeGSNKE / freegs4e are resolved via the project's virtual-env.
# If running outside the installed env, callers must add freegsnke to sys.path.
from freegs4e.machine import Circuit, Coil, Wall
from freegsnke.equilibrium_update import Equilibrium
from freegsnke.machine_update import Machine


def build_machine(data: Any) -> Machine:
    """Build a FreeGSNKE Machine from a freegs_input NPZ record.

    Args:
        data: numpy NPZ-like object with keys:
              efit_fcoil_r, efit_fcoil_z, efit_fcoil_xmult,
              efit_fcoil_circ, efit_fcoil_c, limiter_r, limiter_z

    Returns:
        Configured Machine instance.
    """
    fcoil_r    = data["efit_fcoil_r"]
    fcoil_z    = data["efit_fcoil_z"]
    fcoil_xm   = data["efit_fcoil_xmult"]
    fcoil_circ = data["efit_fcoil_circ"]
    fcoil_c    = data["efit_fcoil_c"]

    segs: dict[int, list] = {}
    for i in range(len(fcoil_r)):
        c = int(fcoil_circ[i])
        segs.setdefault(c, []).append((
            f"seg_{i}",
            Coil(float(fcoil_r[i]), float(fcoil_z[i]), turns=1, control=False),
            float(fcoil_xm[i]),
        ))
    coils = []
    for c in sorted(segs):
        coils.append((
            f"circuit_{c}",
            Circuit(segs[c], current=float(fcoil_c[c - 1]), control=False),
        ))
    valid = ~np.isnan(data["limiter_r"]) & ~np.isnan(data["limiter_z"])
    wall  = Wall(data["limiter_r"][valid], data["limiter_z"][valid])
    return Machine(coils, wall=wall, limiter=wall)


def make_equilibrium(
    tokamak: Machine,
    Rmin: float, Rmax: float,
    Zmin: float, Zmax: float,
    nx: int, ny: int,
) -> Equilibrium:
    """Create an empty Equilibrium on the given grid."""
    return Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
        nx=nx, ny=ny,
    )


def compute_psi_init(eq: Equilibrium) -> np.ndarray:
    """Vacuum coil field — solution to Laplace equation from coil currents only.

    No plasma contribution. Encodes coil geometry + currents for this time slice.

    Returns:
        float32 array of shape (nx, ny).
    """
    return eq.tokamak.calcPsiFromGreens(eq._pgreen).astype(np.float32)


def project_profile_to_2d(
    profile_1d: np.ndarray,
    psi_norm_1d: np.ndarray,
    psi_init: np.ndarray,
) -> np.ndarray:
    """Project a 1D radial profile f(ψ_norm) onto the 2D spatial grid.

    Uses psi_init as an approximate flux coordinate: normalise it to [0, 1]
    and evaluate the profile at each grid cell.

    This is a first-order approximation — the true ψ_norm coordinate requires
    the converged solution, which we don't have yet. Good enough as model input.

    Args:
        profile_1d:  1D array of profile values, indexed by psi_norm_1d.
        psi_norm_1d: 1D array of normalised flux coordinates in [0, 1].
        psi_init:    2D vacuum flux array (nx, ny).

    Returns:
        float32 array of shape (nx, ny).
    """
    lo, hi = float(psi_init.min()), float(psi_init.max())
    if abs(hi - lo) < 1e-10:
        return np.zeros_like(psi_init, dtype=np.float32)
    psi_norm_2d = np.clip((psi_init - lo) / (hi - lo), 0.0, 1.0)
    f = interp1d(
        psi_norm_1d, profile_1d,
        kind="linear", bounds_error=False,
        fill_value=(float(profile_1d[0]), float(profile_1d[-1])),
    )
    return f(psi_norm_2d).astype(np.float32)


def build_input_fields(
    data: Any,
    Rmin: float, Rmax: float,
    Zmin: float, Zmax: float,
    nx: int, ny: int,
) -> dict[str, np.ndarray]:
    """Compute the three 2D physical input fields from a raw freegs_input NPZ.

    This is the online version of prepare_dataset.process_sample — used during
    inference when no pre-built dataset exists.

    Args:
        data:        numpy NPZ-like object (freegs_input_t*.npz).
        Rmin/Rmax/Zmin/Zmax: grid boundaries in metres.
        nx, ny:      grid resolution.

    Returns:
        dict with keys "psi_init", "pprime_map", "ffprime_map",
        each a float32 array of shape (nx, ny).

    Raises:
        Exception: if machine construction fails (bad coil data).
    """
    tokamak     = build_machine(data)
    eq          = make_equilibrium(tokamak, Rmin, Rmax, Zmin, Zmax, nx, ny)
    psi_init    = compute_psi_init(eq)
    psi_norm_1d = data["psi_norm"].astype(np.float32)

    pprime_map  = project_profile_to_2d(
        data["pprime"].astype(np.float32), psi_norm_1d, psi_init,
    )
    ffprime_map = project_profile_to_2d(
        data["ffprime"].astype(np.float32), psi_norm_1d, psi_init,
    )
    return {
        "psi_init":    psi_init,
        "pprime_map":  pprime_map,
        "ffprime_map": ffprime_map,
    }
