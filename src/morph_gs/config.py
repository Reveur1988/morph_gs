"""Machine and solver constants for morph_gs.

Canonical source for MAST coil configuration and FreeGSNKE solver defaults.
Migrated from experiments/02_freegsnke_efit_init/scripts/config.py.
All new imports of AMC_SHORT / PF_ACTIVE_COILS should come from here.
"""

# AMC channel name → short name
AMC_SHORT: dict[str, str] = {
    "AMC_P2IL FEED CURRENT": "P2IL",
    "AMC_P2IU FEED CURRENT": "P2IU",
    "AMC_P2OL FEED CURRENT": "P2OL",
    "AMC_P2OU FEED CURRENT": "P2OU",
    "AMC_P3L FEED CURRENT":  "P3L",
    "AMC_P3U FEED CURRENT":  "P3U",
    "AMC_P4L FEED CURRENT":  "P4L",
    "AMC_P4U FEED CURRENT":  "P4U",
    "AMC_P5L FEED CURRENT":  "P5L",
    "AMC_P5U FEED CURRENT":  "P5U",
}

# pf_active dataset variable prefixes for per-turn coil geometry
PF_ACTIVE_COILS: dict[str, str] = {
    "P2IU": "p2_inner_upper", "P2OU": "p2_outer_upper",
    "P2IL": "p2_inner_lower", "P2OL": "p2_outer_lower",
    "P3U":  "p3_upper",       "P3L":  "p3_lower",
    "P4U":  "p4_upper",       "P4L":  "p4_lower",
    "P5U":  "p5_upper",       "P5L":  "p5_lower",
    "P6U":  "p6_upper",       "P6L":  "p6_lower",
}

# FreeGSNKE solver defaults
SOLVER_TOL             = 1e-3
SOLVER_MAXITS          = 100
SOLVER_PICARD_HANDOVER = 0.11

# GS grid (MAST geometry)
NX, NY     = 65, 65
RMIN, RMAX = 0.06, 2.0
ZMIN, ZMAX = -2.0, 2.0

# Reference vacuum toroidal field for MAST; fallback if absent in dataset
MAST_FVAC_FALLBACK = 0.44

# EFIT coil → circuit number mapping (used when building equilibrium from EFIT data)
EFIT_CIRCUIT_MAP: dict[str, int] = {
    "P2IU": 1, "P2OU": 2, "P2IL": 3, "P2OL": 4,
    "P3U":  5, "P3L":  6,
    "P4U":  7, "P4L":  8,
    "P5U":  9, "P5L":  10,
    "P6U":  11, "P6L": 12,
    "SOL":  0,
}
