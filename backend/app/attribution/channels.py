"""Evidence channels E1-E4 — §5.3.

E2 and E3 MUST NOT touch the current field. That independence is the entire
reason the system survives a coarse ocean model, and it is the answer to the
sharpest question a judge can ask. Do not refactor it away (§5.3).

    E1  spatiotemporal mass overlap   m_j = sum_t integral O(x,y,t) * C_j(x,y,t) dA
                                      C_j Gaussian corridor, sigma_c = 500 m
                                      s1_j = m_j / sum_k m_k, t*_j = argmax snapshot
    E2  axial coherence               dtheta = fold(|orient_slick - COG_j(t*)|, 90)
                                      s2_j = exp(-(dtheta / 25 deg)^2)
    E3  kinematic consistency         L_released = major_axis_km / stretch_factor
                                      tau_j = L_released / SOG_j
                                      s3_j = lognormal(tau; median 90 min, sigma 0.9)
    E4  dark-gap coincidence          s4_j = 1 + 0.4 * min(g_j / 30, 3)
                                      boost only, max 2.2x, never a penalty
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class ChannelScores:
    s1: float
    s2: float
    s3: float
    s4: float
    t_star: datetime | None


def e1_mass_overlap(
    density_by_snapshot: list[np.ndarray],
    snapshot_times: list[datetime],
    vessel_positions: np.ndarray,
    corridor_sigma_m: float,
) -> tuple[float, datetime]: ...


def e2_axial_coherence(
    slick_orientation_deg: float,
    cog_deg: float,
    sigma_theta_deg: float,
) -> float: ...


def e3_kinematic_consistency(
    major_axis_km: float,
    stretch_factor: float,
    sog_kn: float,
    tau_median_min: float,
    tau_sigma: float,
) -> float: ...


def e4_dark_gap(
    gap_minutes: float,
    coefficient: float,
    reference_min: float,
    cap: float,
) -> float: ...
