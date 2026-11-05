"""
Heat Exchanger Digital Twin — v3
==================================
Double-pipe (concentric tube) heat exchanger, matching the NC State
CHE 330 rig exactly (per the official lab handout, Cooper/Lim/Dickey/
Crothers, rev. 1/6/2018).

Hot stream flows in the inner tube. Cold stream flows in the annulus
(outer "shell"). Counter-current operation only (per project scope).

Changes from v2:
  - Your own 16-trial dataset (2026-10-12) is built in as
    MEASURED_2026_10_12, with a use_in_fit flag (trials 1-4 excluded:
    energy balance fails, hot flow reading suspect at low rotameter end).
  - wilson_combined(): fits 1/UA = C + b_hot/v_hot^0.8 + b_cold/v_cold^0.8
    in one regression (both flows vary in a factorial, so a one-stream
    Wilson plot needs the other flow held fixed; the combined fit uses
    all valid trials at once).
  - Section 11 (physics vs ML vs actual) is now implemented:
    compare_models() runs the physics twin and a leave-one-out Ridge
    model on every trial and reports MAE for each.
  - Self-test now runs on your data, not just the reference set.

Changes from v1:
  - Real rig geometry (Table 1 of the handout) replaces placeholders.
  - Annulus (hydraulic-diameter) correlation replaces Kern's shell-side
    correlation — Kern's method is for baffled shell-and-tube bundles,
    not a concentric double-pipe annulus.
  - Dittus-Boelter exponent fixed at n = 0.3 for BOTH streams, matching
    Eq. 6-7 of the handout exactly (not the heating/cooling n=0.4/0.3
    split used in v1 and in general textbooks).
  - Midpoint temperature prediction added (the rig has 6 thermistors:
    hot in/mid/out, cold in/mid/out).
  - Wilson's Method of Analysis added (Eq. 11-12 of the handout) — an
    independent, empirical way to extract h_i and h_o from real trial
    data, to compare against the Dittus-Boelter predictions.
  - Loader for the reference counter-current dataset (16 runs, prior
    semester) as a baseline / sanity check before your own re-run.

Structure:
  1. Fixed geometry & material constants (real rig dimensions)
  2. Water property lookup (function of T)
  3. Convective heat transfer coefficient (Dittus-Boelter, tube + annulus)
  4. Overall U, Q via effectiveness-NTU, + midpoint prediction
  5. Pressure drop (Darcy-Weisbach + Haaland) and pumping power
  6. Full "predict" function — the physics-based digital twin
  7. DOE trial log: 4x4 factorial, counter-current only (16 trials)
  8. Reference dataset (prior semester, counter-current, 16 runs)
  8b. Your dataset (2026-10-12, 16 trials, use_in_fit flag)
  9. Wilson's Method of Analysis (one-stream and combined fits)
  10. ML/regression model (fit once your own data exists)
  11. Comparison: physics vs ML vs actual
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. FIXED GEOMETRY & MATERIAL CONSTANTS — real CHE 330 double-pipe rig
#    (Table 1 of the lab handout)
# ---------------------------------------------------------------------------

GEOMETRY = {
    # Inner pipe (hot stream, tube side)
    "inner_pipe_OD_m": 0.0127,
    "inner_pipe_wall_m": 0.00076,
    # Outer pipe (cold stream, annulus / "shell" side)
    "outer_pipe_OD_m": 0.0190,
    "outer_pipe_wall_m": 0.00124,
    # Shared
    "length_m": 3.048,          # single pass
    "wall_k_W_mK": 16.3,        # Type 314 stainless steel, approx. — confirm if precision needed
    "A_inner_m2": 0.1070,       # Ai — inner surface area of inner tube (hot-side heat transfer area)
    "A_outer_m2": 0.1216,       # Ao — outer surface area of inner tube (cold-side heat transfer area)
    "A_mean_m2": 0.1143,        # Aw — mean = 0.5*(Ai+Ao), used for wall conduction resistance
    "pump_efficiency": 0.6,     # not specified in handout — assumed typical value
    "minor_loss_K_total": 2.0,  # not specified in handout — assumed; refine if needed
}

# Derived diameters
GEOMETRY["Di_inner_m"] = GEOMETRY["inner_pipe_OD_m"] - 2 * GEOMETRY["inner_pipe_wall_m"]   # tube-side flow ID
GEOMETRY["ID_outer_m"] = GEOMETRY["outer_pipe_OD_m"] - 2 * GEOMETRY["outer_pipe_wall_m"]   # outer pipe ID

# Annulus flow area (Eq. 8) and "equivalent diameter" (Eq. 9), exactly as defined in the handout
_r_annulus_outer = GEOMETRY["ID_outer_m"] / 2
_r_annulus_inner = GEOMETRY["inner_pipe_OD_m"] / 2
GEOMETRY["Ah_annulus_m2"] = np.pi * (_r_annulus_outer**2 - _r_annulus_inner**2)
GEOMETRY["Dh_annulus_m"] = 2 * np.sqrt(GEOMETRY["Ah_annulus_m2"] / np.pi)

# Tube-side flow area
GEOMETRY["A_tube_flow_m2"] = np.pi * (GEOMETRY["Di_inner_m"] / 2) ** 2

# Rotameter calibration (Eq. 13-14) — only needed if you're converting raw rotameter
# readings (RR) to flow rate yourself; the reference dataset already gives flow in L/min.
def hot_flow_from_rotameter(RR_hot):
    return 1.296 * RR_hot - 1.075  # L/min

def cold_flow_from_rotameter(RR_cold):
    return 1.029 * RR_cold + 0.499  # L/min


# ---------------------------------------------------------------------------
# 2. WATER PROPERTY LOOKUP (function of temperature, deg C)
# ---------------------------------------------------------------------------
_T_ref = np.array([10, 20, 30, 40, 50, 60, 70, 80])
_rho_ref = np.array([999.7, 998.2, 995.7, 992.2, 988.0, 983.2, 977.8, 971.8])
_mu_ref = np.array([1.307e-3, 1.002e-3, 0.7975e-3, 0.6529e-3,
                     0.5468e-3, 0.4665e-3, 0.4035e-3, 0.3536e-3])
_cp_ref = np.array([4192, 4184, 4180, 4179, 4181, 4185, 4190, 4197])
_k_ref = np.array([0.580, 0.598, 0.615, 0.630, 0.643, 0.654, 0.663, 0.670])


def water_properties(T_C):
    """Return (rho, mu, cp, k) for water at temperature T_C [deg C],
    linearly interpolated from NIST reference data."""
    rho = np.interp(T_C, _T_ref, _rho_ref)
    mu = np.interp(T_C, _T_ref, _mu_ref)
    cp = np.interp(T_C, _T_ref, _cp_ref)
    k = np.interp(T_C, _T_ref, _k_ref)
    return rho, mu, cp, k


# ---------------------------------------------------------------------------
# 3. CONVECTIVE HEAT TRANSFER COEFFICIENT
#    Dittus-Boelter, Eq. 6-7 of the handout: Nu = 0.023 Re^0.8 Pr^0.3
#    (handout uses n=0.3 for BOTH streams — not the heating/cooling split)
# ---------------------------------------------------------------------------

def reynolds_number(mdot_kg_s, D_m, mu):
    """Re = rho*v*D/mu, expressed via mass flow: Re = 4*mdot/(pi*D*mu)."""
    return 4 * mdot_kg_s / (np.pi * D_m * mu)


def prandtl_number(mu, cp, k):
    return mu * cp / k


def nusselt_dittus_boelter(Re, Pr):
    """Eq. 6-7 of the CHE 330 handout. Valid for fully-developed turbulent flow."""
    if Re < 2300:
        return 3.66  # laminar fallback, fully-developed, constant wall T
    return 0.023 * Re**0.8 * Pr**0.3


def convective_coefficient(Nu, k, D):
    return Nu * k / D


# ---------------------------------------------------------------------------
# 4. OVERALL U, HEAT DUTY (effectiveness-NTU), AND MIDPOINT PREDICTION
# ---------------------------------------------------------------------------

def overall_UA(h_i, h_o, geometry=GEOMETRY):
    """Eq. 5 of the handout: 1/UA = 1/(hi*Ai) + tw/(kw*Aw) + 1/(ho*Ao)
    (scale/fouling resistances Ri/Ai, Ro/Ao omitted — assume clean tubes
    unless doing a fouling study)."""
    R_i = 1 / (h_i * geometry["A_inner_m2"])
    R_wall = geometry["inner_pipe_wall_m"] / (geometry["wall_k_W_mK"] * geometry["A_mean_m2"])
    R_o = 1 / (h_o * geometry["A_outer_m2"])
    R_total = R_i + R_wall + R_o
    return 1 / R_total  # UA, W/K


def effectiveness_ntu(UA, C_hot, C_cold, config="counter", length_fraction=1.0):
    """Effectiveness-NTU, evaluated over a fraction of the total exchanger
    length (1.0 = full length/outlet, 0.5 = midpoint). UA scales linearly
    with length for a double-pipe exchanger of uniform cross-section."""
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    Cr = C_min / C_max
    NTU = (UA * length_fraction) / C_min

    if config == "counter":
        if Cr < 1:
            eps = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
        else:
            eps = NTU / (1 + NTU)
    elif config == "co":
        eps = (1 - np.exp(-NTU * (1 + Cr))) / (1 + Cr)
    else:
        raise ValueError("config must be 'counter' or 'co'")

    return eps, NTU, C_min, C_max


# ---------------------------------------------------------------------------
# 5. PRESSURE DROP AND PUMPING POWER
# ---------------------------------------------------------------------------

def haaland_friction_factor(Re, relative_roughness=1e-5):
    if Re < 2300:
        return 64 / Re
    term = (relative_roughness / 3.7) ** 1.11 + 6.9 / Re
    return (-1.8 * np.log10(term)) ** -2


def pressure_drop(mdot_kg_s, D_m, length_m, rho, mu, area_m2, minor_loss_K=0.0):
    velocity = mdot_kg_s / (rho * area_m2)
    Re = reynolds_number(mdot_kg_s, D_m, mu)
    f = haaland_friction_factor(Re)
    dP_major = f * (length_m / D_m) * (rho * velocity**2 / 2)
    dP_minor = minor_loss_K * (rho * velocity**2 / 2)
    return dP_major + dP_minor, velocity, Re


def pumping_power(dP_Pa, volumetric_flow_m3_s, pump_efficiency=0.6):
    return dP_Pa * volumetric_flow_m3_s / pump_efficiency


# ---------------------------------------------------------------------------
# 6. FULL PREDICTION FUNCTION — THE PHYSICS-BASED DIGITAL TWIN
# ---------------------------------------------------------------------------

def predict_heat_exchanger(hot_flow_Lmin, cold_flow_Lmin,
                            hot_in_C, cold_in_C,
                            config="counter", geometry=GEOMETRY):
    """
    Counter-current double-pipe exchanger. Hot stream in inner tube,
    cold stream in annulus. Returns predicted outlet AND midpoint
    temperatures, heat duty, U, pressure drop, and pumping power.
    """
    rho_h, mu_h, cp_h, k_h = water_properties(hot_in_C)
    rho_c, mu_c, cp_c, k_c = water_properties(cold_in_C)

    hot_flow_m3s = hot_flow_Lmin / 60000.0
    cold_flow_m3s = cold_flow_Lmin / 60000.0
    mdot_hot = hot_flow_m3s * rho_h
    mdot_cold = cold_flow_m3s * rho_c

    # Tube side (hot, inner pipe)
    Re_tube = reynolds_number(mdot_hot, geometry["Di_inner_m"], mu_h)
    Pr_tube = prandtl_number(mu_h, cp_h, k_h)
    Nu_tube = nusselt_dittus_boelter(Re_tube, Pr_tube)
    h_i = convective_coefficient(Nu_tube, k_h, geometry["Di_inner_m"])

    # Annulus side (cold, outer pipe)
    Re_ann = reynolds_number(mdot_cold, geometry["Dh_annulus_m"], mu_c)
    Pr_ann = prandtl_number(mu_c, cp_c, k_c)
    Nu_ann = nusselt_dittus_boelter(Re_ann, Pr_ann)
    h_o = convective_coefficient(Nu_ann, k_c, geometry["Dh_annulus_m"])

    UA_full = overall_UA(h_i, h_o, geometry)

    C_hot = mdot_hot * cp_h
    C_cold = mdot_cold * cp_c

    # Outlet prediction (full length)
    eps, NTU, C_min, C_max = effectiveness_ntu(UA_full, C_hot, C_cold, config, length_fraction=1.0)
    Q_max = C_min * (hot_in_C - cold_in_C)
    Q = eps * Q_max
    hot_out_C = hot_in_C - Q / C_hot
    cold_out_C = cold_in_C + Q / C_cold

    # Midpoint prediction: axial temperature profile at x = L/2 measured from the
    # hot inlet. For counter-current flow the cold stream enters at the far end,
    # so the cold mid temperature must be found from the whole-exchanger solution
    # (a half-length exchanger with the same inlets is wrong for the cold side).
    #   dT/dx = -(UA/C_hot)(Th-Tc);  Th-Tc = dT0 * exp(-a x),  a = UA(1/C_hot - 1/C_cold)
    if config == "counter":
        f = 0.5
        dT0 = hot_in_C - cold_out_C                      # (Th - Tc) at the hot-inlet end
        a = UA_full * (1.0 / C_hot - 1.0 / C_cold)
        integral = dT0 * f if abs(a) < 1e-9 else dT0 * (1.0 - np.exp(-a * f)) / a
        Q_mid = UA_full * integral                       # heat moved in the hot-inlet half
        hot_mid_C = hot_in_C - Q_mid / C_hot
        cold_mid_C = cold_out_C - Q_mid / C_cold
    else:
        eps_mid, _, _, _ = effectiveness_ntu(UA_full, C_hot, C_cold, config, length_fraction=0.5)
        Q_mid = eps_mid * Q_max
        hot_mid_C = hot_in_C - Q_mid / C_hot
        cold_mid_C = cold_in_C + Q_mid / C_cold

    # Pressure drop & pumping power
    dP_hot, v_hot, _ = pressure_drop(mdot_hot, geometry["Di_inner_m"], geometry["length_m"],
                                      rho_h, mu_h, geometry["A_tube_flow_m2"],
                                      geometry["minor_loss_K_total"] / 2)
    dP_cold, v_cold, _ = pressure_drop(mdot_cold, geometry["Dh_annulus_m"], geometry["length_m"],
                                        rho_c, mu_c, geometry["Ah_annulus_m2"],
                                        geometry["minor_loss_K_total"] / 2)
    power_hot = pumping_power(dP_hot, hot_flow_m3s, geometry["pump_efficiency"])
    power_cold = pumping_power(dP_cold, cold_flow_m3s, geometry["pump_efficiency"])

    U_areal = UA_full / geometry["A_outer_m2"]  # reported per Ao, consistent with handout's Ao-based U

    return {
        "Q_W": Q, "Q_mid_W": Q_mid,
        "hot_out_C": hot_out_C, "cold_out_C": cold_out_C,
        "hot_mid_C": hot_mid_C, "cold_mid_C": cold_mid_C,
        "U_W_m2K": U_areal, "UA_W_K": UA_full,
        "h_i_W_m2K": h_i, "h_o_W_m2K": h_o,
        "effectiveness": eps, "NTU": NTU,
        "dP_hot_Pa": dP_hot, "dP_cold_Pa": dP_cold,
        "pumping_power_hot_W": power_hot, "pumping_power_cold_W": power_cold,
        "pumping_power_total_W": power_hot + power_cold,
        "Re_tube": Re_tube, "Re_annulus": Re_ann,
    }


# ---------------------------------------------------------------------------
# 7. DOE TRIAL LOG — 4x4 FACTORIAL, COUNTER-CURRENT ONLY (16 TRIALS)
#    Hot 2-5 L/min (rig max ~5), cold 4-7 L/min. Per handout Section 5: vary hot flow (4 levels) x cold flow (4 levels),
#    evenly spaced over as wide a range as possible.
# ---------------------------------------------------------------------------

def build_doe_trial_log(hot_flow_levels=(2, 3, 4, 5), cold_flow_levels=(4, 5, 6, 7)):
    trials = []
    for hf in hot_flow_levels:
        for cf in cold_flow_levels:
            trials.append({
                "hot_flow_Lmin": hf, "cold_flow_Lmin": cf,
                "config": "counter",
            })
    df = pd.DataFrame(trials)
    df.insert(0, "trial_id", range(1, len(df) + 1))
    measured_cols = [
        "hot_in_C_measured", "hot_mid_C_measured", "hot_out_C_measured",
        "cold_in_C_measured", "cold_mid_C_measured", "cold_out_C_measured",
        "dP_hot_Pa_measured", "dP_cold_Pa_measured", "notes",
    ]
    for col in measured_cols:
        df[col] = np.nan if col != "notes" else ""
    return df


# ---------------------------------------------------------------------------
# 8. REFERENCE DATASET — prior semester, counter-current, 16 runs
#    (from Heat_Exchanger_raw_data.xlsx). Use this to sanity-check your
#    model BEFORE running your own trials, and as a baseline comparison.
# ---------------------------------------------------------------------------

REFERENCE_COUNTER_CURRENT = pd.DataFrame([
    dict(run=1, hot_flow_Lmin=4, cold_flow_Lmin=4, hot_in_C=58.3, hot_mid_C=48.6, hot_out_C=39.2,
         cold_in_C=11.0, cold_mid_C=16.0, cold_out_C=25.7),
    dict(run=2, hot_flow_Lmin=4, cold_flow_Lmin=5, hot_in_C=58.9, hot_mid_C=47.7, hot_out_C=38.5,
         cold_in_C=11.3, cold_mid_C=16.4, cold_out_C=24.8),
    dict(run=3, hot_flow_Lmin=4, cold_flow_Lmin=6, hot_in_C=58.2, hot_mid_C=46.6, hot_out_C=37.0,
         cold_in_C=11.2, cold_mid_C=15.6, cold_out_C=23.4),
    dict(run=4, hot_flow_Lmin=4, cold_flow_Lmin=7, hot_in_C=57.4, hot_mid_C=45.1, hot_out_C=35.5,
         cold_in_C=11.1, cold_mid_C=14.9, cold_out_C=22.0),
    dict(run=5, hot_flow_Lmin=5, cold_flow_Lmin=4, hot_in_C=50.2, hot_mid_C=43.8, hot_out_C=37.3,
         cold_in_C=11.3, cold_mid_C=17.2, cold_out_C=24.7),
    dict(run=6, hot_flow_Lmin=5, cold_flow_Lmin=5, hot_in_C=50.3, hot_mid_C=42.9, hot_out_C=36.0,
         cold_in_C=11.2, cold_mid_C=16.2, cold_out_C=23.3),
    dict(run=7, hot_flow_Lmin=5, cold_flow_Lmin=6, hot_in_C=50.5, hot_mid_C=42.8, hot_out_C=35.7,
         cold_in_C=11.1, cold_mid_C=15.8, cold_out_C=22.7),
    dict(run=8, hot_flow_Lmin=5, cold_flow_Lmin=7, hot_in_C=52.1, hot_mid_C=42.9, hot_out_C=35.4,
         cold_in_C=11.1, cold_mid_C=15.2, cold_out_C=22.1),
    dict(run=9, hot_flow_Lmin=6, cold_flow_Lmin=4, hot_in_C=50.1, hot_mid_C=44.8, hot_out_C=39.1,
         cold_in_C=11.4, cold_mid_C=18.0, cold_out_C=26.0),
    dict(run=10, hot_flow_Lmin=6, cold_flow_Lmin=5, hot_in_C=50.1, hot_mid_C=44.0, hot_out_C=37.9,
         cold_in_C=11.3, cold_mid_C=17.0, cold_out_C=24.7),
    dict(run=11, hot_flow_Lmin=6, cold_flow_Lmin=6, hot_in_C=50.1, hot_mid_C=43.4, hot_out_C=36.9,
         cold_in_C=11.2, cold_mid_C=16.3, cold_out_C=23.6),
    dict(run=12, hot_flow_Lmin=6, cold_flow_Lmin=7, hot_in_C=50.1, hot_mid_C=42.8, hot_out_C=36.0,
         cold_in_C=11.2, cold_mid_C=15.6, cold_out_C=22.5),
    dict(run=13, hot_flow_Lmin=7, cold_flow_Lmin=4, hot_in_C=44.9, hot_mid_C=40.9, hot_out_C=36.6,
         cold_in_C=11.2, cold_mid_C=17.4, cold_out_C=24.5),
    dict(run=14, hot_flow_Lmin=7, cold_flow_Lmin=5, hot_in_C=45.5, hot_mid_C=40.8, hot_out_C=36.1,
         cold_in_C=11.2, cold_mid_C=16.7, cold_out_C=23.7),
    dict(run=15, hot_flow_Lmin=7, cold_flow_Lmin=6, hot_in_C=48.0, hot_mid_C=42.1, hot_out_C=36.9,
         cold_in_C=11.2, cold_mid_C=16.5, cold_out_C=23.8),
    dict(run=16, hot_flow_Lmin=7, cold_flow_Lmin=7, hot_in_C=50.0, hot_mid_C=43.6, hot_out_C=37.4,
         cold_in_C=11.1, cold_mid_C=16.3, cold_out_C=23.3),
])


# ---------------------------------------------------------------------------
# 8b. YOUR DATA — 16 trials run 2026-10-12 (heat_exchanger_raw_measurements.csv)
#     Hot 2/3/4/5 x cold 4/5/6/7 L/min, counter-current.
#     use_in_fit = False for trials 1-4: energy balance closure ~110-120%,
#     hot flow reading likely wrong at the low end of the rotameter; the
#     runs cannot be repeated. Trial 9 closes at 15.6% (just over the 15%
#     handout limit) and is kept. Trial 13 hot mid = 47.2 C (typo corrected).
#     Trial 16: hot inlet fell to 49 C (reservoir drift).
# ---------------------------------------------------------------------------

_cols = ["trial", "hot_flow_Lmin", "cold_flow_Lmin", "hot_in_C", "hot_mid_C", "hot_out_C",
         "cold_in_C", "cold_mid_C", "cold_out_C", "time"]
_rows = [
    (1, 2, 4, 55.2, 32.4, 22.6, 10.0, 10.7, 14.58, "11:02"),
    (2, 2, 5, 54.9, 31.4, 22.3, 10.0, 10.2, 13.38, "11:05"),
    (3, 2, 6, 55.3, 30.1, 21.1, 9.8, 10.0, 12.96, "11:07"),
    (4, 2, 7, 55.9, 27.6, 19.1, 9.7, 9.6, 12.2, "11:18"),
    (5, 3, 4, 55.9, 45.0, 36.0, 10.0, 16.0, 24.1, "11:22"),
    (6, 3, 5, 55.9, 43.7, 34.5, 10.0, 14.6, 22.2, "11:24"),
    (7, 3, 6, 55.9, 42.7, 33.04, 10.2, 14.4, 21.0, "11:26"),
    (8, 3, 7, 55.9, 42.0, 32.2, 10.6, 14.1, 20.3, "11:30"),
    (9, 4, 4, 55.9, 46.5, 38.07, 10.8, 17.6, 25.8, "11:33"),
    (10, 4, 5, 55.8, 45.9, 37.5, 10.8, 16.4, 24.6, "11:36"),
    (11, 4, 6, 55.8, 45.0, 36.2, 10.4, 15.6, 23.2, "11:39"),
    (12, 4, 7, 55.7, 43.9, 34.6, 10.2, 14.5, 21.5, "11:41"),
    (13, 5, 4, 55.7, 47.2, 39.5, 10.6, 19.13, 28.3, "11:44"),
    (14, 5, 5, 55.7, 47.5, 40.2, 10.8, 17.8, 26.5, "11:48"),
    (15, 5, 6, 54.8, 46.0, 38.5, 10.5, 16.6, 24.6, "11:50"),
    (16, 5, 7, 49.0, 39.8, 33.5, 10.3, 14.3, 20.9, "11:52"),
]
MEASURED_2026_10_12 = pd.DataFrame(_rows, columns=_cols)
MEASURED_2026_10_12["use_in_fit"] = MEASURED_2026_10_12["trial"] >= 5


# ---------------------------------------------------------------------------
# 9. WILSON'S METHOD OF ANALYSIS (Eq. 11-12 of the handout)
#    Empirically extracts alpha_i*A_i (or alpha_o*A_o) and the intercept C
#    from a linear regression of 1/UA vs 1/v^0.8, holding the OTHER
#    stream's conditions roughly constant. This is independent of the
#    Dittus-Boelter correlation — a direct check on whether theory matches
#    reality for YOUR specific rig.
# ---------------------------------------------------------------------------

def compute_UA_from_measurements(df, geometry=GEOMETRY):
    """Given a dataframe with hot_flow_Lmin, cold_flow_Lmin, hot_in_C,
    hot_out_C, cold_in_C, cold_out_C, compute Q (energy balance, both
    sides), LMTD (counter-current), and UA for each row."""
    out = df.copy()
    rho_h, mu_h, cp_h, k_h = zip(*[water_properties(t) for t in df["hot_in_C"]])
    rho_c, mu_c, cp_c, k_c = zip(*[water_properties(t) for t in df["cold_in_C"]])
    out["mdot_hot"] = (df["hot_flow_Lmin"] / 60000.0) * np.array(rho_h)
    out["mdot_cold"] = (df["cold_flow_Lmin"] / 60000.0) * np.array(rho_c)
    out["cp_hot"] = np.array(cp_h)
    out["cp_cold"] = np.array(cp_c)

    out["Q_hot_W"] = out["mdot_hot"] * out["cp_hot"] * (df["hot_in_C"] - df["hot_out_C"])
    out["Q_cold_W"] = out["mdot_cold"] * out["cp_cold"] * (df["cold_out_C"] - df["cold_in_C"])
    out["Q_avg_W"] = (out["Q_hot_W"] + out["Q_cold_W"]) / 2
    out["energy_balance_closure"] = (out["Q_hot_W"] - out["Q_cold_W"]).abs() / out["Q_avg_W"]

    # Counter-current LMTD (Eq. 4): ends are (hot_in - cold_out) and (hot_out - cold_in)
    dT1 = df["hot_in_C"] - df["cold_out_C"]
    dT2 = df["hot_out_C"] - df["cold_in_C"]
    out["LMTD_C"] = (dT1 - dT2) / np.log(dT1 / dT2)
    out["UA_W_K"] = out["Q_avg_W"] / out["LMTD_C"]

    # Velocities, for the Wilson plot x-axis (1/v^0.8)
    out["v_hot_m_s"] = out["mdot_hot"] / (np.array(rho_h) * geometry["A_tube_flow_m2"])
    out["v_cold_m_s"] = out["mdot_cold"] / (np.array(rho_c) * geometry["Ah_annulus_m2"])
    return out


def wilson_analysis(df_with_UA, stream="hot", geometry=GEOMETRY):
    """
    stream='hot': vary hot (tube-side) flow -> fit 1/UA vs 1/v_hot^0.8
                  to get alpha_i*A_i (slope^-1) and C_i (intercept).
    stream='cold': vary cold (annulus-side) flow -> fit 1/UA vs 1/v_cold^0.8
                   to get alpha_o*A_o (slope^-1) and C_o (intercept).
    Returns slope, intercept, implied h (at mean velocity), and R^2.
    """
    vcol = "v_hot_m_s" if stream == "hot" else "v_cold_m_s"
    A = geometry["A_inner_m2"] if stream == "hot" else geometry["A_outer_m2"]

    x = 1 / df_with_UA[vcol] ** 0.8
    y = 1 / df_with_UA["UA_W_K"]

    slope, intercept = np.polyfit(x, y, 1)
    alphaA = 1 / slope          # alpha_i * A_i  or  alpha_o * A_o
    h_at_mean_v = alphaA / A * df_with_UA[vcol].mean() ** 0.8  # implied h at mean velocity

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        "stream": stream, "slope": slope, "intercept_C": intercept,
        "alphaA": alphaA, "h_implied_W_m2K_at_mean_v": h_at_mean_v,
        "r_squared": r2,
    }


def wilson_combined(df_with_UA, geometry=GEOMETRY):
    """
    Combined Wilson fit for a factorial where BOTH flows vary:
        1/UA = C + b_hot * (1/v_hot^0.8) + b_cold * (1/v_cold^0.8)
    C ~ wall resistance + any scale/fouling (Eq. 5 of the handout).
    1/b_hot = alpha_i*A_i at v_hot = 1 m/s; 1/b_cold = alpha_o*A_o at v_cold = 1 m/s.
    Returns coefficients, R^2, the predicted wall resistance tw/(kw*Aw) for
    comparison with C, and Wilson-implied h_i, h_o at the mean velocities.
    """
    x1 = 1 / df_with_UA["v_hot_m_s"] ** 0.8
    x2 = 1 / df_with_UA["v_cold_m_s"] ** 0.8
    y = 1 / df_with_UA["UA_W_K"]
    X = np.column_stack([np.ones(len(y)), x1, x2])
    (C, b_hot, b_cold), *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ np.array([C, b_hot, b_cold])
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    R_wall = geometry["inner_pipe_wall_m"] / (geometry["wall_k_W_mK"] * geometry["A_mean_m2"])
    return {
        "n_trials": len(y), "intercept_C": C, "b_hot": b_hot, "b_cold": b_cold, "r_squared": r2,
        "alphaA_hot_at_1ms": 1 / b_hot, "alphaA_cold_at_1ms": 1 / b_cold,
        "R_wall_predicted": R_wall, "R_scale_implied": C - R_wall,
        "h_i_wilson_at_mean_v": df_with_UA["v_hot_m_s"].mean() ** 0.8 / (b_hot * geometry["A_inner_m2"]),
        "h_o_wilson_at_mean_v": df_with_UA["v_cold_m_s"].mean() ** 0.8 / (b_cold * geometry["A_outer_m2"]),
    }


# ---------------------------------------------------------------------------
# 10. ML / REGRESSION MODEL (fit once your own real data exists)
# ---------------------------------------------------------------------------

def fit_ml_model(df_with_measurements, target_col="hot_out_C_measured",
                  feature_cols=("hot_flow_Lmin", "cold_flow_Lmin", "hot_in_C_measured", "cold_in_C_measured")):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    data = df_with_measurements.dropna(subset=[target_col])
    X = data[list(feature_cols)].values
    y = data[target_col].values
    if len(data) < 5:
        raise ValueError("Need at least ~5 real trials before fitting a model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"ML model ({target_col}): MAE={mean_absolute_error(y_test, y_pred):.3f}, "
          f"R2={r2_score(y_test, y_pred) if len(y_test) > 1 else float('nan'):.3f}")
    return model


# ---------------------------------------------------------------------------
# 11. COMPARISON: PHYSICS vs ML vs ACTUAL
#     Physics twin: first-principles, no fitting to your data.
#     ML twin: Ridge regression on (hot flow, cold flow, hot in T, cold in T),
#     scored by leave-one-out cross-validation so every prediction is for a
#     trial the model did NOT see (with only ~12 valid trials, a single
#     train/test split leaves just 3 test points).
# ---------------------------------------------------------------------------

def compare_models(df, targets=("hot_out_C", "cold_out_C", "hot_mid_C", "cold_mid_C"),
                   features=("hot_flow_Lmin", "cold_flow_Lmin", "hot_in_C", "cold_in_C"),
                   ridge_alpha=1.0):
    """
    df: measured trials with columns hot_flow_Lmin, cold_flow_Lmin and the
    six temperatures (e.g. MEASURED_2026_10_12[MEASURED_2026_10_12.use_in_fit]).
    Returns (per_trial_table, summary_table). Errors are predicted - measured, deg C.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut

    data = df.reset_index(drop=True)
    table = data[["hot_flow_Lmin", "cold_flow_Lmin"] + list(targets)].copy()
    if "trial" in data:
        table.insert(0, "trial", data["trial"])

    # Physics twin
    phys = [predict_heat_exchanger(r.hot_flow_Lmin, r.cold_flow_Lmin, r.hot_in_C, r.cold_in_C)
            for r in data.itertuples()]
    for t in targets:
        table[f"{t}_physics"] = [p[t] for p in phys]

    # ML twin, leave-one-out
    X = data[list(features)].values
    for t in targets:
        y = data[t].values
        preds = np.empty(len(y))
        for train, test in LeaveOneOut().split(X):
            preds[test] = Ridge(alpha=ridge_alpha).fit(X[train], y[train]).predict(X[test])
        table[f"{t}_ml"] = preds

    rows = []
    for t in targets:
        for m in ("physics", "ml"):
            err = table[f"{t}_{m}"] - table[t]
            rows.append({"output": t, "model": m, "MAE_C": err.abs().mean(),
                         "bias_C": err.mean(), "max_abs_err_C": err.abs().max()})
    summary = pd.DataFrame(rows)
    return table, summary


# ---------------------------------------------------------------------------
# SELF-TEST — sanity-check against the reference dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Geometry (derived) ===")
    for k in ["Di_inner_m", "ID_outer_m", "Ah_annulus_m2", "Dh_annulus_m", "A_tube_flow_m2"]:
        print(f"{k:20s}: {GEOMETRY[k]:.6f}")

    print("\n=== Physics model prediction, one test case ===")
    result = predict_heat_exchanger(hot_flow_Lmin=5, cold_flow_Lmin=5,
                                     hot_in_C=50.3, cold_in_C=11.2, config="counter")
    for k, v in result.items():
        print(f"{k:25s}: {v:.4f}")

    print("\n=== Compare prediction to reference run #6 (5 L/min hot, 5 L/min cold) ===")
    ref_row = REFERENCE_COUNTER_CURRENT[REFERENCE_COUNTER_CURRENT.run == 6].iloc[0]
    print(f"Measured:  hot_mid={ref_row.hot_mid_C}, hot_out={ref_row.hot_out_C}, "
          f"cold_mid={ref_row.cold_mid_C}, cold_out={ref_row.cold_out_C}")
    print(f"Predicted: hot_mid={result['hot_mid_C']:.2f}, hot_out={result['hot_out_C']:.2f}, "
          f"cold_mid={result['cold_mid_C']:.2f}, cold_out={result['cold_out_C']:.2f}")

    print("\n=== Wilson's Method on reference dataset ===")
    ref_UA = compute_UA_from_measurements(REFERENCE_COUNTER_CURRENT)
    print(f"Mean energy balance closure: {ref_UA['energy_balance_closure'].mean()*100:.1f}%")
    # Hold cold flow ~constant (use cold_flow==5 subset) to isolate hot-side effect, etc.
    # For a quick full-dataset check (mixing both factors), just show the fit:
    wilson_hot = wilson_analysis(ref_UA[ref_UA.cold_flow_Lmin == 5], stream="hot")
    wilson_cold = wilson_analysis(ref_UA[ref_UA.hot_flow_Lmin == 5], stream="cold")
    print("Hot-side Wilson fit (cold flow held at 5 L/min):", wilson_hot)
    print("Cold-side Wilson fit (hot flow held at 5 L/min):", wilson_cold)

    print("\n=== DOE trial log template (4x4, counter-current) ===")
    print(build_doe_trial_log().to_string(index=False))

    # ---- Your data (2026-10-12) ----
    mine = compute_UA_from_measurements(MEASURED_2026_10_12)
    valid = mine[mine.use_in_fit]
    print("\n=== YOUR DATA (2026-10-12): energy balance ===")
    print(mine[["trial", "hot_flow_Lmin", "cold_flow_Lmin", "Q_hot_W", "Q_cold_W",
                "energy_balance_closure", "UA_W_K", "use_in_fit"]].round(3).to_string(index=False))
    print(f"Mean closure, trials used in fit: {valid['energy_balance_closure'].mean()*100:.1f}%  "
          f"(all 16: {mine['energy_balance_closure'].mean()*100:.1f}%)")

    print("\n=== YOUR DATA: combined Wilson fit (trials used in fit) ===")
    for k, v in wilson_combined(valid).items():
        print(f"{k:24s}: {v:.6g}")

    print("\n=== YOUR DATA: physics UA vs measured UA ===")
    ua_pred = [predict_heat_exchanger(r.hot_flow_Lmin, r.cold_flow_Lmin, r.hot_in_C, r.cold_in_C)["UA_W_K"]
               for r in valid.itertuples()]
    print(f"Mean UA over-prediction: {np.mean(np.array(ua_pred) / valid['UA_W_K'].values - 1)*100:.1f}%")

    print("\n=== YOUR DATA: physics vs ML (leave-one-out) vs actual ===")
    per_trial, summary = compare_models(MEASURED_2026_10_12[MEASURED_2026_10_12.use_in_fit])
    print(summary.round(2).to_string(index=False))
