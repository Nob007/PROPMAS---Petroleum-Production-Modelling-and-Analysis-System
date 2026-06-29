"""
gas_lift/gl.py  —  Gas Lift Analysis Engine
=============================================
Provides:
  - apply_gas_lift_design(vlp_obj, injection_depth, injection_rate_mscf, injection_sg)
        → A VLP-wrapping function (same style as calibration.calibrate.apply_calibration_factors)
          that splits the GLR profile above/below the injection depth inside
          calculate_pressure_traverse. No changes to HagedornBrown / Beggs_Brill
          holdup or friction correlations.

  - compute_glpc(state, sweep_param, sweep_values, build_ipr_fn, build_vlp_fn, build_pvt_fn,
                 get_fp_fn, find_op_fn)
        → The shared sweep engine behind the three Gas Lift tabs
          (depth / rate / GLR). Loops find_operating_points exactly as Sensitivity does.

Units:
    injection_rate : Mscf/day  (1 Mscf = 1000 scf)
    injection_depth: ft
    injection_sg   : gas specific gravity (air = 1)
    GLR            : scf/STB
    q              : STB/day
"""

import copy
import numpy as np
from typing import Any, Callable, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  VLP WRAPPER — applies gas lift injection inside calculate_pressure_traverse
# ─────────────────────────────────────────────────────────────────────────────

def apply_gas_lift_design(
    vlp_obj: Any,
    injection_depth: float,
    injection_rate_mscf: float,
    injection_sg: float = 0.65,
) -> Any:
    """
    Wrap a VLP model so that every call to calculate_pressure_traverse
    incorporates a continuous gas lift injection.

    The wrapper adds injected gas (as an equivalent free-gas GLR increment)
    **above** the injection depth (i.e., from surface to injection_depth).
    Below the injection point, the well flows at its natural producing GOR.

    Modelled mechanism:
        For each depth step z:
            if z <= injection_depth:
                effective_GLR = natural_gor + (injection_rate_scf / Ql_oil)
            else:
                effective_GLR = natural_gor
        The effective_GLR is substituted into self.fp["producing_gor"] before
        fluid properties are updated at that depth step. Both HagedornBrown
        and Beggs_Brill key off producing_gor inside fluid_properties_dict.

    Args:
        vlp_obj             : An instantiated VLP model (HagedornBrown or Beggs_Brill).
        injection_depth     : Injection point measured depth, ft.
        injection_rate_mscf : Injection gas rate, Mscf/day (thousands of scf/day).
        injection_sg        : Injected gas specific gravity (air=1). Reserved for
                              future density corrections; not used in the GLR split.

    Returns:
        A deep-copied VLP model whose calculate_pressure_traverse is patched
        to simulate gas lift.
    """
    gl_obj = copy.deepcopy(vlp_obj)

    # Store original traverse method
    _original_traverse = gl_obj.calculate_pressure_traverse

    # Convert injection rate to scf/day
    injection_rate_scf = injection_rate_mscf * 1000.0

    def _gl_traverse(Pth, surface_temp, bottomhole_temp,
                     total_depth, step_size, Ql, **kwargs):
        """
        Patched pressure traverse that applies gas injection above injection_depth.

        We replicate the Euler integration loop, modifying the fluid properties
        at each depth step based on the GLR split model. The original traverse
        is NOT called — we reproduce the integration here so we can inject
        per-step, without touching HagedornBrown/Beggs_Brill internals.
        """
        depths = [0.0]
        pressures = [Pth]

        holdups = []
        frictions = []
        hydro_losses = []
        fric_losses = []
        total_gradients = []

        current_P = Pth
        current_depth = 0.0
        temp_gradient = (bottomhole_temp - surface_temp) / max(total_depth, 1.0)

        # Natural GOR (constant; we'll modify fp["producing_gor"] per step)
        natural_gor = gl_obj.fp.get("producing_gor", gl_obj.fp.get("gor", 500.0))

        while current_depth < total_depth:
            next_depth = min(current_depth + step_size, total_depth)
            actual_step = next_depth - current_depth
            current_temp = surface_temp + temp_gradient * current_depth

            # ── Gas-lift GLR split ──────────────────────────────────────────
            # Above (≤) the injection depth: augment GOR with injected gas.
            # Ql_oil = Ql * (1 - wc); guard against div-by-zero.
            Ql_oil = Ql * (1.0 - getattr(gl_obj, "wc", 0.0))
            if current_depth <= injection_depth and Ql_oil > 1e-6:
                injected_gor = injection_rate_scf / Ql_oil   # scf/STB oil
                effective_gor = natural_gor + injected_gor
            else:
                effective_gor = natural_gor

            # Temporarily override producing GOR in the fp dict
            gl_obj.fp["producing_gor"] = effective_gor

            # ── Update fluid properties and compute gradient ────────────────
            # HagedornBrown uses update_fluid_properties; Beggs_Brill uses _update_fluid_properties
            if hasattr(gl_obj, "_update_fluid_properties"):
                gl_obj._update_fluid_properties(current_P, current_temp, Ql)
            elif hasattr(gl_obj, "update_fluid_properties"):
                gl_obj.update_fluid_properties(current_P, current_temp, Ql)

            # The calculate_gradient method in HagedornBrown requires Ql, but Beggs_Brill and DunsRos do not.
            # We check the method signature to pass it only when needed.
            import inspect
            sig = inspect.signature(gl_obj.calculate_gradient)
            if 'Ql' in sig.parameters:  # HagedornBrown
                # HagedornBrown returns more values, which we unpack safely
                dp_dz, Hl, f, dp_h, dp_f, *_ = gl_obj.calculate_gradient(current_P, Ql, return_components=True)
            else:  # Beggs_Brill and DunsRos
                dp_dz, Hl, f, dp_h, dp_f = gl_obj.calculate_gradient(current_P, return_components=True)

            current_P += dp_dz * actual_step
            current_depth = next_depth

            depths.append(current_depth)
            pressures.append(current_P)
            holdups.append(Hl)
            frictions.append(f)
            hydro_losses.append(dp_h)
            fric_losses.append(dp_f)
            total_gradients.append(dp_dz)

        # Restore natural GOR for subsequent calls
        gl_obj.fp["producing_gor"] = natural_gor

        if holdups:
            holdups.insert(0, holdups[0])
            frictions.insert(0, frictions[0])
            hydro_losses.insert(0, hydro_losses[0])
            fric_losses.insert(0, fric_losses[0])
            total_gradients.insert(0, total_gradients[0])
        else:
            holdups = [0.0]; frictions = [0.0]
            hydro_losses = [0.0]; fric_losses = [0.0]; total_gradients = [0.0]

        profiles = {
            "holdup": holdups,
            "friction_factor": frictions,
            "hydrostatic_loss": hydro_losses,
            "frictional_loss": fric_losses,
            "total_gradient": total_gradients,
        }
        return depths, pressures, profiles

    # Monkey-patch the deep-copied object's traverse method
    gl_obj.calculate_pressure_traverse = _gl_traverse

    # Tag the object so callers can inspect it
    gl_obj._gl_injection_depth = injection_depth
    gl_obj._gl_injection_rate_mscf = injection_rate_mscf
    gl_obj._gl_injection_sg = injection_sg

    return gl_obj


# ─────────────────────────────────────────────────────────────────────────────
#  GAS LIFT PERFORMANCE CURVE SWEEP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_glpc(
    state: Any,
    sweep_param: str,
    sweep_values: list,
    build_ipr_fn: Callable,
    build_vlp_fn: Callable,
    build_pvt_fn: Callable,
    get_fp_fn: Callable,
    find_op_fn: Callable,
    gl_inj_depth: Optional[float] = None,
    gl_inj_rate_mscf: Optional[float] = None,
    gl_sg_gas: Optional[float] = None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Shared sweep engine for Gas Lift Tabs 1 (depth), 2 (rate), 3 (GLR).

    For each swept value, builds a gas-lifted VLP object (via apply_gas_lift_design),
    runs find_operating_points, and returns the resulting rate and Pwf.

    Args:
        state          : AppState snapshot (deep-copied internally per iteration)
        sweep_param    : One of 'depth', 'rate', 'glr'
            'depth' : sweeps injection depth (ft); injection rate fixed at gl_inj_rate_mscf
            'rate'  : sweeps injection rate (Mscf/day); depth fixed at gl_inj_depth
            'glr'   : sweeps total GLR above injection (scf/STB) — converts to equiv. injection rate
        sweep_values   : List of values to sweep (same unit as sweep_param)
        build_ipr_fn   : Callable(state) → (ipr_obj, err_str)
        build_vlp_fn   : Callable(state, pvt_obj, fp_dict) → (vlp_obj, err_str)
        build_pvt_fn   : Callable(state) → pvt_obj
        get_fp_fn      : Callable(state, pvt_obj) → fp_dict
        find_op_fn     : Callable(ipr_model, vlp_model, vlp_params, pr, q_min, q_max) → NodalResult
        gl_inj_depth   : Fixed injection depth (ft) — used when sweep_param ∈ {'rate', 'glr'}
        gl_inj_rate_mscf: Fixed injection rate (Mscf/day) — used when sweep_param == 'depth'
        gl_sg_gas      : Injected gas SG — defaults to state.sg_gas
        progress_callback: Optional Callable(pct: int, msg: str)

    Returns:
        dict with keys:
            sweep_param  : echo of input
            sweep_values : list of swept values
            q_results    : list of operating rates (STB/day) or None per swept value
            pwf_results  : list of operating Pwf (psia) or None per swept value
            baseline_q   : no-lift operating rate (STB/day) or None
            baseline_pwf : no-lift Pwf (psia) or None
            errors       : list of error strings per swept value ('' = success)
    """
    sweep_param_lower = sweep_param.lower()
    if sweep_param_lower not in ("depth", "rate", "glr"):
        raise ValueError(f"sweep_param must be 'depth', 'rate', or 'glr'; got '{sweep_param}'")

    sg_gas_inj = gl_sg_gas if gl_sg_gas is not None else getattr(state, "sg_gas", 0.65)

    results_q = []
    results_pwf = []
    errors = []

    total = len(sweep_values)

    # ── No-lift baseline ────────────────────────────────────────────────────
    baseline_q = None
    baseline_pwf = None
    try:
        base_pvt = build_pvt_fn(state)
        base_fp  = get_fp_fn(state, base_pvt)
        base_vlp, _ = build_vlp_fn(state, base_pvt, base_fp)
        base_ipr, _ = build_ipr_fn(state)
        if base_vlp and base_ipr:
            vlp_params_base = {
                "Pth": state.thp, "surface_temp": state.T_surface,
                "bottomhole_temp": state.T_bh, "depth": state.depth,
                "step_size": state.dz_step,
            }
            nr_base = find_op_fn(
                ipr_model=base_ipr, vlp_model=base_vlp,
                vlp_params=vlp_params_base, pr=state.Pr,
                q_min=state.q_min, q_max=min(base_ipr.q_max, state.q_max_sweep),
            )
            if nr_base.success and nr_base.stable_point:
                baseline_q   = nr_base.stable_point.rate
                baseline_pwf = nr_base.stable_point.pwf
    except Exception as e:
        pass  # baseline failure is non-fatal

    # ── Sweep loop ──────────────────────────────────────────────────────────
    for vi, val in enumerate(sweep_values):
        pct = int((vi + 1) / total * 100)

        # Determine injection depth and rate for this iteration
        if sweep_param_lower == "depth":
            inj_depth = float(val)
            inj_rate  = float(gl_inj_rate_mscf) if gl_inj_rate_mscf else 500.0
        elif sweep_param_lower == "rate":
            inj_depth = float(gl_inj_depth) if gl_inj_depth else (state.depth or 5000.0) * 0.55
            inj_rate  = float(val)   # Mscf/day
        else:  # glr
            inj_depth = float(gl_inj_depth) if gl_inj_depth else (state.depth or 5000.0) * 0.55
            # Convert total GLR to injection rate:
            #   injected_gor = total_glr - natural_gor  (scf/STB)
            #   injection_rate_mscf = injected_gor * Ql_oil / 1000
            # Ql_oil is unknown at this point, so use state test rate as a proxy
            natural_gor = getattr(state, "gor", 500.0) or 500.0
            q_proxy = getattr(state, "Qo_test", 1000.0) or 1000.0
            injected_gor = max(0.0, float(val) - natural_gor)
            inj_rate = injected_gor * q_proxy * (1.0 - getattr(state, "wc", 0.0)) / 1000.0

        try:
            pvt = build_pvt_fn(state)
            fp  = get_fp_fn(state, pvt)
            vlp_base, err = build_vlp_fn(state, pvt, fp)
            if vlp_base is None:
                raise RuntimeError(f"VLP build failed: {err}")

            ipr_obj, err = build_ipr_fn(state)
            if ipr_obj is None:
                raise RuntimeError(f"IPR build failed: {err}")

            # Wrap VLP with gas lift
            vlp_gl = apply_gas_lift_design(
                vlp_base, inj_depth, inj_rate, sg_gas_inj
            )

            vlp_params = {
                "Pth": state.thp, "surface_temp": state.T_surface,
                "bottomhole_temp": state.T_bh, "depth": state.depth,
                "step_size": state.dz_step,
            }

            nr = find_op_fn(
                ipr_model=ipr_obj, vlp_model=vlp_gl,
                vlp_params=vlp_params, pr=state.Pr,
                q_min=state.q_min,
                q_max=min(ipr_obj.q_max, state.q_max_sweep),
            )

            if nr.success and nr.stable_point:
                results_q.append(nr.stable_point.rate)
                results_pwf.append(nr.stable_point.pwf)
            else:
                results_q.append(None)
                results_pwf.append(None)
            errors.append("")

        except Exception as e:
            results_q.append(None)
            results_pwf.append(None)
            errors.append(str(e))

        if progress_callback:
            progress_callback(pct, f"{sweep_param_lower}={val:.3g}")

    return {
        "sweep_param":  sweep_param_lower,
        "sweep_values": sweep_values,
        "q_results":    results_q,
        "pwf_results":  results_pwf,
        "baseline_q":   baseline_q,
        "baseline_pwf": baseline_pwf,
        "errors":       errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMUM FINDER
# ─────────────────────────────────────────────────────────────────────────────

def find_optimum(sweep_result: dict, econ_slope: float = 0.5) -> dict:
    """
    Identify the optimum swept value from a gas lift performance curve.

    The optimum is where the marginal gain in liquid rate per unit increase
    in swept parameter drops below `econ_slope` (STB/day per Mscf/day for
    rate sweep, or STB/day per ft of depth, etc.).

    For maximum-rate search (depth sweep), the optimum is simply the
    depth giving the highest q*.

    Args:
        sweep_result : Output of compute_glpc()
        econ_slope   : Economic threshold (STB/day per sweep-unit)

    Returns:
        dict with keys:
            opt_value   : Optimal swept value
            opt_q       : Corresponding q* (STB/day)
            opt_pwf     : Corresponding Pwf* (psia)
            opt_index   : Index in sweep_values
            method      : 'econ_slope' or 'max_rate'
    """
    q_vals = sweep_result["q_results"]
    sv     = sweep_result["sweep_values"]

    # Filter out None results
    valid = [(sv[i], q_vals[i]) for i in range(len(sv)) if q_vals[i] is not None]
    if not valid:
        return {"opt_value": None, "opt_q": None, "opt_pwf": None, "opt_index": None, "method": "none"}

    param = sweep_result["sweep_param"]
    pwf_vals = sweep_result["pwf_results"]

    if param == "rate":
        # Use econ slope: dq/d(inj_rate) < econ_slope → optimum
        opt_idx = len(valid) - 1  # default: last point (max rate if monotone)
        for i in range(1, len(valid)):
            dq   = valid[i][1] - valid[i-1][1]
            dinj = valid[i][0] - valid[i-1][0]
            if dinj > 0 and dq / dinj < econ_slope:
                opt_idx = i - 1
                break
        sv_val, q_opt = valid[opt_idx]
        global_idx = sv.index(sv_val)
        return {
            "opt_value": sv_val,
            "opt_q":     q_opt,
            "opt_pwf":   pwf_vals[global_idx],
            "opt_index": global_idx,
            "method":    "econ_slope",
        }
    else:
        # For depth and GLR sweeps: find the point with highest q*
        sv_val, q_opt = max(valid, key=lambda x: x[1])
        global_idx = sv.index(sv_val)
        return {
            "opt_value": sv_val,
            "opt_q":     q_opt,
            "opt_pwf":   pwf_vals[global_idx],
            "opt_index": global_idx,
            "method":    "max_rate",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  INJECTION FEASIBILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def injection_feasibility_mask(
    sweep_result: dict,
    p_inj_surface: float,
    sg_gas_inj: float = 0.65,
    state_depth: float = 8000.0,
    p_flowing_at_depth: Optional[List[float]] = None,
) -> list:
    """
    For each swept value, determine whether the injection pressure is
    sufficient to inject at the given depth/rate.

    Simplified check:
        Static gas column pressure at injection depth:
            P_inj_BH ≈ P_inj_surface + rho_gas × depth_inj / 144  (psia)
        This must exceed the flowing tubing pressure at the injection depth.
        If p_flowing_at_depth is not supplied, we approximate it as
        the average of THP and bottomhole pressure (coarse estimate).

    Args:
        sweep_result         : Output of compute_glpc()
        p_inj_surface        : Available surface injection pressure, psia
        sg_gas_inj           : Injected gas SG
        state_depth          : Total well depth, ft (fallback)
        p_flowing_at_depth   : If provided, a list of flowing pressures at the
                               injection depth per swept value (one per sweep point).

    Returns:
        List of bool (True = feasible, False = injection pressure insufficient)
    """
    sv     = sweep_result["sweep_values"]
    param  = sweep_result["sweep_param"]

    feasible = []
    for i, val in enumerate(sv):
        if param == "depth":
            inj_depth = float(val)
        else:
            inj_depth = state_depth * 0.55  # approximate mid-point

        # Hydrostatic gas column from surface to injection depth
        rho_gas_lbm_ft3 = 0.0764 * sg_gas_inj   # approx at surface (P≈14.7)
        p_inj_bh = p_inj_surface + rho_gas_lbm_ft3 * inj_depth / 144.0  # psia

        # Flowing pressure at injection depth
        if p_flowing_at_depth and i < len(p_flowing_at_depth):
            p_flowing = p_flowing_at_depth[i]
        else:
            # Very coarse approximation: linear interpolation of THP → Pwf
            pwf = sweep_result["pwf_results"][i]
            if pwf is not None:
                p_flowing = pwf * (inj_depth / max(state_depth, 1.0))
            else:
                feasible.append(False)
                continue

        feasible.append(p_inj_bh >= p_flowing)

    return feasible
