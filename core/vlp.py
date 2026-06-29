import numpy as np
import matplotlib.pyplot as plt


_H_PTS     = np.array([1e-7, 1e-6, 3e-6, 1e-5, 7.12e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 1.0])
_HLPSI_PTS = np.array([0.03, 0.08, 0.09, 0.20, 0.44,    0.4896, 0.65, 0.80, 0.92, 0.98, 1.00])
_LOG_H_PTS = np.log10(_H_PTS)


def _holdup_over_psi(H):
    """
    HL/psi = f(H) — Hagedorn-Brown Chart 3 / Takács Fig 2-30.

    Log-log digitized lookup (np.interp clamps to the endpoint values
    outside the table range), in place of the old rational-polynomial
    fit which severely overestimated holdup at small H.
    """
    H_safe = max(H, 1e-9)
    log_H = np.log10(H_safe)
    return float(np.interp(log_H, _LOG_H_PTS, _HLPSI_PTS))


def _psi_correction(B):
    """
    psi correction factor = f(B) — Hagedorn-Brown Chart 4 / Fig 2-32.

    Same three polynomial branches as before, but blended across a
    small window around each branch boundary (B=0.025, B=0.055) so the
    piecewise fit doesn't introduce its own small step discontinuities.
    """
    def branch1(b): return 27170*b**3 - 317.52*b**2 + 0.5472*b + 0.9999
    def branch2(b): return -533.33*b**2 + 58.524*b + 0.1171
    def branch3(b): return 2.5714*b + 1.5962

    w = 0.0015  # half-width of the blend window around each boundary
    if B <= 0.025 - w:
        psi = branch1(B)
    elif B <= 0.025 + w:
        t = (B - (0.025 - w)) / (2*w)
        psi = (1.0 - t)*branch1(B) + t*branch2(B)
    elif B <= 0.055 - w:
        psi = branch2(B)
    elif B <= 0.055 + w:
        t = (B - (0.055 - w)) / (2*w)
        psi = (1.0 - t)*branch2(B) + t*branch3(B)
    else:
        psi = branch3(B)

    return max(psi, 1.0)
    # if B <= 0.025:
    #     return max(27170*B**3 - 317.52*B**2 + 0.5472*B + 0.9999, 1.0)
    # elif B <= 0.055:
    #     return max(-533.33*B**2 + 58.524*B + 0.1171, 1.0)
    # else:
    #     return max(2.5714*B + 1.5962, 1.0)



class HagedornBrown:
    """
    Calculates multiphase flow pressure gradients and VLP curves for vertical wellbores
    using the Hagedorn-Brown correlation and a coupled Black Oil PVT model.
    Includes Griffith-Wallis bubble flow detection and holdup correction.
    """

    def __init__(self, tubing_id, tubing_od, casing_id, roughness, pvt_model,
                 fluid_properties, watercut=0.0, theta=0.0):
        self.tid      = tubing_id #ft
        self.tod      = tubing_od #ft
        self.cid      = casing_id #ft
        self.roughness = roughness #ft
        self.pvt_model = pvt_model
        self.wc       = watercut
        self.wor      = watercut / (1.0 - watercut + 1e-9)
        self.theta    = np.radians(theta)
        self.Ap       = (np.pi / 4.0) * self.tid**2
        self.fp       = fluid_properties

    # ------------------------------------------------------------------
    # Fluid property update
    # ------------------------------------------------------------------

    def update_fluid_properties(self, P, T, Ql):
        self.Ql = Ql
        self.fp = self.pvt_model.fluid_properties_dict(
            P, T, self.fp["Rsb"], self.fp["producing_gor"], self.fp["Pb"]
        )

    # ------------------------------------------------------------------
    # Hagedorn-Brown dimensionless groups
    # ------------------------------------------------------------------

    def dimensionless_numbers(self):
        """
        Calculates the required Hagedorn-Brown dimensionless groups and
        sets Vsl, Vsg, Vm in self.fp.

        Returns:
            tuple: Nl, CNl, Nlv, Ngv, Nd (all dimensionless).
        """
        fo = 1.0 / (1.0 + self.wor)
        fw = self.wor  / (1.0 + self.wor)

        # Ensure GLR is always available in the fluid properties dictionary
        if "glr" not in self.fp:
            self.fp["glr"] = self.fp.get("producing_gor", 0.0) / (1.0 + self.wor)

        # In-situ liquid volumetric rate  [ft³/day]
        q_liquid_insitu = 5.615 * self.Ql * (self.fp["Bo"] * fo + self.fp["Bw"] * fw)
        self.fp["Vsl"]  = q_liquid_insitu / (86400.0 * self.Ap)

        free_gas_scf   = max(0.0,  (self.fp["producing_gor"] - self.fp["gor"]) * self.Ql * (1.0 - self.wc))
        q_gas_insitu   = (free_gas_scf
                          * (14.7 / self.fp["Pr"])
                          * ((self.fp["Tr"] + 460.0) / 520.0)
                          * self.fp["Z"])
        self.fp["Vsg"] = q_gas_insitu / (86400.0 * self.Ap)
        self.fp["Vm"]  = self.fp["Vsl"] + self.fp["Vsg"]

        # H-B dimensionless groups
        base_term = 1.0 / (self.fp["rho_l"] * self.fp["sigma_l"] ** 3)
        Nl  = 0.15726 * self.fp["mu_l"] * (base_term ** 0.25)
        CNl = 0.061 * Nl**3 - 0.0929 * Nl**2 + 0.0505 * Nl + 0.0019

        Nlv = 1.938 * self.fp["Vsl"] * (self.fp["rho_l"] / self.fp["sigma_l"]) ** 0.25
        Ngv = 1.938 * self.fp["Vsg"] * (self.fp["rho_l"] / self.fp["sigma_l"]) ** 0.25
        Nd  = 120.872 * self.tid     * (self.fp["rho_l"] / self.fp["sigma_l"]) ** 0.5

        return Nl, CNl, Nlv, Ngv, Nd

    # ------------------------------------------------------------------
    # Bubble-flow regime detection  (Griffith & Wallis, 1961)
    # ------------------------------------------------------------------

    def is_bubble_flow(self):
        """
        Determines whether the current flow conditions fall in the bubble-flow
        regime using the Griffith-Wallis criterion.

        Bubble flow exists when the in-situ gas void fraction (λg = Vsg/Vm)
        is less than the boundary value LB:
            LB = max(0.25,  1.071 − 0.2218 · Vm² / d)

        Velocities must be computed by dimensionless_numbers() before calling
        this method (they are stored in self.fp).

        Returns:
            bool: True if bubble flow, False otherwise.
        """
        Vm  = self.fp.get("Vm",  0.0)
        Vsg = self.fp.get("Vsg", 0.0)

        if Vm <= 1e-9:
            return False, 0., 0.

        LB = 1.071 - 0.2218 * (Vm ** 2) / (self.tid*12)
        LB = max(LB, 0.13)

        lambda_g = Vsg / Vm          # in-situ gas void fraction (no-slip)
        return lambda_g < LB, lambda_g, LB

    # ------------------------------------------------------------------
    # Griffith holdup for bubble flow
    # ------------------------------------------------------------------

    def griffith_holdup(self):
        """
        Calculates liquid holdup for bubble-flow regime using the
        Griffith-Wallis correlation and updates mixture properties in self.fp.

        The bubble-rise velocity Vs = 0.8 ft/s is a standard field-unit
        constant for oil–gas systems.

        Quadratic form (Griffith & Wallis):
            Vs·Hl² − (Vm + Vs)·Hl + Vsl = 0
        Solved directly:
            Hl = 1 − ½·[1 + Vm/Vs − √((1 + Vm/Vs)² − 4·Vsg/Vs)]

        Returns:
            float: Liquid holdup (0.0 – 1.0).
        """
        Vm  = self.fp["Vm"]
        Vsg = self.fp["Vsg"]
        Vs  = 0.8   # bubble rise velocity, ft/s

        discriminant = (1.0 + Vm / Vs) ** 2 - 4.0 * Vsg / Vs
        # Discriminant is always ≥ 0 inside bubble-flow region (λg < LB ≤ 0.25)
        discriminant = max(discriminant, 0.0)

        Hl = 1.0 - 0.5 * (1.0 + Vm / Vs - np.sqrt(discriminant))
        Hl = max(0.0, min(Hl, 1.0))

        return max(min(Hl, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Hagedorn-Brown holdup (slug / transition / mist)
    # ------------------------------------------------------------------

    def liquid_holdup(self, Nl, CNl, Nlv, Ngv, Nd):
        """
        Calculates liquid holdup using the Hagedorn-Brown correlation.
        Also sets rho_m and mu_m in self.fp.
        Args:
            Liquid viscosity no
            CNl
            Nlv
            Ngv, 
            Nd

        Returns:
            float: Liquid holdup (0.0 – 1.0).
        """
        # # Ngv_safe = max(Ngv, 1e-6)

        # # H = ((Nlv / (Ngv_safe ** 0.575))
        # #      * (self.fp["Pr"] / 14.7) ** 0.1
        # #      * (CNl / Nd))
        
        # # H = max(H, 5e-3)

        # # Hl_psi = np.sqrt(
        # #     (0.0047 + 1123.32 * H + 729489.64 * H ** 2)
        # #     / (1.0 + 1097.1566 * H + 722153.97 * H ** 2)
        # # )
        # Ngv_safe = max(Ngv, 1e-6)

        # # 1. Calculate the correlating parameter H (The X-axis of the original chart)
        # H = ((Nlv / (Ngv_safe ** 0.575))
        #      * (self.fp["Pr"] / 14.7) ** 0.1
        #      * (CNl / Nd))
        # H_safe = max(H, 1e-6)

        # # Digitized Chart for Hl/Psi to prevent polynomial crashing at mist flow
        # chart_H = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5,  1.0, 10.0])
        # chart_Hl_psi = np.array([0.004, 0.008, 0.012, 0.025, 0.035, 0.07, 0.11, 0.30, 0.45, 0.82, 0.94, 1.0])
        # Hl_psi = np.interp(H, chart_H, chart_Hl_psi)

        # # B = Ngv * (Nlv ** 0.38) / (Nd ** 2.14)
        # # H_safe = max(H, 1e-6)
        # # Hl_psi = np.sqrt(
        # #     (0.0047 + 1123.32 * H_safe + 729489.64 * H_safe ** 2)
        # #     / (1.0 + 1097.1566 * H_safe + 722153.97 * H_safe ** 2)
        # #     )
        # B = Ngv * (Nlv ** 0.38) / (Nd ** 2.14)   # Nlv, not Nl
        # if B <= 0.025:
        #     psi = 27170 * B**3 - 317.52 * B**2 + 0.5472 * B + 0.9999
        # elif B <= 0.055:
        #     psi = -533.33 * B**2 + 58.524 * B + 0.1171
        # else:
        #     psi = 2.5714 * B + 1.5962

        # Hl = max(0.0, min(Hl_psi * psi, 1.0))

        # self.fp["rho_m"] = self.fp["rho_l"] * Hl + self.fp["rho_g"] * (1.0 - Hl)
        # self.fp["mu_m"]  = (self.fp["mu_l"] ** Hl) * (self.fp["mu_g"] ** (1.0 - Hl))

        # return Hl

        Ngv_safe = max(Ngv, 1e-6)
        Nl = max(min(Nl, 10.0), 1e-6)

        H = ((Nlv / (Ngv_safe ** 0.575))* (self.fp["Pr"] / 14.7) ** 0.1 * (CNl / Nd)) if (Ngv > 0 and Nd > 0) else 0.0

        Hl_psi = _holdup_over_psi(H)

        B = Ngv * (Nlv ** 0.38) / (Nd ** 2.14)  if Nd > 0 else 0.0
        psi = _psi_correction(B)

        Hl = max(0.0, min(Hl_psi * psi, 1.0))

        return Hl, H, Hl_psi, B, psi

    # ------------------------------------------------------------------
    # Combined holdup dispatcher
    # ------------------------------------------------------------------

    def get_holdup(self):
        """
        Routes to the correct holdup correlation based on flow regime.

        Calls dimensionless_numbers() first so that Vm / Vsg are current,
        then checks for bubble flow. Returns Hl from whichever method applies.

        Returns:
            float: Liquid holdup (0.0 – 1.0).
        """
        Nl, CNl, Nlv, Ngv, Nd = self.dimensionless_numbers() # These are the values we need

        is_bubble, lambda_g, LB = self.is_bubble_flow()

        Cl = self.fp["Vsl"] / max(self.fp["Vm"], 1e-6)
        Hl_hagedron, H, Hl_psi, B, psi = self.liquid_holdup(Nl, CNl, Nlv, Ngv, Nd)
        Hl_griffith = self.griffith_holdup()
        if is_bubble and lambda_g < 0.7 * LB:
            Hl = Hl_griffith
        elif lambda_g>LB:
            Hl = Hl_hagedron
        else:
            alpha = (lambda_g - 0.7 * LB)/max(LB - 0.7 * LB, 1e-10)
            Hl = alpha * Hl_hagedron + (1.0 - alpha) * Hl_griffith
        Hl = max(min(Hl, 1.0), Cl)

        self.fp["rho_m"] = self.fp["rho_l"] * Hl + self.fp["rho_g"] * (1.0 - Hl)
        self.fp["mu_m"]  = (self.fp["mu_l"] ** Hl) * (self.fp["mu_g"] ** (1.0 - Hl))
        # return 1.0 * Hl + 0.0 * self.griffith_holdup()
        return Hl, H, Hl_psi, B, psi, Nl, Nlv, Ngv, Nd

    # ------------------------------------------------------------------
    # Friction factor  (Jain / Colebrook approximation)
    # ------------------------------------------------------------------

    def frictional_factor(self, Hl: float) -> float:
        """
        Calculates the two-phase Darcy friction factor using the Jain (1976)
        explicit approximation to the Colebrook-White equation.

        Uses the standard oilfield Re definition:
            Re = 1488 * rho_ns [lbm/ft³] * Vm [ft/s] * d [ft] / mu_m [cp]

        Args:
            Hl (float): Liquid holdup fraction (dimensionless), passed in from
                    calculate_gradient() to avoid a redundant holdup call.

        Returns:
            float: Darcy-Weisbach friction factor (dimensionless).
        """
        lambda_l = self.fp["Vsl"] / max(self.fp["Vm"], 1e-6)

    # No-slip mixture density — weighted by input-liquid fraction [lbm/ft³]
        self.fp["rho_ns"] = (
        self.fp["rho_l"] * lambda_l
        + self.fp["rho_g"] * (1.0 - lambda_l)
        )

    # Hagedorn-Brown mixture viscosity — weighted exponential [cp]
        mu_m = (
        self.fp["mu_l"] ** Hl
        * self.fp["mu_g"] ** (1.0 - Hl)
        )

        Re = (
        1488.0
        * self.fp["rho_ns"]   # [lbm/ft³]
        * self.fp["Vm"]       # [ft/s]
        * self.tid            # [ft]
        / mu_m                # [cp]
        )

        if Re<=0: return 0.025
        elif Re < 2100:
            return 64.0 / max(Re, 1.0)
        
        relative_roughness = self.roughness / self.tid   # dimensionless [ft/ft]

        # f = (1.14 - 2.0 * np.log10(relative_roughness + 21.25 / (Re ** 0.9))) ** -2
        f = -4.0 * np.log10(
        relative_roughness / 3.7065
        - (5.0452 / Re) * np.log10(relative_roughness**1.1098 / 2.8257 + (7.149 / Re)**0.8981)
        )
        f = (1.0 / f)**2
        x = max(lambda_l / Hl ** 2, 1e-6) if Hl > 0 else lambda_l
        if ((x > 1) and (x < 1.2)):
            s = np.log(2.2 * x - 1.2)
        else:
            s = np.log(x) / (-0.0523 + 3.182 * np.log(x) - 0.8725 * (np.log(x)) ** 2 + 0.01853 * (np.log(x)) ** 4)
    
        f = f * np.exp(s) 
           # Darcy friction factor, turbulent
        return f

    # ------------------------------------------------------------------
    # Pressure gradient
    # ------------------------------------------------------------------

    def calculate_gradient(self, P,  return_components=False):
        """
        Calculates the total multiphase pressure gradient (psi/ft).

        Routes holdup through get_holdup() which selects Griffith (bubble)
        or Hagedorn-Brown (slug/mist) automatically.

        Returns:
            float: Total pressure gradient in psi/ft.
        """
        Hl, H, Hl_psi, B, psi, Nl, Nlv, Ngv, Nd = self.get_holdup()
        f  = self.frictional_factor(Hl)

        dp_dh_el   = self.fp["rho_m"] * np.cos(self.theta) / 144.0
        gc         = 32.174
        dp_dh_fric = (2 * f * self.fp["rho_ns"] * self.fp["Vm"] ** 2) / ( gc * self.tid * 144.0)
        Ek = self.fp["Vm"] * self.fp["Vsg"] * self.fp["rho_m"] / (32.17 * P * 144 )
        Ek = max(min(Ek, 0.99), 0.0)

        dp_dz = (dp_dh_el + dp_dh_fric)/(1-Ek)
        if return_components:
            return dp_dz, Hl, f, dp_dh_el, dp_dh_fric, H, Hl_psi, B, psi, Nl, Nlv, Ngv, Nd
        return dp_dz

    # ------------------------------------------------------------------
    # Pressure traverse
    # ------------------------------------------------------------------

    def calculate_pressure_traverse(self, Pth, surface_temp, bottomhole_temp,
                                    total_depth, step_size, Ql, gl_depth = 0.0, gl_rate = 0.0):
        """
        Calculates the wellbore pressure profile via Euler integration.

        Returns:
            tuple: (depths [ft], pressures [psia], profiles [dict])
        """
        depths        = [0.0]
        pressures     = [Pth]
        
        holdups = []
        frictions = []
        hydro_losses = []
        fric_losses = []
        total_gradients = []
        
        current_P     = Pth
        current_depth = 0.0
        temp_gradient = (bottomhole_temp - surface_temp) / total_depth

        formation_gor = self.fp["gor"]

        while current_depth < total_depth:
            Qo = Ql * (1-self.wc)
            
            if current_depth <= gl_depth and Qo > 0:
                injected_gor = gl_rate / Qo
                effective_gor = formation_gor + injected_gor
            else:
                effective_gor = formation_gor

            next_depth   = min(current_depth + step_size, total_depth)
            actual_step  = next_depth - current_depth
            current_temp = surface_temp + temp_gradient * current_depth

            self.update_fluid_properties(current_P, current_temp, Ql)
            dp_dz, Hl, f, dp_dh_el, dp_dh_fric, H, Hl_psi, B, psi, Nl, Nlv, Ngv, Nd = self.calculate_gradient(current_P, return_components=True)
            # print(f"Vsg: {self.fp["Vsg"]}, Hl: {Hl}, gor: {self.fp["gor"]}, producing GLR: {self.fp["producing_glr"]} at depth: {current_depth}, Ql: {Ql}")
            
            # Print the detailed debug line at the last depth step for the current flow rate
            # if next_depth == total_depth:
            #     # lambda_g, LB = self.is_bubble_flow()
            #     # Hl, H, Hl_psi, B, psi = self.liquid_holdup(*self.dimensionless_numbers())
            #     # Cl = self.fp["Vsl"] / max(self.fp["Vm"], 1e-6)
            #     print(f"Ql={Ql:.2f} Depth={next_depth:.2f} Vsl={self.fp['Vsl']:.3f} Vsg={self.fp['Vsg']:.3f} gor={self.fp['gor']:.1f} glr={self.fp['glr']:.1f} Nl={Nl:.4f} Nlv={Nlv:.4f} Ngv={Ngv:.4f} Nd={Nd:.4f} H={H:.5f} B={B:.5f} Hl_psi={Hl_psi:.4f} psi={psi:.4f} Hl={Hl:.5f} friction_loss:{dp_dh_fric:.5f} hydro_loss:{dp_dh_el:.5f} total_gradient:{dp_dz:.5f}")

            current_P += dp_dz * actual_step
            current_depth = next_depth

            depths.append(current_depth)
            pressures.append(current_P)
            
            holdups.append(Hl)
            frictions.append(f)
            hydro_losses.append(dp_dh_el)
            fric_losses.append(dp_dh_fric)
            total_gradients.append(dp_dz)

        if holdups:
            holdups.insert(0, holdups[0])
            frictions.insert(0, frictions[0])
            hydro_losses.insert(0, hydro_losses[0])
            fric_losses.insert(0, fric_losses[0])
            total_gradients.insert(0, total_gradients[0])
        else:
            holdups = [0.0]; frictions = [0.0]; hydro_losses = [0.0]; fric_losses = [0.0]; total_gradients = [0.0]
            
        profiles = {
            "holdup": holdups,
            "friction_factor": frictions,
            "hydrostatic_loss": hydro_losses,
            "frictional_loss": fric_losses,
            "total_gradient": total_gradients
        }

        return depths, pressures, profiles

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_pressure_traverse(self, Pth, surface_temp, bottomhole_temp,
                               total_depth, step_size, Ql):
        depths, pressures, _ = self.calculate_pressure_traverse(
            Pth, surface_temp, bottomhole_temp, total_depth, step_size, Ql
        )
        plt.plot(pressures, depths, color='blue', linewidth=2)
        plt.gca().invert_yaxis()
        plt.xlabel('Pressure (psia)')
        plt.ylabel('Depth (ft)')
        plt.title('Pressure Traverse Curve')
        plt.grid(True)
        plt.show()
        return pressures[-1]

    def plot_vlp_curve(self, Pth, surface_temp, bottomhole_temp,
                       depth, Qmin, Qmax, step_size):
        Pwf_points = []
        rates = np.linspace(Qmin, Qmax, Qmax - Qmin)

        for q in rates:
            _, pressures, _ = self.calculate_pressure_traverse(
                Pth, surface_temp, bottomhole_temp, depth, step_size, q
            )
            Pwf_points.append(pressures[-1])

        plt.plot(rates, Pwf_points, color='blue')
        plt.xlabel('Liquid Rate (stb/day)')
        plt.ylabel('Pwf (psi)')
        plt.title('VLP Curve')
        plt.grid(True)
        plt.show()

    def vlp_curve_plot_linear(self, Pth, depth, Qmin, Qmax, step_size):
        """Simplified single-gradient VLP (stale PVT — approximate only)."""
        Pwf_points = []
        rates = np.linspace(Qmin, Qmax, int((Qmax - Qmin) / step_size))

        for q in rates:
            self.Ql = q
            Pwf = Pth + self.calculate_gradient() * depth
            Pwf_points.append(Pwf)

        plt.plot(rates, Pwf_points, color='blue')
        plt.xlabel('Liquid Rate (stb/day)')
        plt.ylabel('Pwf (psi)')




import numpy as np
import matplotlib.pyplot as plt

class Beggs_Brill:
    """
    Calculates multiphase flow pressure gradients and VLP curves for wellbores
    at any inclination angle using the Beggs and Brill (1973) correlation.
    """
    def __init__(self, tubing_id, tubing_od, casing_id, roughness, pvt_model,
                 fluid_properties, watercut=0.0, theta=0.0):
        """
        Initializes the wellbore geometry and base PVT parameters.

        Args:
            tubing_id (float): Tubing inner diameter in feet (ft).
            tubing_od (float): Tubing outer diameter in feet (ft).
            casing_id (float): Casing inner diameter in feet (ft).
            roughness (float): Absolute pipe roughness in feet (ft).
            pvt_model (object): Instance of the BlackOilPVT class.
            fluid_properties (dict): Initial dictionary of PVT properties.
            watercut (float): Watercut as a decimal fraction (0.0 to 1.0).
            theta (float): Well deviation from vertical in degrees (0 = vertical).
        """
        self.tid = tubing_id
        self.tod = tubing_od
        self.cid = casing_id
        self.roughness = roughness
        self.pvt_model = pvt_model 
        self.wc = watercut
        self.Ap = (np.pi / 4.0) * self.tid**2
        self.fp = fluid_properties 
        self.Ql = 0.0
        
        # CORRECTED ANGLE LOGIC
        self.theta_ui = theta                 # UI deviation from vertical (0=vertical)
        self.theta_rad = np.radians(theta)    # Radians for gravity (cos(0) = 1)

    def _update_fluid_properties(self, P, T, Ql):
        """
        Updates the in-situ fluid properties and superficial velocities 
        based on local pressure and temperature.
        """
        self.Ql = Ql
        self.fp = self.pvt_model.fluid_properties_dict(
            P, T, self.fp["Rsb"], self.fp["producing_gor"], self.fp["Pb"]
        )
        self._superficial_velocities()
    
    def _superficial_velocities(self):
        """Calculates in-situ superficial velocities and no-slip liquid holdup (Cl)."""
        fo = (1.0 - self.wc)
        fw = self.wc

        q_liquid_insitu = 5.615 * self.Ql * (self.fp["Bo"] * fo + self.fp["Bw"] * fw)
        self.fp["Vsl"] = q_liquid_insitu / (86400.0 * self.Ap)

        free_gas_scf = max(0.0, self.Ql * (self.fp["glr"] - self.fp["gor"] * fo))
        q_gas_insitu = (free_gas_scf * (14.7 / self.fp["Pr"]) 
                        * ((self.fp["Tr"] + 460.0) / 520.0) * self.fp["Z"])
        
        self.fp["Vsg"] = q_gas_insitu / (86400.0 * self.Ap)
        self.fp["Vm"]  = self.fp["Vsl"] + self.fp["Vsg"]
        
        # No-slip liquid holdup
        self.fp["Cl"] = self.fp["Vsl"] / max(self.fp["Vm"], 1e-6)

    def _dimensionless_numbers(self):
        """Calculates Beggs and Brill specific dimensionless groups."""
        self._superficial_velocities()
        Gm = self.fp["rho_l"] * self.fp["Vsl"] + self.fp["rho_g"] * self.fp["Vsg"]
        
        Cl = max(self.fp["Cl"], 1e-6) # Prevent divide-by-zero
        L1 = 316.0 * Cl ** 0.302
        L2 = 0.0009252 * Cl ** (-2.4684)
        L3 = 0.1 * Cl ** (-1.4516)
        L4 = 0.5 * Cl ** (-6.738)

        # Froude Number
        Nfr = (self.fp["Vm"]**2) / (32.174 * self.tid)

        return Gm, L1, L2, L3, L4, Nfr
    
    def find_flow_pattern(self, L1, L2, L3, L4, Nfr):
        """Determines the Beggs and Brill flow pattern (Segregated, Intermittent, Distributed, Transitional)."""
        Cl = self.fp["Cl"]
        if (Cl < 0.01 and Nfr < L1) or (Cl >= 0.01 and Nfr < L2):
            return "segregated"
        elif (Cl >= 0.01 and L2 <= Nfr <= L3):
            return "transitional"
        elif (0.01 <= Cl < 0.4 and L3 < Nfr <= L1) or (Cl >= 0.4 and L3 < Nfr <= L4):
            return "intermittent"
        elif (Cl < 0.4 and Nfr >= L1) or (Cl >= 0.4 and Nfr > L4):
            return "distributed"
        return "distributed" # Fallback
        
    def calculate_holdup(self, L1, L2, L3, L4, Nfr):
        """
        Calculates the actual liquid holdup fraction (Hl), including angle 
        correction (psi) and transitional blending.
        """
        Cl = max(self.fp["Cl"], 1e-6)
        Nlv = 1.938 * self.fp["Vsl"] * (self.fp["rho_l"] / max(self.fp["sigma_l"], 1e-6))**0.25
        flow_pattern = self.find_flow_pattern(L1, L2, L3, L4, Nfr)
        
        # B&B specifically requires angle from HORIZONTAL for the correction
        theta_h = np.radians(90.0 - self.theta_ui)
        sin_term = np.sin(1.8 * theta_h)

        # def get_C_and_Hl0(pattern):
        #     if pattern == "segregated":
        #         hl0 = 0.98 * (Cl**0.4846 / Nfr**0.0868)
        #         c = (1 - Cl) * np.log(0.011 * Nlv**3.539 * Cl**(-3.768) * Nfr**(-1.614))
        #     elif pattern == "intermittent":
        #         hl0 = 0.845 * (Cl**0.5351 / Nfr**0.0173)
        #         c = (1 - Cl) * np.log(2.96 * Nlv**(-0.4473) * Cl**0.305 * Nfr**(0.0978))
        #     else:
        #         hl0 = 1.065 * (Cl**0.5824 / Nfr**0.0609)
        #         c = 0.0
        #     return max(hl0, Cl), max(c, 0.0)

        # if flow_pattern == "transitional":
        #     hl0_seg, c_seg = get_C_and_Hl0("segregated")
        #     hl_seg = hl0_seg * (1.0 + c_seg * (sin_term - 0.333 * sin_term**3))

        #     hl0_int, c_int = get_C_and_Hl0("intermittent")
        #     hl_int = hl0_int * (1.0 + c_int * (sin_term - 0.333 * sin_term**3))

        #     A = (L3 - Nfr) / (L3 - L2)
        #     Hl = A * hl_seg + (1.0 - A) * hl_int
        # else:
        #     hl0, c = get_C_and_Hl0(flow_pattern)
        #     Hl = hl0 * (1.0 + c * (sin_term - 0.333 * sin_term**3))

        # return max(Cl, min(Hl, 1.0))
        def get_Hl(pattern):
            """Calculates the specific holdup for a given theoretical flow pattern."""
            if pattern == "segregated":
                hl0 = 0.98 * (Cl**0.4846 / max(Nfr, 1e-6)**0.0868)
                c = (1 - Cl) * np.log(0.011 * Nlv**3.539 * Cl**(-3.768) * max(Nfr, 1e-6)**(-1.614))
            elif pattern == "intermittent":
                hl0 = 0.845 * (Cl**0.5351 / max(Nfr, 1e-6)**0.0173)
                c = (1 - Cl) * np.log(2.96 * Nlv**(-0.4473) * Cl**0.305 * max(Nfr, 1e-6)**(0.0978))
            else: # distributed
                hl0 = 1.065 * (Cl**0.5824 / max(Nfr, 1e-6)**0.0609)
                c = 0.0
            
            hl0 = max(hl0, Cl)
            c = max(c, 0.0)
            psi = 1.0 + c * (sin_term - 0.333 * sin_term**3)
            return hl0 * psi

        boundary = L1 if Cl < 0.4 else L4
        boundary_L3 = L3

        # Calculate actual holdup with continuous mathematical smoothing
        if flow_pattern == "transitional":
            hl_seg = get_Hl("segregated")
            hl_int = get_Hl("intermittent")
            A = (L3 - Nfr) / (L3 - L2)
            Hl = A * hl_seg + (1.0 - A) * hl_int
        elif 0.8 * boundary_L3 < Nfr < 1.2 * boundary_L3 and Nfr > L3:
            # Smooth the transitional -> intermittent boundary the same way
            # the intermittent -> distributed boundary is already smoothed.
            hl_trans_blend = get_Hl("segregated") * 0 + get_Hl("intermittent")  # transitional limit as Nfr->L3
            hl_int = get_Hl("intermittent")
            weight = (1.2 * boundary_L3 - Nfr) / (0.4 * boundary_L3)
            weight = max(0.0, min(weight, 1.0))
            Hl = weight * hl_trans_blend + (1.0 - weight) * hl_int
        elif 0.8 * boundary < Nfr < 1.2 * boundary:
            hl_int = get_Hl("intermittent")
            hl_dist = get_Hl("distributed")
            
            weight = (1.2 * boundary - Nfr) / (0.4 * boundary)
            weight = max(0.0, min(weight, 1.0))
            
            Hl = weight * hl_int + (1.0 - weight) * hl_dist
        else:
            Hl = get_Hl(flow_pattern)

        return max(Cl, min(Hl, 1.0))
    
    def frictional_factor(self, Gm, L1, L2, L3, L4, Nfr):
        """Calculates the corrected two-phase friction factor and returns the actual Holdup."""
        Cl = self.fp["Cl"]
        rho_ns = self.fp["rho_l"] * Cl + self.fp["rho_g"] * (1 - Cl)
        mu_ns = self.fp["mu_l"] * Cl + self.fp["mu_g"] * (1 - Cl)
        
        Re = 1488.0 * rho_ns * self.fp["Vm"] * self.tid / max(mu_ns, 1e-6)

        if Re < 2000:
            f = 64.0 / max(Re, 1.0)
        else:
            f = (1.14 - 2.0 * np.log10((self.roughness/self.tid) + 21.25 / (Re ** 0.9))) ** -2
        
        H_L = max(self.calculate_holdup(L1, L2, L3, L4, Nfr), 1e-6) 
        y = max(Cl / (H_L ** 2), 1e-6)
        ln_y = np.log(y)
        
        if 1.0 <= y <= 1.2:
            S = np.log(2.2 * y - 1.2)
        else:
            denominator = -0.0523 + 3.182 * ln_y - 0.8725 * (ln_y ** 2) + 0.01853 * (ln_y ** 4)
            S = ln_y / denominator
            
        f_prime = f * np.exp(S)
        
        # # PDF Modification: Force approach gas at low liquid content
        if Cl < 0.001:
            f_prime = f
            
        return f_prime, H_L
    
    def calculate_gradient(self, P, return_components=False):
        """
        Calculates the total multiphase pressure gradient (psi/ft).
        
        Args:
            P (float): Local node pressure in psia.
            
        Returns:
            float: Pressure gradient (dp/dz) in psi/ft.
        """
        Gm, L1, L2, L3, L4, Nfr = self._dimensionless_numbers()
        f_prime, Hl = self.frictional_factor(Gm, L1, L2, L3, L4, Nfr)
        
        rho_m = self.fp["rho_l"] * Hl + self.fp["rho_g"] * (1.0 - Hl)
        gc = 32.174
        
        # CORRECTED ELEVATION LOGIC: Vertical well = theta_rad is 0 -> cos(0) = 1 (Full gravity)
        hydrostatic = np.cos(self.theta_rad) * rho_m
        friction = (f_prime * Gm * self.fp["Vm"]) / (2.0 * gc * self.tid)
        kinetic_term = 1.0 - (rho_m * self.fp["Vm"] * self.fp["Vsg"]) / (gc * P * 144.0)
        
        dp_dh_el = hydrostatic / kinetic_term / 144.0
        dp_dh_fric = friction / kinetic_term / 144.0
        dp_dz = dp_dh_el + dp_dh_fric
        if return_components:
            return dp_dz, Hl, f_prime, dp_dh_el, dp_dh_fric
        return dp_dz
    
    def calculate_pressure_traverse(self, Pth, surface_temp, bottomhole_temp,
                                    total_depth, step_size, Ql, gl_depth = 0.0, gl_rate = 0.0):
        """
        Calculates the wellbore pressure profile via Euler integration.

        Returns:
            tuple: (depths [ft], pressures [psia], profiles [dict])
        """
        depths        = [0.0]
        pressures     = [Pth]
        
        holdups = []
        frictions = []
        hydro_losses = []
        fric_losses = []
        total_gradients = []
        
        current_P     = Pth
        current_depth = 0.0
        temp_gradient = (bottomhole_temp - surface_temp) / total_depth

        formation_gor = self.fp["gor"]

        while current_depth < total_depth:
            Qo = Ql * (1-self.wc)
            
            if current_depth <= gl_depth and Qo > 0:
                injected_gor = gl_rate / Qo
                effective_gor = formation_gor + injected_gor
            else:
                effective_gor = formation_gor

            next_depth   = min(current_depth + step_size, total_depth)
            actual_step  = next_depth - current_depth
            current_temp = surface_temp + temp_gradient * current_depth

            self.update_fluid_properties(current_P, current_temp, Ql)
            dp_dz, Hl, f, dp_dh_el, dp_dh_fric, H, Hl_psi, B, psi, Nl, Nlv, Ngv, Nd = self.calculate_gradient(current_P,return_components=True)
            # print(f"Vsg: {self.fp["Vsg"]}, Hl: {Hl}, gor: {self.fp["gor"]}, producing GLR: {self.fp["producing_glr"]} at depth: {current_depth}, Ql: {Ql}")
            
            # Print the detailed debug line at the last depth step for the current flow rate
            # if next_depth == total_depth:
            #     # lambda_g, LB = self.is_bubble_flow()
            #     # Hl, H, Hl_psi, B, psi = self.liquid_holdup(*self.dimensionless_numbers())
            #     # Cl = self.fp["Vsl"] / max(self.fp["Vm"], 1e-6)
            #     print(f"Ql={Ql:.2f} Depth={next_depth:.2f} Vsl={self.fp['Vsl']:.3f} Vsg={self.fp['Vsg']:.3f} gor={self.fp['gor']:.1f} glr={self.fp['glr']:.1f} Nl={Nl:.4f} Nlv={Nlv:.4f} Ngv={Ngv:.4f} Nd={Nd:.4f} H={H:.5f} B={B:.5f} Hl_psi={Hl_psi:.4f} psi={psi:.4f} Hl={Hl:.5f} friction_loss:{dp_dh_fric:.5f} hydro_loss:{dp_dh_el:.5f} total_gradient:{dp_dz:.5f}")

            current_P += dp_dz * actual_step
            current_depth = next_depth

            depths.append(current_depth)
            pressures.append(current_P)
            
            holdups.append(Hl)
            frictions.append(f)
            hydro_losses.append(dp_dh_el)
            fric_losses.append(dp_dh_fric)
            total_gradients.append(dp_dz)

        if holdups:
            holdups.insert(0, holdups[0])
            frictions.insert(0, frictions[0])
            hydro_losses.insert(0, hydro_losses[0])
            fric_losses.insert(0, fric_losses[0])
            total_gradients.insert(0, total_gradients[0])
        else:
            holdups = [0.0]; frictions = [0.0]; hydro_losses = [0.0]; fric_losses = [0.0]; total_gradients = [0.0]
            
        profiles = {
            "holdup": holdups,
            "friction_factor": frictions,
            "hydrostatic_loss": hydro_losses,
            "frictional_loss": fric_losses,
            "total_gradient": total_gradients
        }

        return depths, pressures, profiles

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_pressure_traverse(self, Pth, surface_temp, bottomhole_temp,
                               total_depth, step_size, Ql):
        depths, pressures, _ = self.calculate_pressure_traverse(
            Pth, surface_temp, bottomhole_temp, total_depth, step_size, Ql
        )
        plt.plot(pressures, depths, color='blue', linewidth=2)
        plt.gca().invert_yaxis()
        plt.xlabel('Pressure (psia)')
        plt.ylabel('Depth (ft)')
        plt.title('Pressure Traverse Curve')
        plt.grid(True)
        plt.show()
        return pressures[-1]

    def plot_vlp_curve(self, Pth, surface_temp, bottomhole_temp,
                       depth, Qmin, Qmax, step_size):
        Pwf_points = []
        rates = np.linspace(Qmin, Qmax, Qmax - Qmin)

        for q in rates:
            _, pressures, _ = self.calculate_pressure_traverse(
                Pth, surface_temp, bottomhole_temp, depth, step_size, q
            )
            Pwf_points.append(pressures[-1])

        plt.plot(rates, Pwf_points, color='blue')
        plt.xlabel('Liquid Rate (stb/day)')
        plt.ylabel('Pwf (psi)')
        plt.title('VLP Curve')
        plt.grid(True)
        plt.show()

    def vlp_curve_plot_linear(self, Pth, depth, Qmin, Qmax, step_size):
        """Simplified single-gradient VLP (stale PVT — approximate only)."""
        Pwf_points = []
        rates = np.linspace(Qmin, Qmax, int((Qmax - Qmin) / step_size))

        for q in rates:
            self.Ql = q
            Pwf = Pth + self.calculate_gradient() * depth
            Pwf_points.append(Pwf)

        plt.plot(rates, Pwf_points, color='blue')
        plt.xlabel('Liquid Rate (stb/day)')
        plt.ylabel('Pwf (psi)')

import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

class ChartFunction:
    """
    Digitized chart interpolation for Duns and Ros empirical curves.

    Supports:
        - linear interpolation
        - log-x interpolation
        - log-log interpolation
    """

    def __init__(
        self,
        points: list[tuple[float, float]],
        name: str,
        logx: bool = True,
        logy: bool = False,
    ):
        pts = sorted(points)

        self.name = name
        self.logx = logx
        self.logy = logy

        self.x = np.array([p[0] for p in pts], dtype=float)
        self.y = np.array([p[1] for p in pts], dtype=float)

        x_interp = np.log10(self.x) if logx else self.x
        y_interp = np.log10(self.y) if logy else self.y

        self._interp = PchipInterpolator(
            x_interp,
            y_interp,
            extrapolate=True,
        )

    def __call__(self, x):

        x = np.clip(x, self.x.min(), self.x.max())

        if self.logx:
            x = np.log10(x)

        y = self._interp(x)

        if self.logy:
            y = 10 ** y

        return float(y)

# Note: CHART_DATA dictionary is assumed to be defined here exactly as provided in the prompt.
# ... [Insert CHART_DATA dictionary here] ...
CHART_DATA = {
    # L1, L2 vs Nd (flow regime boundary chart, Duns & Ros Fig.)
    "L1": ChartFunction(
    [
        (18.633328021022624, 1.9673441797585194),
        (21.081474510081954, 1.9668080742156544),
        (23.470311678288823, 1.966341895482728),
        (26.129838783932364, 1.9658757167498018),
        (28.928803671488236, 1.9454337793109877),
        (31.01098401000279, 1.920131928581418),
        (33.05444524262925, 1.8648547853246935),
        (34.85264905282858, 1.7996247261199945),
        (36.16365701336464, 1.7394643606358677),
        (38.33415078713324, 1.6692112255838887),
        (41.06022129878563, 1.5689128711948161),
        (45.38779351524844, 1.4034776933476294),
        (50.44959401045713, 1.253018507295697),
        (54.31928958587173, 1.1376975432380774),
        (56.0728940865842, 1.0975595543331311),
        (58.194609196629045, 1.0573982564915387),
        (62.04264687193397, 1.0221201808773483),
        (66.15223119848132, 0.9968416390844246),
        (70.17907440825354, 1.0015850076919488),
        (75.64325706044883, 0.986259381847),
        (80.25214860704445, 0.9960025173651574),
        (85.58626689674351, 0.9907230432147682),
        (100.52735960982052, 0.9800242412941118),
        (130.7950478568692, 0.9988811710409768),
        (170.12121383613172, 0.9877394993240407),
        (214.25959640296347, 0.9767376812269819),
        (259.9276344803129, 0.9758985595077148),
    ],
    "L1 vs Nd",
    logx=True,
    logy=False,
    ),
    "L2": ChartFunction(
    [
        (14.712432733805487, 0.45837023914968933),
        (16.036834321876487, 0.4879958976271501),
        (18.053502301358748, 0.5224814693953661),
        (20.656983216109897, 0.5718964150855439),
        (22.517717060630957, 0.6065218404736372),
        (24.676831962353383, 0.636124190014451),
        (28.699924316922647, 0.705468276537224),
        (31.972627801598087, 0.7649993007319),
        (36.39341385478015, 0.8294368560906249),
        (41.42767582563093, 0.898874178359983),
        (46.65732801549707, 0.973357885413267),
        (50.311691631081686, 0.9980303948533864),
        (53.38574342176924, 1.022772831103445),
        (56.641539909976416, 1.037515733532236),
        (60.74444891702114, 1.0522120180877346),
        (65.14106159771141, 1.0619085357325995),
        (68.74751498232226, 1.0816745140086705),
        (72.9284250387007, 1.0814181157055611),
        (77.7799319099715, 1.0811384084658053),
        (81.63359275458912, 1.0859283949466223),
        (90.89849875583491, 1.1004615169455967),
        (100.10739925184689, 1.0900424222646958),
        (126.76582612488139, 1.0840170621416247),
        (150.53497201988117, 1.09327070999021),
        (198.9637464087474, 1.0770593445527012),
        (240.10535497735594, 1.0862430655913473),
        (292.84978644218404, 1.0853806349354338),
    ],
    "L2 vs Nd",
    logx=True,
    logy=False,
    ),
 
    # Bubble flow slip-velocity factors F1-F4 (functions of NLv mostly)
    "F1": ChartFunction(
    [
        (0.0047097734208786275, 1.1884377832092687),
        (0.005876826558681099, 1.1948064745418547),
        (0.00716416256290527, 1.201310241689731),
        (0.008632326130161976, 1.2005029048843316),
        (0.010280875429245573, 1.2071391163674168),
        (0.011962227378650511, 1.1990913708102204),
        (0.013597958411538933, 1.22082952939805),
        (0.0160072163705201, 1.2276296839020686),
        (0.019064177487373227, 1.2496751612401715),
        (0.021925013954577844, 1.2722768877953283),
        (0.025215157445256677, 1.3032686916358431),
        (0.029682726540805376, 1.3349032527324867),
        (0.0341370192366346, 1.375846363347798),
        (0.03835545170522658, 1.4094794888767415),
        (0.041614832900421804, 1.4619683560857015),
        (0.0462156972405333, 1.5162844782051967),
        (0.0495621813410424, 1.5825082169568252),
        (0.05253529373289193, 1.6215339003682143),
        (0.0556867557666538, 1.7133465473816603),
        (0.06184338769249529, 1.798968584678263),
        (0.0711238205295976, 1.8886313813676314),
        (0.08470661648913988, 1.9825131677969534),
        (0.09971475843062777, 2.018199457409207),
        (0.13035956500855575, 2.0537513391547657),
        (0.1645681974742974, 2.0773924698341797),
        (0.205347193422057, 2.063022767715851),
        (0.2562309759326464, 2.0237359185694817),
        (0.3234704706443913, 1.9608743347023152),
        (0.39432762101387847, 1.8427282999136727),
        (0.4696339189259032, 1.7425175828221893),
        (0.546438653165416, 1.6580489671646061),
        (0.5860063406375826, 1.5684744399947577),
        (0.6358041649890819, 1.5020173187960186),
        (0.706097579767229, 1.4119927469751794),
        (0.7841625135653214, 1.3273638676208042),
        (0.8708581721577248, 1.255496023477772),
        (0.9899404878797302, 1.172920419752072),
        (1.1518370482015015, 1.0756767552025779),
        (1.3559171381563269, 0.9986479657940792),
        (1.6148619711038166, 0.9385565840701039),
        (1.8356803652333444, 0.8931350634623554),
    ],
    "F1 vs NL",
    logx=True,
    logy=True,
    ),
    "F2": ChartFunction(
    [
        (0.0023961350191944964, 0.24569931139646597),
        (0.002887178892992733, 0.2425360644020282),
        (0.0039087474120943376, 0.23784740124687861),
        (0.0046552162907420045, 0.24211860742808483),
        (0.005674955001538332, 0.238991418153314),
        (0.006680431712859501, 0.2373881374894177),
        (0.007772960939634417, 0.2372585063543518),
        (0.009475650712270009, 0.238549990885161),
        (0.011285253026238723, 0.24433010716695236),
        (0.012978757160916374, 0.24722572610193402),
        (0.015101325302466297, 0.2501451511793063),
        (0.017367482452809117, 0.2578174080066484),
        (0.01974233532192502, 0.26902106241493245),
        (0.022441928760969402, 0.28593268496916946),
        (0.026418142972389008, 0.32312073084127485),
        (0.030382540957106366, 0.36969024153915314),
        (0.0341370192366346, 0.4152830195386351),
        (0.03971984574157566, 0.4780429467551138),
        (0.046757325309184146, 0.5435453181912899),
        (0.05377389140242689, 0.614289758694431),
        (0.06112700551252204, 0.6774195726140168),
        (0.0711238205295976, 0.756208160125373),
        (0.08874786960881664, 0.8543801321026386),
        (0.11334988838084534, 0.9475910561080507),
        (0.12298216953621659, 0.9889355513084734),
        (0.1464685585270488, 1.0317370920262887),
        (0.187071473755072, 1.0564699259522583),
        (0.22021640797257086, 1.0493825676807544),
        (0.2562309759326464, 1.036002910485813),
        (0.2878943458607774, 1.0042443003367811),
        (0.31972344950658704, 0.9916068008858617),
        (0.45350178884617676, 0.9428648986372893),
        (0.6432555159906893, 0.8801485257431317),
        (0.8708581721577248, 0.8318999606670943),
        (1.086651517521671, 0.7913740415023925),
        (1.9232585180412767, 0.6899148958956611),
    ],
    "F2 vs NL",
    logx=True,
    logy=True,
    ),
    "F3": ChartFunction(
    [
        (0.002157595024730548, 0.8500974373239191),
        (0.0025696394391900197, 0.8495618273218977),
        (0.0032063804249099306, 0.8436852534115794),
        (0.0037744802325752045, 0.8483846728946501),
        (0.004443234781319241, 0.853110268660153),
        (0.005291776816682573, 0.884582614507784),
        (0.0059457004350657395, 0.9230616181163386),
        (0.006837932824266713, 0.9690681392428466),
        (0.008143799770748056, 1.055431716261751),
        (0.009365886555305094, 1.1638482519535336),
        (0.010771363913185596, 1.291309662988984),
        (0.012532931188056616, 1.4238963309801262),
        (0.014582588187581536, 1.5895053349355652),
        (0.017776946677014453, 1.7850091316332415),
        (0.020444616832878213, 2.0049804086980276),
        (0.026418142972389008, 2.1964739098523873),
        (0.033741582421624915, 2.4211853467951143),
        (0.04113277449733655, 2.588593722237542),
        (0.05073068346493268, 2.7336685330211092),
        (0.06555321920772467, 2.886388746745508),
        (0.08084938724349282, 3.0109332930006794),
        (0.10818834938218304, 3.1592757583036177),
        (0.12884950430771666, 3.2358256721454928),
        (0.18490447296194223, 3.3943922785435547),
        (0.2716016369711303, 3.4954168586989294),
        (0.39432762101387847, 3.599599393935142),
        (0.5860063406375826, 3.6838763407936925),
        (0.9448622217470078, 3.7690177486136833),
        (1.3878849009319898, 3.8103217953732416),
        (1.8356803652333444, 3.853535438648476),
    ],
    "F3 vs NL",
    logx=True,
    logy=True,
    ),
    "F4": ChartFunction(
    [
        (0.0022268088724126454, -17.112700900473712),
        (0.0025422502164772543, -12.736612900427431),
        (0.0029675749575328756, -9.463475007084682),
        (0.0035543707501802446, -3.7209167828164027),
        (0.004253738397308992, 1.1972482635374604),
        (0.005336438759844322, 6.657490356198906),
        (0.006542311977171822, 12.396289377649495),
        (0.008305944165907105, 18.129449594873606),
        (0.010671458248729926, 24.13552793666031),
        (0.014711261806232138, 30.679924121936963),
        (0.020045492618077247, 37.22619990862245),
        (0.024822747189981904, 41.314332972835416),
        (0.0310734391599242, 44.57619325772515),
        (0.037107021896188815, 47.29597649630729),
        (0.044794969596732785, 49.18948695556621),
        (0.05610531994860591, 53.00094269239888),
        (0.07521618470276938, 54.87753673897829),
        (0.09402914412307273, 56.76540839401072),
        (0.11207419966712848, 57.56160755079261),
        (0.15000541885596605, 57.78941524154323),
        (0.22822386317236415, 57.996547316796686),
        (0.343113773409134, 57.93076126748751),
        (0.5865238997951222, 58.119097328652636),
        (1.1524181910739129, 57.73528272096882),
        (1.9245933247620426, 57.92737798495161),
        ],
    "F4 vs NL",
    logx=True,
    logy=False,
    ),
 
    # Slug flow slip-velocity factors F5-F7
    "F5": ChartFunction(
    [
        (0.0019632644688572223, 0.21785354835439572),
        (0.0025093579826765845, 0.2150478893587228),
        (0.003950786491029975, 0.21089307736271523),
        (0.0049294083612042225, 0.20553867555252173),
        (0.0060779091213905324, 0.19777954314172455),
        (0.009232837382508524, 0.18786765860887447),
        (0.012556267289580522, 0.18076196820262094),
        (0.015672587454780014, 0.17393652128682818),
        (0.020540895633127745, 0.16843481563900084),
        (0.0272904055169515, 0.15597412762428822),
        (0.033681519436155515, 0.14536893318135607),
        (0.04106318352028642, 0.1354860989913991),
        (0.05395469248542661, 0.12074718149402199),
        (0.06269450292781399, 0.10900533355346764),
        (0.07022107438159088, 0.09840808612911657),
        (0.07770861837724322, 0.08827630867853933),
        (0.08289124634704675, 0.07918990737863194),
        (0.08966624924110221, 0.06706989223521147),
        (0.09688188935589001, 0.059023901785505185),
        (0.10328296883450494, 0.053972787974449155),
        (0.1184365991719433, 0.050305847855399925),
        (0.137407525928562, 0.04779464753851634),
        (0.1652890099168218, 0.046285933675209584),
        (0.20100505879067126, 0.04687388883294972),
        (0.23824388870938212, 0.049324611144728525),
        (0.2855292195226913, 0.05393054270100632),
        (0.32997878903301314, 0.05821979761372023),
        (0.441057063746522, 0.06613745375454232),
        (0.603915417703538, 0.07609630462318888),
        (0.7505821447164076, 0.08426940107310579),
        (0.9685463331413948, 0.09096387106047263),
        (1.1470896407164806, 0.09819664286992245),
        (1.4094074068464237, 0.10600151091281479),
        (1.8882469082245943, 0.11153309882338129),
    ],
    "F5 vs NL",
    logx=True,
    logy=True,
    ),
    "F6": ChartFunction(
    [
        (0.0023045680598371802, 0.8960501146673954),
        (0.0024992695145586094, 0.7947692297049089),
        (0.0029103762958418472, 0.6428063937017678),
        (0.003390856974545664, 0.5020840755870868),
        (0.004250885393400691, 0.3556418779855899),
        (0.005393335942398626, 0.20356281801609866),
        (0.00763254901445598, 0.02885329145580906),
        (0.010204379904056348, -0.05022881593375228),
        (0.011938262991657695, -0.10102699094075662),
        (0.015059215916485206, -0.1125829738807389),
        (0.019015691693788365, -0.10165792104380156),
        (0.022089663809180482, -0.04003431374244615),
        (0.024490303307265735, 0.07223803775203486),
        (0.027846858709661252, 0.20133795923167686),
        (0.0316634518677349, 0.3304378807113184),
        (0.034695032746403974, 0.4539673535180233),
        (0.03707758951008682, 0.5662895152837977),
        (0.04161374855515424, 0.6785452633545148),
        (0.04559800917640705, 0.8020747361612193),
        (0.05190108991824019, 0.95365569341778),
        (0.05760100814781605, 1.0884090806891806),
        (0.06313224959236584, 1.2175588124401155),
        (0.06836979743348712, 1.3523454065590446),
        (0.0748964440897356, 1.4702546204215197),
        (0.08743167231137836, 1.6386631476632414),
        (0.09492993554579546, 1.8296523312244681),
        (0.10695733942836569, 2.026211963458632),
        (0.11991883175623952, 2.1159866757524295),
        (0.14264893520263436, 2.160716299373567),
        (0.16489781668162143, 2.1155549867345567),
        (0.18600332101946462, 2.0704268809430744),
        (0.2283599641551177, 2.0027015154083236),
        (0.28028967283159484, 1.9293558909293425),
        (0.34027858972778907, 1.8841281645952743),
        (0.43874789748909243, 1.8163363853654657),
        (0.5734247682008801, 1.7766292974330418),
        (0.6966912995641216, 1.7482623479316632),
        (0.8367976443169406, 1.7367727786867384),
        (1.0953544369153667, 1.7307872444196932),
        (1.5637937290078308, 1.7471665219632166),
        (2.049624842317953, 1.7692822824173204),
    ],
    "F6 vs NL",
    logx=True,
    logy=False,
    ),
    "F7": ChartFunction(
    [
        (0.002367610168455471, 0.12984734695554803),
        (0.003263980092212702, 0.1194724399811137),
        (0.00496693631950634, 0.1071456606620092),
        (0.007196920449404337, 0.09609434464187412),
        (0.011650508671800987, 0.08454013377834456),
        (0.017516595023889983, 0.075335578234457),
        (0.025091399176240904, 0.065861625214953),
        (0.03379305317665792, 0.05832204681515861),
        (0.04663231949035999, 0.05197549478818534),
        (0.07016647268285738, 0.04514823797547856),
        (0.10294208410343082, 0.04023334998155953),
        (0.15297849431459018, 0.03517274155594147),
        (0.2162532015936405, 0.03174763750720955),
        (0.31322162876521514, 0.028839130759407344),
        (0.48789637003908565, 0.026873445931480038),
        (0.6319088363903568, 0.025693367595397698),
        (0.9952777133846487, 0.02487715196042026),
        (1.5485077563770697, 0.024087092742026098),
        (2.2380694742980918, 0.023472883800341893),
        (2.9304205855120666, 0.023468012563309177),
    ],
    "F7 vs NL",
    logx=True,
    logy=True,
    ),
    "Bubble_f2": ChartFunction(
    [
        (0.0010000000, 1.0000000000),
        (0.0020000000, 1.0004000000),
        (0.0050000000, 1.0016000000),
        (0.0100000000, 1.0035000000),
        (0.0200000000, 1.0075000000),
        (0.0300000000, 1.0110000000),
        (0.0400000000, 1.0150000000),

        (0.3881326932, 1.0195800412),
        (0.4473847829, 0.9855621266),
        (0.4942342516, 0.9573087268),
        (0.5460593839, 0.9253684718),
        (0.6122091826, 0.8773160081),
        (0.7578614592, 0.8237335342),
        (0.9123960213, 0.7622570417),
        (0.9935708143, 0.7261057940),
        (1.3170622459, 0.6459728153),
        (1.8789719673, 0.5718157012),
        (2.7213021792, 0.4988478935),
        (3.6049268160, 0.4546840763),
        (4.8492404677, 0.4044926867),
        (6.9153640160, 0.3633029011),
        (9.7195047370, 0.3247394239),
        (11.7862263013, 0.2999844071),
        (17.8813142696, 0.2637647260),
        (31.9159008027, 0.2305598160),
        (39.8748508053, 0.2204067694),
        (52.0883997580, 0.2086091551),
        (62.2170188641, 0.2043735294),
        (75.4103257933, 0.2011782976),
        (99.8800137257, 0.1969649502),
    ],
    "Bubble Flow f2",
    logx=True,
    logy=True,
    )
}

class DunsRos:
    """
    Calculates the wellbore pressure traverse using the Duns and Ros method, 
    specifically designed for vertical flow in high Gas-Oil Ratio (GOR) wells.
    """
    
    def __init__(self, tubing_id: float, tubing_od: float, casing_id: float, roughness: float, pvt_model,
                 fluid_properties: dict, watercut: float = 0.0, theta: float = 0.0):
        """
        Initializes the Duns and Ros pressure drop model.

        Args:
            tubing_id (float): Tubing Inner Diameter in feet.
            tubing_od (float): Tubing Outer Diameter in feet (for annulus calculations if needed).
            casing_id (float): Casing Inner Diameter in feet.
            roughness (float): Absolute pipe roughness in feet.
            pvt_model (object): Object containing PVT calculation methods.
            fluid_properties (dict): Dictionary of initial fluid properties (rho, mu, sigma, etc.).
            watercut (float): Watercut as a fraction (0.0 to 1.0).
            theta (float): Deviation angle from vertical in degrees (0 = vertical).
        """
        self.tid      = tubing_id
        self.tod      = tubing_od
        self.cid      = casing_id
        self.roughness = roughness
        self.pvt_model = pvt_model
        self.wc       = watercut
        self.wor      = watercut / (1.0 - watercut + 1e-9)
        self.Ap       = (np.pi / 4.0) * self.tid**2
        self.fp       = fluid_properties
        self.theta_ui = theta      
        self.theta_rad = np.radians(theta)  
        self.Ql       = 0.0

    def update_fluid_properties(self, P: float, T: float, Ql: float):
        """Updates in-situ fluid properties based on current pressure and temperature."""
        self.Ql = Ql
        self.fp = self.pvt_model.fluid_properties_dict(
            P, T, self.fp["Rsb"], self.fp["producing_gor"], self.fp["Pb"]
        )

    def _superficial_velocities(self):
        """Calculates in-situ superficial velocities (Vsl, Vsg) and no-slip liquid holdup (Cl)."""
        fo = (1.0 - self.wc)
        fw = self.wc

        # Liquid flow rate in situ (ft3/D) converted to ft/sec
        q_liquid_insitu = 5.615 * self.Ql * (self.fp["Bo"] * fo + self.fp["Bw"] * fw)
        self.fp["Vsl"] = q_liquid_insitu / (86400.0 * self.Ap)

        # Gas flow rate in situ (ft3/D) converted to ft/sec
        free_gas_scf = max(0.0, self.Ql * (self.fp["glr"] - self.fp["gor"] * fo))
        q_gas_insitu = (free_gas_scf * (14.7 / self.fp["Pr"]) 
                        * ((self.fp["Tr"] + 460.0) / 520.0) * self.fp["Z"])
        
        self.fp["Vsg"] = q_gas_insitu / (86400.0 * self.Ap)
        self.fp["Vm"]  = self.fp["Vsl"] + self.fp["Vsg"]
        
        # No-slip liquid holdup
        self.fp["Cl"] = self.fp["Vsl"] / max(self.fp["Vm"], 1e-6)

    def _dimensionless_numbers(self):
        """Calculates Duns and Ros dimensionless numbers for flow regime mapping."""
        Nl = 0.15726 * self.fp["mu_l"] * (1.0 / (self.fp["rho_l"] * self.fp["sigma_l"]**3))**0.25
        base_term = self.fp["rho_l"] / self.fp["sigma_l"]
        
        Nlv = 1.938 * self.fp["Vsl"] * base_term**0.25
        Ngv = 1.938 * self.fp["Vsg"] * base_term**0.25
        Nd = 120.872 * self.tid * base_term**0.5
        
        # Fetch empirical chart values
        L1 = CHART_DATA["L1"](Nd)
        L2 = CHART_DATA["L2"](Nd)
        F1 = CHART_DATA["F1"](Nl)
        F2 = CHART_DATA["F2"](Nl)
        F3 = CHART_DATA["F3"](Nl)
        F4 = CHART_DATA["F4"](Nl)
        F5 = CHART_DATA["F5"](Nl)
        F6 = CHART_DATA["F6"](Nl)
        F7 = CHART_DATA["F7"](Nl)
        
        return Nl, Nlv, Ngv, Nd, L1, L2, F1, F2, F3, F4, F5, F6, F7 
    
    def flow_regime(self, L1: float, L2: float, Nlv: float, Ngv: float) -> int:
        """Identifies the Duns and Ros flow regime based on dimensionless boundaries."""
        if Ngv < (L1 + L2 * Nlv):
            return 1 # Bubble flow
        elif Ngv < (50.0 + 36.0 * Nlv):
            return 2 # Slug flow
        elif Ngv < (75.0 + 84.0 * (Nlv**0.75)):
            return 3 # Transition flow
        else:
            return 4 # Mist flow
    
    def _bubble_flow_holdup(self, F1, F2, F3, F4, Nlv, Ngv, Nd) -> float:
        """Calculates liquid holdup (Hl) for the bubble flow regime."""
        F3_ = F3 - F4 / Nd
        S = F1 + F2 * Nlv + F3_ * (Ngv / (1.0 + Nlv))**2
        Vs = S / (1.938 * (self.fp["rho_l"] / self.fp["sigma_l"])**0.25)
        Hl = (Vs - self.fp["Vm"] + ((self.fp["Vm"] - Vs)**2 + 4.0 * Vs * self.fp["Vsl"])**0.5) / (2.0 * Vs)
        return Hl

    def _slug_flow_holdup(self, F5, F6, F7, Nlv, Ngv, Nd) -> float:
        """Calculates liquid holdup (Hl) for the slug flow regime."""
        F6_ = 0.029 * Nd + F6
        S = (1.0 + F5) * ((Ngv**0.982 + F6_) / (1.0 + F7 * Nlv)**2)
        Vs = S / (1.938 * (self.fp["rho_l"] / self.fp["sigma_l"])**0.25)
        Hl = (Vs - self.fp["Vm"] + ((self.fp["Vm"] - Vs)**2 + 4.0 * Vs * self.fp["Vsl"])**0.5) / (2.0 * Vs)
        return Hl
    
    def _mix_properties(self, Hl: float):
        """Updates mixture density and viscosity based on the calculated liquid holdup."""
        self.fp["rho_m"] = self.fp["rho_l"] * Hl + self.fp["rho_g"] * (1.0 - Hl)
        self.fp["mu_m"]  = (self.fp["mu_l"] ** Hl) * (self.fp["mu_g"] ** (1.0 - Hl))

    def moody_friction_factor(self, Re: float, rel_roughness: float, tol: float=1e-8, max_iter: int=50) -> float:
        """Calculates the Darcy friction factor using the Colebrook-White equation."""
        if Re <= 0:
            raise ValueError("Reynolds number must be positive.")
        if Re < 2100:
            return 64.0 / Re

        # Initial guess (Haaland)
        f = (-1.8 * np.log10((rel_roughness / 3.7) ** 1.11 + 6.9 / Re)) ** (-2)

        for _ in range(max_iter):
            rhs = -2.0 * np.log10(rel_roughness / 3.7 + 2.51 / (Re * np.sqrt(f)))
            f_new = 1.0 / rhs**2
            if abs(f_new - f) < tol:
                return f_new
            f = f_new

        return f
    
    def _bubble_slug_flow_friction(self, Nd: float) -> float:
        """Calculates the combined friction factor for Bubble and Slug flow regimes."""
        Re = (1488.0 * self.fp["rho_l"] * self.fp["Vsl"] * self.tid) / self.fp["mu_l"]
        relative_roughness = self.roughness / self.tid
        
        f1 = self.moody_friction_factor(Re, relative_roughness)
        X = f1 * self.fp["Vsg"] * Nd**(2.0/3.0) / self.fp["Vsl"]
        f2 = CHART_DATA["Bubble_f2"](X)
        f3 = 1.0 + (f1 / 4.0) * (self.fp["Vsg"] / (50.0 * self.fp["Vsl"]))**0.5
        
        return f1 * f2 / f3

    def _mist_flow_friction(self) -> float:
        """Calculates the friction factor specific to the Mist flow regime."""
        # Gas Reynolds number
        Re_g = (1488.0 * self.fp["rho_g"] * self.fp["Vsg"] * self.tid) / self.fp["mu_g"]
        
        # Nwe * Nh simplifies to remove epsilon
        Nwe_Nh = (self.fp["rho_g"] * self.fp["Vsg"]**2 * self.fp["mu_l"]**2) / (self.fp["rho_l"] * self.fp["sigma_l"]**2)
        
        if Nwe_Nh <= 0.005:
            epsilon_d = 0.0749 * self.fp["sigma_l"] / (self.fp["rho_g"] * self.fp["Vsg"]**2 * self.tid)
        else:
            epsilon_d = 0.3713 * self.fp["sigma_l"] * (Nwe_Nh**0.302) / (self.fp["rho_g"] * self.fp["Vsg"]**2 * self.tid)
        
        # Calculate f based on the film relative roughness boundary
        if epsilon_d > 0.05:
            f = 4.0 * (1.0 / (4.0 * np.log10(0.27 * epsilon_d))**2 + 0.067 * epsilon_d**1.73)
        else:
            f = self.moody_friction_factor(Re_g, epsilon_d)
            
        return f
    
    def calculate_gradient(self, P: float, return_components: bool = False):
        """
        Calculates the total pressure gradient (dp/dz) at the current state.
        
        Args:
            P (float): Current absolute pressure in psia.
            return_components (bool): If True, returns a tuple of gradient components.

        Returns:
            float or tuple: The total pressure gradient in psi/ft, or a tuple containing 
                            (dp_dz, Hl, f, dp_dz_elev, dp_dz_fric)
        """
        self._superficial_velocities()
        Nl, Nlv, Ngv, Nd, L1, L2, F1, F2, F3, F4, F5, F6, F7 = self._dimensionless_numbers()
        flow_regime_var = self.flow_regime(L1, L2, Nlv, Ngv)
        
        if flow_regime_var == 1:
            # Bubble flow
            Hl = self._bubble_flow_holdup(F1, F2, F3, F4, Nlv, Ngv, Nd)
            self._mix_properties(Hl)
            f = self._bubble_slug_flow_friction(Nd)
            
            dp_dz_elev = self.fp["rho_m"] / 144.0
            dp_dz_fric = (f * self.fp["rho_l"] * self.fp["Vsl"] * self.fp["Vm"] / (2.0 * self.tid * 32.17)) / 144.0
            dp_dz = dp_dz_elev + dp_dz_fric
            
        elif flow_regime_var == 2:
            # Slug flow
            Hl = self._slug_flow_holdup(F5, F6, F7, Nlv, Ngv, Nd)
            self._mix_properties(Hl)
            f = self._bubble_slug_flow_friction(Nd)
            
            dp_dz_elev = self.fp["rho_m"] / 144.0
            dp_dz_fric = (f * self.fp["rho_l"] * self.fp["Vsl"] * self.fp["Vm"] / (2.0 * self.tid * 32.17)) / 144.0
            dp_dz = dp_dz_elev + dp_dz_fric
            
        elif flow_regime_var == 4:
            # Mist flow
            Ngv_trm = 75.0 + 84.0 * (Nlv**0.75)
            self.fp["rho_g"] = self.fp["rho_g"] * Ngv / Ngv_trm
            Hl = self.fp["Cl"] # No slip condition for Mist
            self._mix_properties(Hl)
            f = self._mist_flow_friction()
            
            # Kinetic energy dimensionless term (using no-slip mixture density)
            rho_n = self.fp["rho_l"] * self.fp["Cl"] + self.fp["rho_g"] * (1.0 - self.fp["Cl"])
            Ek = (self.fp["Vm"] * self.fp["Vsg"] * rho_n) / (144.0 * 32.17 * P)
            
            dp_dz_elev = self.fp["rho_m"] / (144.0 * (1.0 - Ek))
            dp_dz_fric = (f * self.fp["rho_g"] * self.fp["Vsg"]**2 / (2.0 * self.tid * 32.17)) / (144.0 * (1.0 - Ek))
            dp_dz = dp_dz_elev + dp_dz_fric
            
        else:
            # Transition Flow (weighted average of Slug and Mist)
            Ngv_str = 50.0 + 36.0 * Nlv
            Ngv_trm = 75.0 + 84.0 * (Nlv**0.75)
            A = (Ngv_trm - Ngv) / (Ngv_trm - Ngv_str)

            # Slug component
            Hl_slug = self._slug_flow_holdup(F5, F6, F7, Nlv, Ngv, Nd)
            self._mix_properties(Hl_slug)
            f_slug = self._bubble_slug_flow_friction(Nd)
            dp_dz_slug = (f_slug * self.fp["rho_l"] * self.fp["Vsl"] * self.fp["Vm"] / (2.0 * self.tid * 32.17) + self.fp["rho_m"]) / 144.0

            # Mist component
            self.fp["rho_g"] = self.fp["rho_g"] * Ngv / Ngv_trm
            Hl_mist = self.fp["Cl"]
            self._mix_properties(Hl_mist)
            f_mist = self._mist_flow_friction()
            
            rho_n = self.fp["rho_l"] * self.fp["Cl"] + self.fp["rho_g"] * (1.0 - self.fp["Cl"])
            Ek = (self.fp["Vm"] * self.fp["Vsg"] * rho_n) / (144.0 * 32.17 * P)
            dp_dz_mist = (f_mist * self.fp["rho_g"] * self.fp["Vsg"]**2 / (2.0 * self.tid * 32.17) + self.fp["rho_m"]) / (144.0 * (1.0 - Ek))

            dp_dz = A * dp_dz_slug + (1.0 - A) * dp_dz_mist
            
            # For reporting components in transition, we approximate using the weighted values
            Hl = A * Hl_slug + (1.0 - A) * Hl_mist
            f = A * f_slug + (1.0 - A) * f_mist
            dp_dz_elev = A * (self.fp["rho_m"] / 144.0) + (1.0 - A) * (self.fp["rho_m"] / (144.0 * (1.0 - Ek)))
            dp_dz_fric = dp_dz - dp_dz_elev

        if return_components:
            return dp_dz, Hl, f, dp_dz_elev, dp_dz_fric
        return dp_dz
    
    def calculate_pressure_traverse(self, Pth: float, surface_temp: float, bottomhole_temp: float,
                                    total_depth: float, step_size: float, Ql: float, 
                                    gl_depth: float = 0.0, gl_rate: float = 0.0):
        """
        Calculates the wellbore pressure profile via Euler integration.

        Returns:
            tuple: (depths [ft], pressures [psia], profiles [dict])
        """
        depths        = [0.0]
        pressures     = [Pth]
        
        holdups = []
        frictions = []
        hydro_losses = []
        fric_losses = []
        total_gradients = []
        
        current_P     = Pth
        current_depth = 0.0
        temp_gradient = (bottomhole_temp - surface_temp) / total_depth
        formation_gor = self.fp["gor"]

        while current_depth < total_depth:
            Qo = Ql * (1.0 - self.wc)
            
            if current_depth <= gl_depth and Qo > 0:
                injected_gor = gl_rate / Qo
                effective_gor = formation_gor + injected_gor
            else:
                effective_gor = formation_gor

            next_depth   = min(current_depth + step_size, total_depth)
            actual_step  = next_depth - current_depth
            current_temp = surface_temp + temp_gradient * current_depth

            # Update PVT properties at current step
            self.update_fluid_properties(current_P, current_temp, Ql)
            
            # Calculate gradient components
            dp_dz, Hl, f, dp_dz_elev, dp_dz_fric = self.calculate_gradient(current_P, return_components=True)

            current_P += dp_dz * actual_step
            current_depth = next_depth

            depths.append(current_depth)
            pressures.append(current_P)
            
            holdups.append(Hl)
            frictions.append(f)
            hydro_losses.append(dp_dz_elev)
            fric_losses.append(dp_dz_fric)
            total_gradients.append(dp_dz)

        # Pad profile lists to match the dimension of depth/pressure arrays
        if holdups:
            holdups.insert(0, holdups[0])
            frictions.insert(0, frictions[0])
            hydro_losses.insert(0, hydro_losses[0])
            fric_losses.insert(0, fric_losses[0])
            total_gradients.insert(0, total_gradients[0])
        else:
            holdups = [0.0]; frictions = [0.0]; hydro_losses = [0.0]; fric_losses = [0.0]; total_gradients = [0.0]
            
        profiles = {
            "holdup": holdups,
            "friction_factor": frictions,
            "hydrostatic_loss": hydro_losses,
            "frictional_loss": fric_losses,
            "total_gradient": total_gradients
        }

        return depths, pressures, profiles

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_pressure_traverse(self, Pth: float, surface_temp: float, bottomhole_temp: float,
                               total_depth: float, step_size: float, Ql: float):
        """Generates a plot of the pressure traverse from the surface to total depth."""
        depths, pressures, _ = self.calculate_pressure_traverse(
            Pth, surface_temp, bottomhole_temp, total_depth, step_size, Ql
        )
        plt.plot(pressures, depths, color='blue', linewidth=2)
        plt.gca().invert_yaxis()
        plt.xlabel('Pressure (psia)')
        plt.ylabel('Depth (ft)')
        plt.title('Pressure Traverse Curve')
        plt.grid(True)
        plt.show()
        
        return pressures[-1]

    def plot_vlp_curve(self, Pth: float, surface_temp: float, bottomhole_temp: float,
                       depth: float, Qmin: float, Qmax: float, step_size: float):
        """Generates a Vertical Lift Performance (VLP) curve over a range of flow rates."""
        Pwf_points = []
        rates = np.linspace(Qmin, Qmax, int(Qmax - Qmin))

        for q in rates:
            _, pressures, _ = self.calculate_pressure_traverse(
                Pth, surface_temp, bottomhole_temp, depth, step_size, q
            )
            Pwf_points.append(pressures[-1])

        plt.plot(rates, Pwf_points, color='blue')
        plt.xlabel('Liquid Rate (stb/day)')
        plt.ylabel('Pwf (psi)')
        plt.title('VLP Curve')
        plt.grid(True)
        plt.show()

    def vlp_curve_plot_linear(self, Pth: float, depth: float, Qmin: float, Qmax: float, step_size: float):
        """Simplified single-gradient VLP (stale PVT — approximate only)."""
        Pwf_points = []
        rates = np.linspace(Qmin, Qmax, int((Qmax - Qmin) / step_size))

        for q in rates:
            self.Ql = q
            self._superficial_velocities()
            dp_dz = self.calculate_gradient(Pth)
            Pwf = Pth + dp_dz * depth
            Pwf_points.append(Pwf)

        plt.plot(rates, Pwf_points, color='blue')
        plt.xlabel('Liquid Rate (stb/day)')
        plt.ylabel('Pwf (psi)')
        plt.title('Linear VLP Curve')
        plt.grid(True)
        plt.show()