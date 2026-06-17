import numpy as np
import matplotlib.pyplot as plt


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

        # In-situ liquid volumetric rate  [ft³/day]
        q_liquid_insitu = 5.615 * self.Ql * (self.fp["Bo"] * fo + self.fp["Bw"] * fw)
        self.fp["Vsl"]  = q_liquid_insitu / (86400.0 * self.Ap)

        free_gas_scf   = max(0.0, self.Ql * (self.fp["glr"] - self.fp["gor"] * fo))
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
            return False

        # Griffith-Wallis bubble-flow boundary
        LB = 1.071 - 0.2218 * (Vm ** 2) / self.tid
        LB = max(LB, 0.25)

        lambda_g = Vsg / Vm          # in-situ gas void fraction (no-slip)
        return lambda_g < LB

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

        self.fp["rho_m"] = self.fp["rho_l"] * Hl + self.fp["rho_g"] * (1.0 - Hl)
        self.fp["mu_m"]  = (self.fp["mu_l"] ** Hl) * (self.fp["mu_g"] ** (1.0 - Hl))

        return Hl

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
        # Ngv_safe = max(Ngv, 1e-6)

        # H = ((Nlv / (Ngv_safe ** 0.575))
        #      * (self.fp["Pr"] / 14.7) ** 0.1
        #      * (CNl / Nd))
        
        # H = max(H, 5e-3)

        # Hl_psi = np.sqrt(
        #     (0.0047 + 1123.32 * H + 729489.64 * H ** 2)
        #     / (1.0 + 1097.1566 * H + 722153.97 * H ** 2)
        # )
        Ngv_safe = max(Ngv, 1e-6)

        # 1. Calculate the correlating parameter H (The X-axis of the original chart)
        H = ((Nlv / (Ngv_safe ** 0.575))
             * (self.fp["Pr"] / 14.7) ** 0.1
             * (CNl / Nd))

        # ------------------------------------------------------------------
        # THE FIX: Digitized Hagedorn-Brown Chart Lookup
        # ------------------------------------------------------------------
        
        # X-axis data points (H parameter)
        chart_H = np.array([
            0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0
        ])
        
        # Y-axis data points (Hl/psi parameter) read directly from the original publication
        chart_Hl_psi = np.array([
            0.015, 0.028, 0.048, 0.095, 0.160, 0.260, 0.440, 0.600, 0.770, 0.920, 0.980, 0.995, 1.000
        ])

        # numpy.interp naturally flatlines if H drops below 0.001 or exceeds 10.0
        Hl_psi = np.interp(H, chart_H, chart_Hl_psi)

        B = Ngv * (Nlv ** 0.38) / (Nd ** 2.14)
        if B <= 0.025:
            psi = 27170 * B**3 - 317.52 * B**2 + 0.5472 * B + 0.9999
        elif B <= 0.055:
            psi = -533.33 * B**2 + 58.524 * B + 0.1171
        else:
            psi = 2.5714 * B + 1.5962

        Hl = max(0.0, min(Hl_psi * psi, 1.0))

        self.fp["rho_m"] = self.fp["rho_l"] * Hl + self.fp["rho_g"] * (1.0 - Hl)
        self.fp["mu_m"]  = (self.fp["mu_l"] ** Hl) * (self.fp["mu_g"] ** (1.0 - Hl))

        return Hl

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
        # dimensionless_numbers() populates Vsl, Vsg, Vm — required by is_bubble_flow()
        # liquid_holdup() calls dimensionless_numbers() again internally, but
        # self.fp velocities are already set so the second call is consistent.
        Nl, CNl, Nlv, Ngv, Nd = self.dimensionless_numbers()

        if self.is_bubble_flow():
            return self.griffith_holdup()

        Hl = self.liquid_holdup(Nl, CNl, Nlv, Ngv, Nd)

        # Physical floor: holdup can never fall below the no-slip value —
        # slip between phases can only increase liquid holdup, never decrease it.
        Cl = self.fp["Vsl"] / max(self.fp["Vm"], 1e-6)
        Hl = max(Hl, Cl)

        # Re-blend mixture properties since liquid_holdup() set them using
        # the unfloored Hl
        self.fp["rho_m"] = self.fp["rho_l"] * Hl + self.fp["rho_g"] * (1.0 - Hl)
        self.fp["mu_m"]  = (self.fp["mu_l"] ** Hl) * (self.fp["mu_g"] ** (1.0 - Hl))

        # return 0.7 * Hl + 0.3 * self.griffith_holdup()
        return Hl

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

    # Laminar regime — exact Hagen-Poiseuille result
        if Re < 2000:
            return 64.0 / max(Re, 1.0)
        
        relative_roughness = self.roughness / self.tid   # dimensionless [ft/ft]

    # Jain (1976) explicit approximation — accurate to ±1 % for
    # Re ∈ [3 000, 4×10⁸] and e/D ∈ [4×10⁻⁵, 0.05].
    # This covers virtually every wellbore multiphase flow condition,
    # so there is no practical need to iterate Colebrook-White here.
        f = (1.14 - 2.0 * np.log10(relative_roughness + 21.25 / (Re ** 0.9))) ** -2

        return f

    # ------------------------------------------------------------------
    # Pressure gradient
    # ------------------------------------------------------------------

    def calculate_gradient(self):
        """
        Calculates the total multiphase pressure gradient (psi/ft).

        Routes holdup through get_holdup() which selects Griffith (bubble)
        or Hagedorn-Brown (slug/mist) automatically.

        Returns:
            float: Total pressure gradient in psi/ft.
        """
        Hl = self.get_holdup()
        f  = self.frictional_factor(Hl)

        dp_dh_el   = self.fp["rho_m"] * np.cos(self.theta) / 144.0
        gc         = 32.174
        # dp_dh_fric = (f * self.fp["rho_ns"] * self.fp["Vm"] ** 2) / (2.0 * gc * self.tid * 144.0)
        dp_dh_fric = (f * self.Ql**2 * self.fp["M"]**2 )/(2.9652 * 10**11 * self.tid**5 * self.fp["rho_m"] * 144)

        return dp_dh_el + dp_dh_fric

    # ------------------------------------------------------------------
    # Pressure traverse
    # ------------------------------------------------------------------

    def calculate_pressure_traverse(self, Pth, surface_temp, bottomhole_temp,
                                    total_depth, step_size, Ql):
        """
        Calculates the wellbore pressure profile via Euler integration.

        Returns:
            tuple: (depths [ft], pressures [psia])
        """
        depths        = [0.0]
        pressures     = [Pth]
        current_P     = Pth
        current_depth = 0.0
        temp_gradient = (bottomhole_temp - surface_temp) / total_depth

        while current_depth < total_depth:
            next_depth   = min(current_depth + step_size, total_depth)
            actual_step  = next_depth - current_depth
            current_temp = surface_temp + temp_gradient * current_depth

            self.update_fluid_properties(current_P, current_temp, Ql)
            dp_dz     = self.calculate_gradient()
            current_P += dp_dz * actual_step
            current_depth = next_depth

            depths.append(current_depth)
            pressures.append(current_P)

        return depths, pressures

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_pressure_traverse(self, Pth, surface_temp, bottomhole_temp,
                               total_depth, step_size, Ql):
        depths, pressures = self.calculate_pressure_traverse(
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
            _, pressures = self.calculate_pressure_traverse(
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

        def get_C_and_Hl0(pattern):
            if pattern == "segregated":
                hl0 = 0.98 * (Cl**0.4846 / Nfr**0.0868)
                c = (1 - Cl) * np.log(0.011 * Nlv**3.539 * Cl**(-3.768) * Nfr**(-1.614))
            elif pattern == "intermittent":
                hl0 = 0.845 * (Cl**0.5351 / Nfr**0.0173)
                c = (1 - Cl) * np.log(2.96 * Nlv**(-0.4473) * Cl**0.305 * Nfr**(0.0978))
            else:
                hl0 = 1.065 * (Cl**0.5824 / Nfr**0.0609)
                c = 0.0
            return max(hl0, Cl), max(c, 0.0)

        if flow_pattern == "transitional":
            hl0_seg, c_seg = get_C_and_Hl0("segregated")
            hl_seg = hl0_seg * (1.0 + c_seg * (sin_term - 0.333 * sin_term**3))

            hl0_int, c_int = get_C_and_Hl0("intermittent")
            hl_int = hl0_int * (1.0 + c_int * (sin_term - 0.333 * sin_term**3))

            A = (L3 - Nfr) / (L3 - L2)
            Hl = A * hl_seg + (1.0 - A) * hl_int
        else:
            hl0, c = get_C_and_Hl0(flow_pattern)
            Hl = hl0 * (1.0 + c * (sin_term - 0.333 * sin_term**3))

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
    
    def calculate_gradient(self, P):
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
        
        dp_dz = (hydrostatic + friction) / kinetic_term / 144.0
        return dp_dz
    
    def calculate_pressure_traverse(self, Pth, surface_temp, bottomhole_temp,
                                    total_depth, step_size, Ql):
        """
        Calculates the wellbore pressure profile via Euler integration.

        Args:
            Pth (float): Wellhead tubing pressure in psia.
            surface_temp (float): Surface temperature in Fahrenheit (°F).
            bottomhole_temp (float): Bottomhole temperature in Fahrenheit (°F).
            total_depth (float): Total vertical depth of the well in feet (ft).
            step_size (float): Depth integration step size in feet (ft).
            Ql (float): Surface liquid flow rate in STB/day.

        Returns:
            tuple: (depths [ft], pressures [psia])
        """
        depths = [0.0]
        pressures = [Pth]
        current_P = Pth
        current_depth = 0.0
        temp_gradient = (bottomhole_temp - surface_temp) / total_depth

        while current_depth < total_depth:
            next_depth = min(current_depth + step_size, total_depth)
            actual_step = next_depth - current_depth
            current_temp = surface_temp + temp_gradient * current_depth

            # Fixed: Call internal method properly
            self._update_fluid_properties(current_P, current_temp, Ql)
            dp_dz = self.calculate_gradient(current_P)
            
            current_P += dp_dz * actual_step
            current_depth = next_depth

            depths.append(current_depth)
            pressures.append(current_P)

        return depths, pressures
    
    def plot_vlp_curve(self, Pth, surface_temp, bottomhole_temp,
                       depth, Qmin, Qmax, step_size):
        Pwf_points = []
        rates = np.linspace(Qmin, Qmax, Qmax - Qmin)

        for q in rates:
            _, pressures = self.calculate_pressure_traverse(
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