import numpy as np


class BlackOilPVT:
    """
    A comprehensive Black Oil PVT calculator using standard industry empirical correlations
    (Standing, Beggs-Robinson, Lee-Gonzalez-Eakin, Dranchuk-Abou-Kassem).
    Assumes standard US Oilfield Units.
    """

    def __init__(self, sg_gas=0.65, sg_oil=0.84, oil_api=None, sg_water=1.03, watercut=0.0):
        """
        Initializes the fluid with its constant surface properties.

        Args:
            sg_gas (float): Specific gravity of gas (Air = 1.0).
            sg_oil (float): Specific gravity of oil (Water = 1.0).
            oil_api (float, optional): Oil API gravity. Overrides sg_oil if provided.
            sg_water (float, optional): Specific gravity of water. Defaults to 1.03.
            watercut (float): Watercut as a decimal fraction (0.0 to 1.0).
        """
        self.sg_g = sg_gas
        self.sg_o = 141.5 / (131.5 + oil_api) if oil_api else sg_oil
        self.sg_w = sg_water
        self.wc = watercut

        # BUG FIX 1: Guard against division by zero at watercut = 1.0.
        # wor = wc / (1 - wc) blows up at full water production.
        self.wor = watercut / max(1.0 - watercut, 1e-9)

        # Calculate API gravity (used in almost all oil correlations)
        self.api = (141.5 / self.sg_o) - 131.5

    def get_pseudo_critical(self):
        """
        Calculates Sutton's pseudo-critical temperature and pressure for natural gas.

        Returns:
            tuple:
                Tpc (float): Pseudo-critical temperature in Rankine (°R).
                Ppc (float): Pseudo-critical pressure in psia.
        """
        Tpc = 187 + 330 * self.sg_g - 71.5 * self.sg_g**2
        Ppc = 667 + 15 * self.sg_g - 37.5 * self.sg_g**2
        return Tpc, Ppc

    def calculate_dak_z_factor(self, P, T, gamma_g, max_iter=100, tolerance=1e-6):
        """
        Calculates the Gas Z-factor using the Dranchuk-Abou-Kassem (DAK)
        equation of state for dry natural gas via Newton-Raphson iteration.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).
            gamma_g (float): Gas specific gravity.
            max_iter (int, optional): Maximum iterations for the solver. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.

        Returns:
            float: Gas compressibility factor (Z-factor), dimensionless.
        """
        T_r = T + 459.67
        Tpc, Ppc = self.get_pseudo_critical()
        Tpr = T_r / Tpc
        Ppr = P / Ppc

        A1, A2, A3 = 0.3265, -1.0700, -0.5339
        A4, A5, A6 = 0.01569, -0.05165, 0.5475
        A7, A8, A9 = -0.7361, 0.1844, 0.1056
        A10, A11 = 0.6134, 0.7210

        c1 = A1 + A2/Tpr + A3/(Tpr**3) + A4/(Tpr**4) + A5/(Tpr**5)
        c2 = A6 + A7/Tpr + A8/(Tpr**2)
        c3 = A9 * (A7/Tpr + A8/(Tpr**2))
        c4 = A10 / (Tpr**3)

        rho_r = 0.27 * Ppr / Tpr

        for i in range(max_iter):
            term_exp = np.exp(-A11 * (rho_r**2))

            f = (1.0 + c1*rho_r + c2*(rho_r**2) - c3*(rho_r**5) +
                 c4*(rho_r**2)*(1.0 + A11*(rho_r**2))*term_exp - (0.27 * Ppr / (rho_r * Tpr)))

            # BUG FIX 2: The analytical derivative of the DAK EOS c4 term was wrong.
            # Differentiating c4*rho^2*(1 + A11*rho^2)*exp(-A11*rho^2) w.r.t. rho gives:
            #   2*c4*rho*(1 + A11*rho^2 - A11^2*rho^4)*exp(-A11*rho^2)
            # The original code contained a spurious -A11*rho^3 term inside the bracket,
            # which caused Newton-Raphson to converge to a slightly wrong root.
            df = (c1 + 2.0*c2*rho_r - 5.0*c3*(rho_r**4) +
                  2.0*c4*rho_r*(1.0 + A11*(rho_r**2) - A11**2*(rho_r**4))*term_exp +
                  (0.27 * Ppr / ((rho_r**2) * Tpr)))

            rho_r_new = rho_r - f / df

            if abs(rho_r_new - rho_r) < tolerance:
                z_factor = 0.27 * Ppr / (rho_r_new * Tpr)
                return round(z_factor, 4)

            rho_r = rho_r_new

        raise ValueError("Z-factor solver did not converge. Check input parameters.")

    def calc_bubble_point(self, T, Rsb):
        """
        Calculates Bubble Point Pressure (Pb) using Standing's Correlation.

        Args:
            T (float): Local temperature in Fahrenheit (°F).
            Rsb (float): Solution GOR at bubble point (Initial Producing GOR) in scf/STB.

        Returns:
            float: Bubble point pressure in psia.
        """
        a = 0.00091 * T - 0.0125 * self.api
        pb = 18.2 * (((Rsb / self.sg_g) ** 0.83) * (10 ** a) - 1.4)
        return max(pb, 14.7)

    def calc_rs(self, P, T, Pb, Rsb):
        """
        Calculates Solution Gas-Oil Ratio (Rs) using Standing's Correlation.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).
            Pb (float): Bubble point pressure in psia.
            Rsb (float): Solution GOR at bubble point in scf/STB.

        Returns:
            float: Solution Gas-Oil Ratio in scf/STB.
        """
        if P >= Pb:
            return Rsb
        a = 0.00091 * T - 0.0125 * self.api
        rs = self.sg_g * (((P / 18.2) + 1.4) * (10 ** -a)) ** 1.2048
        return max(rs, 0.0)

    def calc_bo(self, P, T, Rs, Pb):
        """
        Calculates Oil Formation Volume Factor (Bo) using Standing's Correlation.
        Includes simplified Vasquez-Beggs isothermal compressibility for undersaturated conditions.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).
            Rs (float): Solution Gas-Oil Ratio at local pressure in scf/STB.
            Pb (float): Bubble point pressure in psia.

        Returns:
            float: Oil Formation Volume Factor (Bo) in bbl/STB.
        """
        F = Rs * np.sqrt(self.sg_g / self.sg_o) + 1.25 * T
        bo_sat = 0.9759 + 0.00012 * (F ** 1.175)

        if P <= Pb:
            return bo_sat

        co = (5 * Rs + 17.2 * T - 1180 * self.sg_g + 12.61 * self.api - 1433) / (P * 10**5)
        co = max(co, 5e-6)
        bo_under = bo_sat * np.exp(-co * (P - Pb))
        return bo_under

    def calc_bg(self, P, T, Z):
        """
        Calculates Gas Formation Volume Factor (Bg) using the Real Gas Law.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).
            Z (float): Gas compressibility factor.

        Returns:
            float: Gas Formation Volume Factor (Bg) in ft³/scf.
        """
        T_rankine = T + 460.0
        bg = 0.02827 * Z * T_rankine / P
        return bg

    def calc_bw(self, P, T):
        """
        Calculates Water Formation Volume Factor (Bw) using polynomial approximation.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).

        Returns:
            float: Water Formation Volume Factor (Bw) in bbl/STB.
        """
        delta_T = T - 60.0
        bw = 1.0 + 1.2e-4 * delta_T + 1e-6 * (delta_T ** 2) - 3.33e-6 * P
        return max(bw, 0.9)

    def calc_density_oil(self, Rs, Bo):
        """
        Calculates in-situ Oil Density.

        Args:
            Rs (float): Solution Gas-Oil Ratio at local pressure in scf/STB.
            Bo (float): Oil Formation Volume Factor at local pressure in bbl/STB.

        Returns:
            float: In-situ oil density in lbm/ft³.
        """
        rho_o_insitu = (350.0 * self.sg_o + 0.0764 * self.sg_g * Rs) / (5.615 * Bo)
        return rho_o_insitu

    def calc_density_gas(self, P, T, Z):
        """
        Calculates in-situ Gas Density using the Real Gas Law.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).
            Z (float): Gas compressibility factor.

        Returns:
            float: In-situ gas density in lbm/ft³.
        """
        T_rankine = T + 460.0
        rho_g_insitu = 28.967 * self.sg_g * P / (Z * 10.732 * T_rankine)
        return rho_g_insitu

    def calc_viscosity_oil(self, P, T, Rs, Pb):
        """
        Calculates Oil Viscosity using Beggs-Robinson Correlation.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).
            Rs (float): Solution Gas-Oil Ratio at local pressure in scf/STB.
            Pb (float): Bubble point pressure in psia.

        Returns:
            float: Oil viscosity in centipoise (cp).
        """
        z = 3.0324 - 0.02023 * self.api
        y = 10 ** z
        x = y * (T ** -1.163)
        mu_od = 10 ** x - 1.0

        a = 10.715 * ((Rs + 100) ** -0.515)
        b = 5.44 * ((Rs + 150) ** -0.338)
        mu_os = a * (mu_od ** b)

        if P <= Pb:
            return mu_os

        m = 2.6 * (P ** 1.187) * np.exp(-11.513 - 8.98e-5 * P)
        mu_o_under = mu_os * ((P / Pb) ** m)
        return mu_o_under

    def calc_viscosity_gas(self, P, T, Z):
        """
        Calculates Gas Viscosity using the Lee-Gonzalez-Eakin correlation.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).
            Z (float): Gas compressibility factor.

        Returns:
            float: Gas viscosity in centipoise (cp).
        """
        T_rankine = T + 460.0
        Mg = 28.97 * self.sg_g
        rho_g = self.calc_density_gas(P, T, Z) / 62.4

        K = (9.379 + 0.01607 * Mg) * (T_rankine ** 1.5) / (209.2 + 19.26 * Mg + T_rankine)
        X = 3.448 + (986.4 / T_rankine) + 0.01009 * Mg
        Y = 2.447 - 0.2224 * X

        mu_g = 1e-4 * K * np.exp(X * (rho_g ** Y))
        return mu_g

    def calc_viscosity_water(self, T):
        """
        Calculates water viscosity using the Brill & Beggs correlation.

        Args:
            T (float): Local temperature in Fahrenheit (°F).

        Returns:
            float: Water viscosity in centipoise (cp).
        """
        return np.exp(1.003 - 0.01479 * T + 0.00001982 * T**2)

    def calc_surface_tension_oil(self, P, T):
        """
        Calculates Liquid-Gas Surface Tension for oil using Baker-Swerdloff.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).

        Returns:
            float: Oil surface tension in dynes/cm.
        """
        sigma_68 = 39.0 - 0.2571 * self.api
        sigma_100 = 37.5 - 0.2571 * self.api

        if T > 100:
            sigma_dead = sigma_100 - (T - 100) * 0.05
        else:
            sigma_dead = sigma_68 - (T - 68) * ((sigma_68 - sigma_100) / 32.0)

        sigma_dead = max(sigma_dead, 1.0)
        sigma_live = sigma_dead * np.exp(-0.0002 * P)
        return max(sigma_live, 1.0)

    def calc_surface_tension_water(self, T):
        """
        Calculates Water-Gas Surface Tension.

        Args:
            T (float): Local temperature in Fahrenheit (°F).

        Returns:
            float: Water surface tension in dynes/cm.
        """
        sigma_w = 75.0 - 0.116 * T
        return max(sigma_w, 40.0)

    def calc_M(self, gor):
        """
        Calculates the mixture molecular weight proxy M used in the H-B friction Re number.

        Args:
            gor (float): Solution Gas-Oil Ratio at current pressure in scf/STB oil.

        Returns:
            float: M (lbm/STB liquid), dimensionless in context of Re formula.
        """
        # BUG FIX 4: GLR conversion was wrong. GLR (gas per STB *liquid*) equals
        # GOR (gas per STB *oil*) multiplied by the oil fraction (1 - wc), i.e.:
        #   glr = gor * (1 - wc) = gor / (1 + wor)
        # The original code used gor / (1 + wc), which is dimensionally incorrect
        # and over-estimates GLR at any nonzero watercut.
        glr = gor / (1.0 + self.wor)
        return (self.sg_o * 350.52 / (1 + self.wor)
                + self.sg_w * 350.52 * self.wor / (1 + self.wor)
                + self.sg_g * 0.0764 * glr)
    
    def rsb_from_test(self, Rs_test, Pwf_test, Pb, T):
        """
        Calculates bubble-point gor from test data.
        Args:
            Rs_test: GOR at test condition, scf/stb
            Pwf_test: Wellbore pressure at test condition, psia
            Pb: Bubble-point pressure, psia
            T: Test temperature, Farenheit
        Returns:
            Rsb: GOR at bubble-point, scf/stb
        """
        if Pwf_test >= Pb:
            return Rs_test          # test point already saturated -> Rs_test IS Rsb
        a = 0.00091 * T - 0.0125 * self.api
        shape_at_test = self.sg_g * (((Pwf_test / 18.2) + 1.4) * (10 ** -a)) ** 1.2048
        shape_at_pb   = self.sg_g * (((Pb / 18.2) + 1.4) * (10 ** -a)) ** 1.2048
        return Rs_test * (shape_at_pb / shape_at_test)

    def fluid_properties_dict(self, P, T, Rsb, Pb=0):
        """
        Calculates all fluid properties at a specific pressure and temperature node.
        Correctly blends oil and water properties based on in-situ volume fractions.

        Args:
            P (float): Local pressure in psia.
            T (float): Local temperature in Fahrenheit (°F).
            Rsb (float): Initial producing Gas-Oil Ratio in scf/STB.
            Pb (float): Bubble-point pressure in psia. Calculated if 0.

        Returns:
            dict: A dictionary containing all blended liquid and gas properties required
                  by multiphase flow correlations like Hagedorn-Brown.
        """
        Pb = self.calc_bubble_point(T, Rsb) if Pb == 0 else Pb

        Rs = self.calc_rs(P, T, Pb, Rsb)
        Z = self.calculate_dak_z_factor(P, T, self.sg_g)
        M = self.calc_M(Rs)

        Bo = self.calc_bo(P, T, Rs, Pb)
        Bw = self.calc_bw(P, T)
        Bg = self.calc_bg(P, T, Z)

        # Surface volume fractions (used for viscosity and surface tension blending,
        # consistent with H-B correlation literature)
        fo = 1.0 / (1.0 + self.wor)
        fw = self.wor / (1.0 + self.wor)

        # In-situ volume fractions (used for density blending)
        # These account for the reservoir FVF of each phase.
        total_res_vol = Bo + self.wor * Bw
        fo_insitu = Bo / total_res_vol
        fw_insitu = (self.wor * Bw) / total_res_vol

        # Pure phase properties
        rho_o = self.calc_density_oil(Rs, Bo)
        rho_w = 62.4 * self.sg_w / Bw
        mu_o = self.calc_viscosity_oil(P, T, Rs, Pb)
        mu_w = self.calc_viscosity_water(T)
        sigma_o = self.calc_surface_tension_oil(P, T)
        sigma_w = self.calc_surface_tension_water(T)

        rho_l = rho_o * fo_insitu + rho_w * fw_insitu

        # Viscosity and surface tension: blended with surface fractions per H-B convention
        mu_l = mu_o * fo + mu_w * fw
        sigma_l = sigma_o * fo + sigma_w * fw

        return {
            "M": M,
            "Pb": Pb,
            "Rsb": Rsb,
            "gor": Rs,
            "glr": Rsb / (1.0 + self.wor),
            "rho_l": rho_l,
            "rho_g": self.calc_density_gas(P, T, Z),
            "mu_l": mu_l,
            "mu_g": self.calc_viscosity_gas(P, T, Z),
            "sigma_l": sigma_l,
            "Bo": Bo,
            "Bg": Bg,   # ft³/scf
            "Bw": Bw,
            "Pr": P,
            "Tr": T,
            "Z": Z,
        }