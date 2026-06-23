import numpy as np
import matplotlib.pyplot as plt

class composite_ipr:
    """
    Calculates and plots the Inflow Performance Relationship (IPR) using the 
    Composite Darcy-Vogel model for undersaturated and saturated oil reservoirs.
    """

    def __init__(self, Pr, Pb, q_test, Pwf_test):
        """
        Initializes the IPR model and locks in the well's Productivity Index (J).

        Args:
            Pr (float): Static Reservoir Pressure in psia.
            Pb (float): Bubble Point Pressure in psia.
            q_test (float): Tested liquid flow rate in STB/day.
            Pwf_test (float): Tested Bottom-Hole Flowing Pressure in psia.
        """
        self.Pr = Pr
        self.Pb = Pb
        self.q_test = q_test
        self.Pwf_test = Pwf_test
        
        # 1. Lock in the Productivity Index (J) based on test data
        self.J = self._calculate_pi()
        
        # 2. Calculate flow rate exactly at the bubble point (q_bp)
        self.q_bp = self.J * (self.Pr - self.Pb)
        
        # 3. Calculate Absolute Open Flow (q_max)
        self.q_max = self._calculate_aof()

    def _calculate_pi(self):
        """
        Calculates the Productivity Index (J) using the initial test point.
        Internal method called only during initialization.

        Returns:
            float: Productivity Index in STB/day/psi.
        """
        if self.Pwf_test >= self.Pb:
            # Darcy linear regime
            return self.q_test / (self.Pr - self.Pwf_test)
        else:
            # Vogel curve regime (Note: Denominator is Pb, not Pr)
            vogel_term = 1.0 - 0.2 * (self.Pwf_test / self.Pb) - 0.8 * (self.Pwf_test / self.Pb)**2
            return self.q_test / ((self.Pr - self.Pb) + (self.Pb / 1.8) * vogel_term)

    def _calculate_aof(self):
        """
        Calculates Absolute Open Flow (AOF) where Pwf equals zero.

        Returns:
            float: Maximum flow rate (q_max) in STB/day.
        """
        return self.q_bp + (self.J * self.Pb) / 1.8

    def calculate_q(self, Pwf):
        """
        Calculates the expected flow rate (q) for a given Bottom-Hole Pressure.

        Args:
            Pwf (float): Target Bottom-Hole Flowing Pressure in psia.

        Returns:
            float: Calculated flow rate in STB/day.
        """
        if Pwf >= self.Pb:
            # Darcy linear regime
            return self.J * (self.Pr - Pwf)
        else:
            # Vogel curve regime
            vogel_term = 1.0 - 0.2 * (Pwf / self.Pb) - 0.8 * (Pwf / self.Pb)**2
            return self.q_bp + (self.J * self.Pb / 1.8) * vogel_term

    def calculate_Pwf(self, q):
        """
        Calculates the required Bottom-Hole Pressure (Pwf) for a target flow rate.

        Args:
            q (float): Target liquid flow rate in STB/day.

        Returns:
            float: Calculated Bottom-Hole Flowing Pressure in psia.
        """
        # Bound the query so we don't try to calculate past Absolute Open Flow
        q_target = min(q, self.q_max)

        if q_target <= self.q_bp:
            # Darcy linear regime
            return self.Pr - (q_target / self.J)
        else:
            # Vogel quadratic inversion
            C = (1.8 / self.Pb) * ((q_target / self.J) - self.Pr + self.Pb)
            radicand = max(0.0, 3.24 - 3.2 * C)
            Pwf = self.Pb * (np.sqrt(radicand) - 0.2) / 1.6
            return max(0.0, Pwf)

    def plot_ipr(self, points=50):
        """
        Generates a plot of the IPR curve from zero flow up to Absolute Open Flow.

        Args:
            points (int): Number of calculation nodes to generate a smooth curve.
        """
        Q = np.linspace(0, self.q_max, points)
        Pwf_points = [self.calculate_Pwf(q) for q in Q]
        
        plt.figure(figsize=(10, 6))
        plt.plot(Q, Pwf_points, label='Calculated IPR Curve', color='crimson', linewidth=2)
        plt.scatter(self.q_test, self.Pwf_test, color='black', zorder=5, label='Well Test Data Point')
        plt.scatter(self.q_bp, self.Pb, color='blue', marker='x', s=100, zorder=5, label='Bubble Point Transition')
        
        plt.title('Inflow Performance Relationship (IPR)', fontweight='bold')
        plt.xlabel('Liquid Flow Rate, q (STB/day)')
        plt.ylabel('Bottom-Hole Flowing Pressure, Pwf (psia)')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

class darcy_ipr:
    """
    Simple linear Darcy IPR model for comparison and testing purposes.
    """

    def __init__(self, Pr, Pb, q_test, Pwf_test):
        self.Pr = Pr
        self.Pb = Pb
        self.q_test = q_test
        self.Pwf_test = Pwf_test
        self.J = self._calculate_pi(q_test, Pwf_test)
        self.q_max = self.J * self.Pr  # Absolute Open Flow at Pwf=0

    def _calculate_pi(self, q_test, Pwf_test):
        return q_test / (self.Pr - Pwf_test)

    def calculate_q(self, Pwf):
        return self.J * (self.Pr - Pwf)

    def calculate_Pwf(self, q):
        return self.Pr - (q / self.J)
    
    def _calculate_aof(self):
        return self.q_max  # For Darcy, AOF is simply J * Pr

    def plot_ipr(self, points=50):
        Q = np.linspace(0, self.q_max, points)
        Pwf_points = [self.calculate_Pwf(q) for q in Q]
        
        plt.figure(figsize=(10, 6))
        plt.plot(Q, Pwf_points, label='Darcy IPR Curve', color='navy', linewidth=2)
        plt.scatter(self.q_test, self.Pwf_test, color='black', zorder=5, label='Well Test Data Point')
        
        plt.title('Darcy Inflow Performance Relationship (IPR)', fontweight='bold')
        plt.xlabel('Liquid Flow Rate, q (STB/day)')
        plt.ylabel('Bottom-Hole Flowing Pressure, Pwf (psia)')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

class vogel_ipr:
    """
    Calculates and plots the Inflow Performance Relationship (IPR) using 
    pure Vogel's equation for saturated oil reservoirs.
    """
    def __init__(self, Pr, Pb, q_test, Pwf_test):
        self.Pr = Pr
        self.Pb = Pb  # Included for API compatibility with other models
        self.q_test = q_test
        self.Pwf_test = Pwf_test
        
        self.q_max = self._calculate_aof()
        # J is approximated at Pr (initial slope) for GUI compatibility
        self.J = 1.8 * self.q_max / self.Pr if self.Pr > 0 else 0.0

    def _calculate_aof(self):
        vogel_term = 1.0 - 0.2 * (self.Pwf_test / self.Pr) - 0.8 * (self.Pwf_test / self.Pr)**2
        return self.q_test / vogel_term if vogel_term > 0 else 0.0

    def calculate_q(self, Pwf):
        if Pwf >= self.Pr:
            return 0.0
        vogel_term = 1.0 - 0.2 * (Pwf / self.Pr) - 0.8 * (Pwf / self.Pr)**2
        return self.q_max * vogel_term

    def calculate_Pwf(self, q):
        q_target = min(q, self.q_max)
        if self.q_max <= 0:
            return 0.0
        radicand = max(0.0, 3.24 - 3.2 * (q_target / self.q_max))
        Pwf = self.Pr * (np.sqrt(radicand) - 0.2) / 1.6
        return max(0.0, Pwf)

    def plot_ipr(self, points=50):
        Q = np.linspace(0, self.q_max, points)
        Pwf_points = [self.calculate_Pwf(q) for q in Q]
        
        plt.figure(figsize=(10, 6))
        plt.plot(Q, Pwf_points, label='Vogel IPR Curve', color='green', linewidth=2)
        plt.scatter(self.q_test, self.Pwf_test, color='black', zorder=5, label='Well Test Data Point')
        
        plt.title('Vogel Inflow Performance Relationship (IPR)', fontweight='bold')
        plt.xlabel('Liquid Flow Rate, q (STB/day)')
        plt.ylabel('Bottom-Hole Flowing Pressure, Pwf (psia)')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


class fetkovich_ipr:
    """
    Calculates and plots the Inflow Performance Relationship (IPR) using 
    Fetkovich for reservoirs for non-Darcy flow.
    """
    def __init__(self, Pr, Pb, q_test, Pwf_test, q_test2, Pwf_test2):
        self.Pr = Pr
        self.Pb = Pb  # Included for API compatibility with other models
        self.q_test = q_test
        self.Pwf_test = Pwf_test
        self.q_test2 = q_test2
        self.Pwf_test2 = Pwf_test2
        
        self.n = self._calculate_exponent()
        self.C = self._calculate_constant()
        self.q_max = self._calculate_aof()
        self.J = (self.q_test * 2 * self.Pb) / ((self.Pr**2 - self.Pwf_test**2)**self.n)

    def _calculate_exponent(self):
        n = (np.log(self.q_test) - np.log(self.q_test2))/(np.log(self.Pr**2 - self.Pwf_test**2) - (np.log(self.Pr**2 - self.Pwf_test2**2) ))
        if n < 0.5 or n > 1.0:
            print(
            f"Calculated Fetkovich exponent (n = {n:.2f}) falls outside the physical "
            f"bounds of 0.5 to 1.0. This indicates un-stabilized well tests, changing skin, "
            f"or bad gauge data. The value will be clipped to the nearest physical bound."
            )
        return np.clip(n, 0.5, 1.0)

    def _calculate_constant(self):
        return self.q_test/(self.Pr**2 - self.Pwf_test**2) **self.n

    def _calculate_aof(self):
        return self.C * self.Pr**(2*self.n)
    
    def calculate_q(self, Pwf):
        return self.C * (self.Pr**2 - Pwf**2)**self.n

    def calculate_Pwf(self, q):
        if q >= self.q_max:
            return 0.0
        core = self.Pr**2 - (q/self.C)**(1/self.n)
        return np.sqrt(max(0.0, core))

    def plot_ipr(self, points=50):
        Q = np.linspace(0, self.q_max, points)
        Pwf_points = [self.calculate_Pwf(q) for q in Q]
        
        plt.figure(figsize=(10, 6))
        plt.plot(Q, Pwf_points, label='Fetkovich IPR Curve', color='green', linewidth=2)
        plt.scatter(self.q_test, self.Pwf_test, color='black', zorder=5, label='Well Test Data Point')
        
        plt.title('Fetkovic Inflow Performance Relationship (IPR)', fontweight='bold')
        plt.xlabel('Liquid Flow Rate, q (STB/day)')
        plt.ylabel('Bottom-Hole Flowing Pressure, Pwf (psia)')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


import numpy as np
import matplotlib.pyplot as plt

class composite_fetkovich_ipr:
    """
    Calculates and plots the Composite Inflow Performance Relationship (IPR) 
    for undersaturated and saturated oil wells using the Fetkovich method.
    """
    def __init__(self, Pr, Pb, q_test, Pwf_test, J = None, C=None, n=1.0):
        """
        Initialize the Composite Fetkovich IPR model.
        
        Parameters:
        Pr (float): Reservoir pressure (psia)
        Pb (float): Bubble point pressure (psia)
        J  (float): Productivity Index above bubble point (STB/d/psi)
        C  (float): Fetkovich performance coefficient. If None, it forces 
                    smooth derivative continuity at Pb.
        n  (float): Fetkovich exponent (typically 0.5 to 1.0)
        """
        self.Pr = Pr
        # If reservoir pressure is below bubble point, it's a saturated reservoir.
        # We cap Pb at Pr to safely handle saturated cases.
        self.Pb = min(Pb, Pr) 
        self.J = J if J is not None else (q_test * 2 * self.Pb) / ((self.Pr**2 - Pwf_test**2)**self.n)
        self.n = n
        
        # 1. Calculate Flow Rate at the Bubble Point (Single-Phase Region)
        self.qb = self.J * (self.Pr - self.Pb)
        
        # 2. Calculate Fetkovich Constant 'C' (if not explicitly provided)
        # By default, we ensure the derivative of the Fetkovich curve matches 
        # the linear PI (J) exactly at the bubble point for a smooth transition.
        if C is None:
            if self.Pb > 0:
                # Derived from matching slopes: dq/dPwf(Fetkovich) = dq/dPwf(Linear) at Pb
                self.C = self.J / (2 * self.n * self.Pb**(2 * self.n - 1))
            else:
                self.C = 0.0
        else:
            self.C = C
            
        # 3. Calculate Absolute Open Flow (AOF / q_max) at Pwf = 0
        self.q_max = self.qb + self.C * (self.Pb**2)**self.n

    def calculate_q(self, Pwf):
        """Calculates flow rate (q) for a given bottom-hole pressure (Pwf)."""
        Pwf = max(0.0, Pwf) # Pressure cannot be negative
        
        if Pwf >= self.Pb:
            # Single-Phase Linear Flow
            return max(0.0, self.J * (self.Pr - Pwf))
        else:
            # Two-Phase Fetkovich Flow
            return self.qb + self.C * (self.Pb**2 - Pwf**2)**self.n

    def calculate_Pwf(self, q):
        """Calculates bottom-hole pressure (Pwf) for a given flow rate (q)."""
        if q <= 0:
            return self.Pr
        if q >= self.q_max:
            return 0.0 # Well is completely drawn down
            
        if q <= self.qb:
            # We are in the Single-Phase Linear Flow regime
            return self.Pr - (q / self.J)
        else:
            # We are in the Two-Phase Fetkovich Flow regime
            # Rearranged: q = qb + C(Pb^2 - Pwf^2)^n
            core = self.Pb**2 - ((q - self.qb) / self.C)**(1 / self.n)
            # max(0, core) prevents math domain errors (taking sqrt of negatives)
            return np.sqrt(max(0.0, core))

    def plot_ipr(self, points=100):
        """Generates the standard IPR curve plot."""
        Q = np.linspace(0, self.q_max, points)
        Pwf_points = [self.calculate_Pwf(q) for q in Q]
        
        plt.figure(figsize=(10, 6))
        
        # Plot the main curve
        plt.plot(Q, Pwf_points, label='Composite Fetkovich IPR', color='#2ca02c', linewidth=2.5)
        
        # Highlight the Bubble Point transition
        if self.Pb < self.Pr:
            plt.scatter(self.qb, self.Pb, color='red', zorder=5, s=60, 
                        label=f'Bubble Point ({self.qb:.1f} STB/d, {self.Pb} psia)')
            
            # Optional: Add dashed lines to show the regimes
            plt.axhline(self.Pb, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(self.qb, color='gray', linestyle='--', alpha=0.5)
            plt.text(self.qb * 0.5, self.Pr, 'Single-Phase\n(Linear)', 
                     ha='center', va='top', alpha=0.7)
            plt.text(self.qb + (self.q_max - self.qb)*0.5, self.Pb * 0.5, 'Two-Phase\n(Fetkovich)', 
                     ha='center', alpha=0.7)

        plt.title('Composite Fetkovich Inflow Performance Relationship', fontsize=14, fontweight='bold')
        plt.xlabel('Liquid Flow Rate, q (STB/day)', fontsize=12)
        plt.ylabel('Bottom-Hole Flowing Pressure, $P_{wf}$ (psia)', fontsize=12)
        plt.ylim(bottom=0, top=self.Pr * 1.05)
        plt.xlim(left=0, right=self.q_max * 1.05)
        
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()