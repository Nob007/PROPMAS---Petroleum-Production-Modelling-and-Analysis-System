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