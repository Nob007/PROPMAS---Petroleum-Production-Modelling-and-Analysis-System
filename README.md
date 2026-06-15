# Nodal Analysis — Integrated Production Modelling (IPM)

A robust, Python-based desktop application for petroleum engineers to perform Nodal Analysis. This application calculates and visualizes the intersection of Inflow Performance Relationship (IPR) and Vertical Lift Performance (VLP) curves to find the stable operating point of a well.

![Theme](https://img.shields.io/badge/Theme-White_%26_Blue-1565C0)
![GUI](https://img.shields.io/badge/GUI-PyQt6-brightgreen)
![Math](https://img.shields.io/badge/Math-SciPy_%26_NumPy-blue)

## ✨ Features

### 📉 Reservoir & Wellbore Modelling
*   **IPR Models**: Supports Composite (Darcy/Vogel), pure Vogel, Darcy (Linear), and Fetkovitch models. Automatically handles the bubble point transition.
*   **VLP Models**: Hagedorn-Brown multiphase flow correlation with Griffith-Wallis bubble flow detection and accurate liquid holdup corrections. Duns and Ross correlation. Beggs and Brills correlation
*   **Black Oil PVT**: Automatically calculates fluid properties (Z-factor, Rs, Bo, Bg, Bw, viscosities, and densities) across the wellbore using industry-standard correlations (Standing, Beggs-Robinson, Lee-Gonzalez-Eakin, Dranchuk-Abou-Kassem).
*   **Solver**: Utilizes Scipy's highly reliable Brent's method (`brentq`) to pinpoint operating points accurately.

### 🖥️ Interactive User Interface
*   **Modern Design**: Clean white and blue accent theme built with PyQt6.
*   **Collapsible Sidebar**: Easily manage inputs for IPR, Fluid Properties, VLP, and Rate Ranges without cluttering the screen.
*   **Interactive Matplotlib Charts**: Includes responsive hover tooltips for exact pressure/rate readings and an auto-updating legend.
*   **Pressure Traverse Mini-Map**: A signature floating widget that shows a detailed Depth vs. Pressure traverse for the solved operating point. Can be expanded/collapsed for closer inspection.
*   **Sensitivity Analysis**: Run multi-curve parameter sweeps (e.g., varying THP, GOR, Tubing ID, or Water Cut) overlaying the results directly onto the chart.
*   **Multithreaded**: Calculations are handled by background workers, keeping the UI responsive and providing a live status/loading bar.

## 📂 Project Structure

```text
Nodal_IPM_Project/
│
├── main.py                    # The core entry point that launches the app
│
├── core/                      # Pure mathematical & physics models
│   ├── __init__.py
│   ├── pvt.py                 # BlackOilPVT class & correlations
│   ├── vlp.py                 # HagedornBrown multiphase wellbore integration
│   ├── ipr.py                 # Darcy & Vogel inflow equations
│   └── solver.py              # Root-finding optimizer for Operating Point
│
├── gui/                       # User interface components
│   ├── __init__.py
│   └── main.py                # Main PyQt6 Window, Chart Canvas, and Threads
│
├── assets/                    # Static files (Icons, images)
├── requirements.txt           # Python dependencies
└── build.py                   # PyInstaller script to compile the .exe
```

## ⚙️ Prerequisites & Installation

Make sure you have Python 3.9+ installed. You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

*(Key dependencies include: `numpy`, `scipy`, `matplotlib`, `PyQt6`)*

## 🚀 Usage

To launch the application, run the main entry point from your terminal:

```bash
python main.py
```

### How to run an analysis:
1. Expand the **IPR Model** section and define your Reservoir Pressure ($P_r$), Bubble Point ($P_b$), and Test Points.
2. Expand the **Fluid Properties** section and define Specific Gravities and Water Cut.
3. Expand the **VLP — Wellbore** section and input tubular geometry, Tubing Head Pressure (THP), TVD, and temperatures.
4. Click the **▶ Run Analysis** button at the top right of the window.
5. View the Operating Point ($q^*$, $P_{wf}^*$) plotted on the chart and summarized in the status bar at the bottom.

## 🛠️ Building the Executable

To package the application into a standalone Windows `.exe` file, a PyInstaller script is provided. Simply run:

```bash
python build.py
```

The executable will be located in the `dist/` directory.