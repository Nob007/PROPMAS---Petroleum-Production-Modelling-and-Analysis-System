"""
PROPMAS - Petroleum Production Modelling & Analysis System
PyQt6 GUI implementing PRD v2.0
Engine modules (ipr, pvt, vlp, solver_other) are used unmodified.
"""

import sys
import os
import json
import copy
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import numpy as np

# ─── PyQt6 imports ───────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QLineEdit,
    QDialog, QDialogButtonBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QScrollArea, QFrame, QSplitter, QMessageBox, QFileDialog,
    QProgressBar, QCheckBox, QGroupBox, QSizePolicy, QMenu,
    QHeaderView, QAbstractItemView, QToolTip, QStatusBar,
    QStackedWidget
)
from PyQt6.QtCore import (
    Qt, QThread, QRunnable, QThreadPool, pyqtSignal, pyqtSlot,
    QObject, QTimer, QSize, QPoint
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QIcon, QPixmap, QPainter,
    QBrush, QPen, QAction, QCursor
)

# ─── Matplotlib embedded ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ─── Engine imports ───────────────────────────────────────────────────────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.ipr import composite_ipr, darcy_ipr, vogel_ipr, fetkovich_ipr, composite_fetkovich_ipr
from core.pvt import BlackOilPVT
from core import vlp as vlp_module
from core.vlp import HagedornBrown, Beggs_Brill
from core.solver_other import find_operating_points, NodalResult, StabilityType

from calibration.calibrate import VLPCalibrator, apply_calibration_factors
from gas_lift import gl as gl_mod
from gas_lift.gl import apply_gas_lift_design, compute_glpc, find_optimum, injection_feasibility_mask
from core import choke as choke_mod
from core.choke import (
    critical_flow_rate, critical_flow_pwh, is_critical_flow,
    sachdeva_choke, erosional_velocity_check, joule_thomson_estimate,
    choke_performance_curve, recommend_bean_size,
)
import csv
# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
NAVY    = "#0D2B55"
BLUE    = "#1565C0"
BLUE_H  = "#1976D2"
BLUE_L  = "#E3F0FB"
BLUE_M  = "#90CAF9"
WHITE   = "#FFFFFF"
OFF_W   = "#F7FAFD"
INK     = "#1A2840"
SLATE   = "#4A6080"
SUCCESS = "#2E7D32"
WARNING = "#E53935"
GOLD    = "#F9A825"
SENS_PALETTE = ["#1565C0", "#00897B", "#6A1B9A", "#E65100", "#37474F"]

# ─── Global QSS stylesheet ────────────────────────────────────────────────────
APP_QSS = f"""
QWidget {{
    font-family: "Segoe UI", "Inter", Arial, sans-serif;
    font-size: 13px;
    color: {INK};
    background-color: {WHITE};
}}
QMainWindow, QDialog {{
    background-color: {WHITE};
}}
/* Cards */
QFrame#card {{
    background-color: {WHITE};
    border: 1px solid {BLUE_M};
    border-radius: 10px;
}}
QFrame#card:hover {{
    border: 1.5px solid {BLUE};
}}
/* Primary button */
QPushButton#primary {{
    background-color: {BLUE};
    color: {WHITE};
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 14px;
}}
QPushButton#primary:hover {{
    background-color: {BLUE_H};
}}
QPushButton#primary:disabled {{
    background-color: #B0BEC5;
    color: #FFFFFF;
}}
/* Secondary button */
QPushButton#secondary {{
    background-color: {BLUE_L};
    color: {BLUE};
    border: 1.5px solid {BLUE_M};
    border-radius: 8px;
    padding: 9px 20px;
    font-weight: 600;
}}
QPushButton#secondary:hover {{
    background-color: {BLUE_M};
    color: {NAVY};
}}
QPushButton#secondary:disabled {{
    background-color: #ECEFF1;
    color: #90A4AE;
    border-color: #CFD8DC;
}}
/* Ghost/disabled card button */
QPushButton#ghost {{
    background-color: #F5F5F5;
    color: #90A4AE;
    border: 1.5px dashed #CFD8DC;
    border-radius: 8px;
    padding: 9px 20px;
    font-weight: 600;
}}
/* Input fields */
QLineEdit {{
    border: 1.5px solid {BLUE_M};
    border-radius: 6px;
    padding: 6px 10px;
    background: {WHITE};
    color: {INK};
    selection-background-color: {BLUE_L};
}}
QLineEdit:focus {{
    border-color: {BLUE};
    background: {BLUE_L};
}}
QLineEdit:disabled {{
    background: #F5F5F5;
    color: #90A4AE;
    border-color: #E0E0E0;
}}
QComboBox {{
    border: 1.5px solid {BLUE_M};
    border-radius: 6px;
    padding: 6px 10px;
    background: {WHITE};
    color: {INK};
}}
QComboBox:focus {{
    border-color: {BLUE};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    border: 1px solid {BLUE_M};
    border-radius: 4px;
    selection-background-color: {BLUE_L};
    selection-color: {NAVY};
    background: {WHITE};
}}
/* Labels */
QLabel#section {{
    font-size: 12px;
    font-weight: 600;
    color: {SLATE};
    letter-spacing: 1px;
}}
QLabel#kpi_value {{
    font-size: 22px;
    font-weight: 700;
    color: {NAVY};
}}
QLabel#kpi_label {{
    font-size: 11px;
    color: {SLATE};
    font-weight: 600;
    letter-spacing: 0.5px;
}}
/* Status chips */
QLabel#chip_success {{
    background-color: #E8F5E9;
    color: {SUCCESS};
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 700;
}}
QLabel#chip_warning {{
    background-color: #FFEBEE;
    color: {WARNING};
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 700;
}}
QLabel#chip_gold {{
    background-color: #FFFDE7;
    color: #F57F17;
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 700;
}}
QLabel#chip_blue {{
    background-color: {BLUE_L};
    color: {BLUE};
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 700;
}}
QLabel#chip_gray {{
    background-color: #F5F5F5;
    color: #90A4AE;
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
}}
/* Tables */
QTableWidget {{
    border: 1px solid {BLUE_M};
    border-radius: 6px;
    gridline-color: {BLUE_L};
    background: {WHITE};
    alternate-background-color: {OFF_W};
}}
QTableWidget::item:selected {{
    background-color: {BLUE_L};
    color: {NAVY};
}}
QHeaderView::section {{
    background-color: {NAVY};
    color: {WHITE};
    font-weight: 600;
    font-size: 12px;
    padding: 6px 8px;
    border: none;
}}
/* Tab widget */
QTabWidget::pane {{
    border: 1px solid {BLUE_M};
    border-radius: 6px;
    background: {WHITE};
}}
QTabBar::tab {{
    background: {OFF_W};
    color: {SLATE};
    border: 1px solid {BLUE_M};
    border-bottom: none;
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    background: {WHITE};
    color: {BLUE};
    border-color: {BLUE};
}}
/* Splitter */
QSplitter::handle {{
    background: {BLUE_L};
}}
/* Progress bar */
QProgressBar {{
    border: 1px solid {BLUE_M};
    border-radius: 4px;
    text-align: center;
    background: {OFF_W};
    height: 8px;
}}
QProgressBar::chunk {{
    background: {BLUE};
    border-radius: 4px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1.5px solid {BLUE_M};
    border-radius: 4px;
}}
QCheckBox::indicator:checked {{
    background-color: {BLUE};
    border-color: {BLUE};
}}
QScrollBar:vertical {{
    width: 8px;
    background: {OFF_W};
}}
QScrollBar::handle:vertical {{
    background: {BLUE_M};
    border-radius: 4px;
    min-height: 20px;
}}
"""

CHART_STYLE = {
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "axes.edgecolor": BLUE_M,
    "axes.labelcolor": SLATE,
    "axes.titlecolor": NAVY,
    "axes.grid": True,
    "grid.color": BLUE_L,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "font.family": "DejaVu Sans",
}
for k, v in CHART_STYLE.items():
    try:
        plt.rcParams[k] = v
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
#  APP STATE
# ─────────────────────────────────────────────────────────────────────────────
SESSION_FILE = os.path.join(os.path.expanduser("~"), ".ipm_session.json")

@dataclass
class AppState:
    well_id: Optional[str] = "Well-01"
    field_name: Optional[str] = "Field-X"
    # IPR
    ipr_model: str = "Composite"
    Pr: Optional[float] = None
    Pb: Optional[float] = None
    Qo_test: Optional[float] = None
    Pwf_test: Optional[float] = None
    # PVT
    sg_gas: float = 0.65
    sg_oil: float = 0.84
    oil_api: Optional[float] = None
    sg_water: float = 1.03
    wc: float = 0.0
    gor: Optional[float] = None
    T_pvt: float = 180.0
    P_min_pvt: float = 14.7
    P_max_pvt: float = 5000.0
    # VLP
    vlp_model: str = "Hagedorn-Brown"
    tubing_id: Optional[float] = None
    tubing_od: Optional[float] = None
    casing_id: Optional[float] = None
    roughness: float = 0.00015
    theta: float = 0.0
    depth: Optional[float] = None
    dz_step: float = 50.0
    thp: Optional[float] = None
    T_surface: Optional[float] = None
    T_bh: Optional[float] = None
    q_min: float = 50.0
    q_max_sweep: float = 5000.0
    q_step: float = 100.0
    # Calibration
    calib_holdup_factor: float = 1.0
    calib_friction_factor: float = 1.0
    calib_csv_path: Optional[str] = None   # display only
    # Gas Lift sweep params (PRD §11)
    gl_depth_min: Optional[float] = None
    gl_depth_max: Optional[float] = None
    gl_depth_step: Optional[float] = 500.0
    gl_inj_depth: Optional[float] = None   # injection depth for rate/GLR sweep
    gl_sg_gas: Optional[float] = None      # injected gas SG; defaults to sg_gas
    gl_q_min: float = 100.0                # Mscf/day
    gl_q_max: float = 3000.0              # Mscf/day
    gl_q_step: float = 100.0              # Mscf/day
    gl_glr_min: Optional[float] = None
    gl_glr_max: Optional[float] = None
    gl_glr_step: Optional[float] = 500.0
    gl_p_inj: Optional[float] = None       # available injection pressure, psia
    gl_q_available: Optional[float] = None # compressor ceiling, Mscf/day
    gl_econ_slope: float = 0.5             # economic threshold, STB/day per Mscf/day
    # Gas Lift applied design (sticky)
    gl_applied: bool = False
    gl_opt_depth: Optional[float] = None   # ft
    gl_opt_rate: Optional[float] = None    # Mscf/day
    # Choke params (PRD §11)
    choke_model: str = "gilbert"
    choke_p_down: Optional[float] = None
    choke_size_64: float = 32.0
    choke_sizes_list: str = "16,20,24,28,32,36,40,48,64"
    choke_target_q: Optional[float] = None
    choke_c_factor: float = 100.0
    # Panel completion flags
    ipr_saved: bool = False
    pvt_saved: bool = False
    vlp_saved: bool = False

    def save(self):
        try:
            with open(SESSION_FILE, "w") as f:
                json.dump(asdict(self), f, indent=2, default=str)
        except Exception as e:
            print(f"[AppState] Could not save session: {e}")

    def load(self):
        try:
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[AppState] Could not load session: {e}")

    @property
    def ipr_complete(self) -> bool:
        return all([self.Pr, self.Qo_test, self.Pwf_test]) and self.ipr_saved

    @property
    def pvt_complete(self) -> bool:
        return self.pvt_saved

    @property
    def vlp_complete(self) -> bool:
        return all([self.tubing_id, self.depth, self.thp, self.T_surface, self.T_bh]) and self.vlp_saved

    @property
    def nodal_ready(self) -> bool:
        return self.ipr_complete and self.pvt_complete and self.vlp_complete

# ─────────────────────────────────────────────────────────────────────────────
#  WORKER SIGNALS & BASE
# ─────────────────────────────────────────────────────────────────────────────
class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

class BaseWorker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

# ─────────────────────────────────────────────────────────────────────────────
#  ENGINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_pvt(state: AppState) -> BlackOilPVT:
    return BlackOilPVT(
        sg_gas=state.sg_gas,
        sg_oil=state.sg_oil,
        oil_api=state.oil_api,
        sg_water=state.sg_water,
        watercut=state.wc,
    )

def build_ipr(state: AppState):
    """Returns (ipr_obj, error_string). ipr_obj is None if error."""
    try:
        Pr = state.Pr; Pb = state.Pb or state.Pr * 0.7
        q_test = state.Qo_test; Pwf_test = state.Pwf_test
        if not all([Pr, q_test, Pwf_test]):
            return None, "Missing IPR inputs (Pr, Qo_test, Pwf_test)."
        if Pwf_test >= Pr:
            return None, "Pwf_test must be less than Pr."
        model = state.ipr_model
        if model == "Composite":
            return composite_ipr(Pr, Pb, q_test, Pwf_test), ""
        elif model == "Darcy":
            return darcy_ipr(Pr, Pb, q_test, Pwf_test), ""
        elif model == "Vogel":
            return vogel_ipr(Pr, Pb, q_test, Pwf_test), ""
        return None, f"Unknown IPR model: {model}"
    except Exception as e:
        return None, str(e)

def build_vlp(state: AppState, pvt_model: BlackOilPVT, fp_dict: dict):
    """Returns (vlp_obj, error_string)."""
    try:
        if state.vlp_model == "Hagedorn-Brown":
            obj = HagedornBrown(
                tubing_id=state.tubing_id,
                tubing_od=state.tubing_od or state.tubing_id * 1.2,
                casing_id=state.casing_id or state.tubing_id * 2.5,
                roughness=state.roughness,
                pvt_model=pvt_model,
                fluid_properties=fp_dict,
                watercut=state.wc,
                theta=state.theta,
            )
        elif state.vlp_model == "Beggs-Brill":
            obj = Beggs_Brill(
                tubing_id=state.tubing_id,
                tubing_od=state.tubing_od or state.tubing_id * 1.2,
                casing_id=state.casing_id or state.tubing_id * 2.5,
                roughness=state.roughness,
                pvt_model=pvt_model,
                fluid_properties=fp_dict,
                watercut=state.wc,
                theta=state.theta,
            ) # This was the missing parenthesis
        elif state.vlp_model == "Duns and Ros":
            obj = vlp_module.DunsRos(
                tubing_id=state.tubing_id,
                tubing_od=state.tubing_od or state.tubing_id * 1.2,
                casing_id=state.casing_id or state.tubing_id * 2.5,
                roughness=state.roughness,
                pvt_model=pvt_model,
                fluid_properties=fp_dict,
                watercut=state.wc,
                theta=state.theta,
            )
        
        # Step 1: Apply calibration factors if they are not default
        if state.calib_holdup_factor != 1.0 or state.calib_friction_factor != 1.0:
            obj = apply_calibration_factors(
                obj, state.calib_holdup_factor, state.calib_friction_factor
            )

        # Step 2: Apply gas lift design if active (after calibration, as PRD specifies)
        if getattr(state, "gl_applied", False):
            inj_depth = state.gl_opt_depth or (state.depth * 0.55 if state.depth else 4000.0)
            inj_rate  = state.gl_opt_rate or 500.0  # Mscf/day
            inj_sg    = state.gl_sg_gas or state.sg_gas
            obj = apply_gas_lift_design(obj, inj_depth, inj_rate, inj_sg)

        return obj, ""
    except Exception as e:
        return None, str(e)

def get_fp(state: AppState, pvt_model: BlackOilPVT) -> dict:
    P = state.thp or 500.0
    T = state.T_surface or 100.0
    Pb = state.Pb or 2000.0
    gor = state.gor or 500.0
    Rsb = pvt_model.calc_true_rsb(Pb, state.T_bh or 180.0)
    if Rsb <= 0:
        Rsb = gor
    return pvt_model.fluid_properties_dict(P, T, Rsb, gor, Pb)

# ─────────────────────────────────────────────────────────────────────────────
#  VLP CURVE WORKER
# ─────────────────────────────────────────────────────────────────────────────
class VLPWorker(BaseWorker):
    def __init__(self, state: AppState):
        super().__init__()
        self._state = copy.deepcopy(state)

    @pyqtSlot()
    def run(self):
        try:
            s = self._state
            pvt = build_pvt(s)
            fp = get_fp(s, pvt)
            vlp_obj, err = build_vlp(s, pvt, fp)
            if vlp_obj is None:
                self.signals.error.emit(f"VLP build error: {err}")
                return

            # Calculate Rsb to pass to the updated traverse method
            Pb = s.Pb or 2000.0
            gor = s.gor or 500.0
            T_bh = s.T_bh or 180.0
            Rsb = pvt.calc_true_rsb(Pb, T_bh)
            if Rsb <= 0: Rsb = gor

            rates = np.arange(s.q_min, s.q_max_sweep + s.q_step, s.q_step)
            pwfs = []
            total = len(rates)
            for i, q in enumerate(rates):
                try:
                    _, pressures, _ = vlp_obj.calculate_pressure_traverse(
                        Pth=s.thp, surface_temp=s.T_surface,
                        bottomhole_temp=s.T_bh, total_depth=s.depth,
                        step_size=s.dz_step, Ql=q
                    )
                    pwfs.append(float(pressures[-1]))
                except Exception:
                    pwfs.append(float("nan"))
                pct = int((i + 1) / total * 100)
                self.signals.progress.emit(pct, f"Rate {q:.0f} STB/d")

            self.signals.finished.emit({"rates": rates.tolist(), "pwfs": pwfs})
        except Exception as e:
            self.signals.error.emit(f"VLP computation failed:\n{traceback.format_exc()}")

# ─────────────────────────────────────────────────────────────────────────────
#  NODAL WORKER
# ─────────────────────────────────────────────────────────────────────────────
class NodalWorker(BaseWorker):
    def __init__(self, state: AppState):
        super().__init__()
        self._state = copy.deepcopy(state)

    @pyqtSlot()
    def run(self):
        try:
            s = self._state
            ipr_obj, err = build_ipr(s)
            if ipr_obj is None:
                self.signals.error.emit(f"IPR error: {err}")
                return

            pvt = build_pvt(s)
            fp = get_fp(s, pvt)
            vlp_obj, err = build_vlp(s, pvt, fp)
            if vlp_obj is None:
                self.signals.error.emit(f"VLP error: {err}")
                return

            # Calculate Rsb for VLP parameters
            Pb = s.Pb or 2000.0
            T_bh = s.T_bh or 180.0
            Rsb = pvt.calc_true_rsb(Pb, T_bh)
            if Rsb <= 0: Rsb = s.gor or 500.0
            vlp_params = {
                "Pth": s.thp, "surface_temp": s.T_surface, "Rsb": Rsb,
                "bottomhole_temp": s.T_bh, "depth": s.depth,
                "step_size": s.dz_step,
            }
            self.signals.progress.emit(10, "Running solver…")
            result: NodalResult = find_operating_points(
                ipr_model=ipr_obj, vlp_model=vlp_obj,
                vlp_params=vlp_params, pr=s.Pr,
                q_min=s.q_min, q_max=min(ipr_obj.q_max, s.q_max_sweep),
            )
            self.signals.progress.emit(60, "Computing curves…")

            # Build IPR curve
            q_ipr = np.linspace(0, ipr_obj.q_max, 60)
            p_ipr = [ipr_obj.calculate_Pwf(q) for q in q_ipr]

            # Build VLP curve (fast scan)
            rates_vlp = np.linspace(s.q_min, min(ipr_obj.q_max, s.q_max_sweep), 60)
            p_vlp = []
            for q in rates_vlp:
                try:
                    _, pressures, _ = vlp_obj.calculate_pressure_traverse(
                        Pth=s.thp, surface_temp=s.T_surface,
                        bottomhole_temp=s.T_bh, total_depth=s.depth,
                        step_size=s.dz_step, Ql=q
                    )
                    p_vlp.append(float(pressures[-1]))
                except Exception:
                    p_vlp.append(float("nan"))

            # Traverse at operating point
            traverse_data = None
            pvt_at_op = None
            if result.success and result.stable_point:
                q_star = result.stable_point.rate
                self.signals.progress.emit(80, "Computing traverse…")
                try:
                    depths, pressures, profiles = vlp_obj.calculate_pressure_traverse(
                        Pth=s.thp, surface_temp=s.T_surface,
                        bottomhole_temp=s.T_bh, total_depth=s.depth,
                        step_size=s.dz_step, Ql=q_star
                    )
                    traverse_data = {
                        "depths": depths, "pressures": pressures,
                        "profiles": profiles,
                    }
                    # PVT at operating point
                    Pb = s.Pb or 2000.0
                    gor = s.gor or 500.0
                    Rsb = pvt.calc_true_rsb(Pb, s.T_bh or 180.0)
                    if Rsb <= 0: Rsb = gor
                    pvt_at_op = pvt.fluid_properties_dict(
                        result.stable_point.pwf, s.T_bh, Rsb, gor, Pb
                    )
                except Exception as e:
                    print(f"[NodalWorker] traverse error: {e}")

            self.signals.progress.emit(100, "Done")
            self.signals.finished.emit({
                "result": result,
                "q_ipr": q_ipr.tolist(), "p_ipr": p_ipr,
                "rates_vlp": rates_vlp.tolist(), "p_vlp": p_vlp,
                "traverse": traverse_data,
                "pvt_at_op": pvt_at_op,
            })
        except Exception as e:
            self.signals.error.emit(f"Nodal solve failed:\n{traceback.format_exc()}")

# ─────────────────────────────────────────────────────────────────────────────
#  SENSITIVITY WORKER
# ─────────────────────────────────────────────────────────────────────────────
class SensitivityWorker(BaseWorker):
    def __init__(self, state: AppState, slots: list):
        super().__init__()
        self._state = copy.deepcopy(state)
        self._slots = slots  # list of {param, values}

    @pyqtSlot()
    def run(self):
        try:
            base = self._state
            results_by_slot = []

            for slot_idx, slot in enumerate(self._slots):
                param = slot["param"]
                values = slot["values"]
                slot_results = []
                n = len(values) * len(self._slots)

                for vi, val in enumerate(values):
                    s = copy.deepcopy(base)
                    setattr(s, param, val)

                    ipr_obj, err = build_ipr(s)
                    if ipr_obj is None:
                        slot_results.append({"val": val, "error": err})
                        continue

                    pvt = build_pvt(s)
                    fp = get_fp(s, pvt)
                    vlp_obj, err = build_vlp(s, pvt, fp)
                    if vlp_obj is None:
                        slot_results.append({"val": val, "error": err})
                        continue

                    # IPR curve
                    q_ipr = np.linspace(0, ipr_obj.q_max, 80).tolist()
                    p_ipr = [ipr_obj.calculate_Pwf(q) for q in q_ipr]

                    # VLP curve
                    rates_vlp = np.linspace(s.q_min, min(ipr_obj.q_max, s.q_max_sweep), 40)
                    p_vlp = []
                    for q in rates_vlp:
                        try:
                            _, pressures, _ = vlp_obj.calculate_pressure_traverse(
                                Pth=s.thp, surface_temp=s.T_surface,
                                bottomhole_temp=s.T_bh, total_depth=s.depth,
                                step_size=s.dz_step, Ql=q
                            )
                            p_vlp.append(float(pressures[-1]))
                        except Exception:
                            p_vlp.append(float("nan"))

                    # Operating point
                    Pb = s.Pb or 2000.0
                    T_bh = s.T_bh or 180.0
                    Rsb = pvt.calc_true_rsb(Pb, T_bh)
                    if Rsb <= 0: Rsb = s.gor or 500.0
                    vlp_params = {
                        "Pth": s.thp, "surface_temp": s.T_surface, "Rsb": Rsb,
                        "bottomhole_temp": s.T_bh, "depth": s.depth,
                        "step_size": s.dz_step,
                    }
                    try:
                        nr = find_operating_points(
                            ipr_model=ipr_obj, vlp_model=vlp_obj,
                            vlp_params=vlp_params, pr=s.Pr,
                            q_min=s.q_min, q_max=min(ipr_obj.q_max, s.q_max_sweep),
                        )
                        op_rate = nr.stable_point.rate if nr.success and nr.stable_point else None
                        op_pwf = nr.stable_point.pwf if nr.success and nr.stable_point else None
                    except Exception:
                        op_rate = None; op_pwf = None

                    slot_results.append({
                        "val": val, "q_ipr": q_ipr, "p_ipr": p_ipr,
                        "rates_vlp": rates_vlp.tolist(), "p_vlp": p_vlp,
                        "op_rate": op_rate, "op_pwf": op_pwf,
                    })
                    pct = int(((slot_idx * len(values) + vi + 1)) / (len(self._slots) * len(values)) * 100)
                    self.signals.progress.emit(pct, f"Slot {slot_idx+1}, val={val:.3g}")

                results_by_slot.append({"param": param, "results": slot_results})

            self.signals.finished.emit(results_by_slot)
        except Exception as e:
            self.signals.error.emit(f"Sensitivity failed:\n{traceback.format_exc()}")

# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION WORKER
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationWorker(BaseWorker):
    def __init__(self, state: AppState, measured_data: list):
        super().__init__()
        self._state = copy.deepcopy(state)
        self._measured_data = measured_data

    @pyqtSlot()
    def run(self):
        try:
            s = self._state
            self.signals.progress.emit(5, "Building models...")
            pvt = build_pvt(s)
            fp = get_fp(s, pvt)
            base_vlp, err = build_vlp(s, pvt, fp)
            if base_vlp is None:
                self.signals.error.emit(f"VLP build error: {err}")
                return

            # Define traverse params for the calibrator
            # Use a representative rate for calibration, e.g., the test rate
            Pb = s.Pb or 2000.0
            T_bh = s.T_bh or 180.0
            gor = s.gor or 500.0
            Rsb = pvt.calc_true_rsb(Pb, T_bh)
            if Rsb <= 0: Rsb = gor
            q_calib = s.Qo_test or 1000.0
            traverse_params = {
                "Pth": s.thp, "surface_temp": s.T_surface,
                "bottomhole_temp": s.T_bh, "total_depth": s.depth,
                "step_size": s.dz_step, "Ql": q_calib
            }

            # Unpack measured data
            measured_depths, measured_pressures = zip(*self._measured_data)

            # Run original traverse for comparison
            self.signals.progress.emit(20, "Running base traverse...")
            orig_depths, orig_pressures, _ = base_vlp.calculate_pressure_traverse(**traverse_params)

            # Run calibrator
            self.signals.progress.emit(40, "Optimizing factors...")
            calibrator = VLPCalibrator(base_vlp, traverse_params)
            result = calibrator.run(measured_depths, measured_pressures)

            calib_depths, calib_pressures = None, None
            if result.success:
                self.signals.progress.emit(80, "Running calibrated traverse...")
                calibrated_model = calibrator.calibrated_model
                calib_depths, calib_pressures, _ = calibrated_model.calculate_pressure_traverse(**traverse_params)

            self.signals.progress.emit(100, "Done")
            self.signals.finished.emit({
                "result": result,
                "measured_depths": measured_depths,
                "measured_pressures": measured_pressures,
                "original_traverse": (orig_depths, orig_pressures),
                "calibrated_traverse": (calib_depths, calib_pressures) if result.success else None,
            })

        except Exception as e:
            self.signals.error.emit(f"Calibration failed:\n{traceback.format_exc()}")




# ─────────────────────────────────────────────────────────────────────────────
#  COMMON WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
def make_label(text, style="section"):
    lbl = QLabel(text)
    lbl.setObjectName(style)
    return lbl

def make_chip(text, style="chip_gray"):
    chip = QLabel(text)
    chip.setObjectName(style)
    chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return chip

def make_input(placeholder="", value=None, unit="", enabled=True):
    """Returns (QLineEdit, optional unit label)."""
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    if value is not None:
        edit.setText(str(value))
    edit.setEnabled(enabled)
    if unit:
        unit_lbl = make_label(unit)
        unit_lbl.setObjectName("")  # reset style
        unit_lbl.setStyleSheet(f"color: {SLATE}; font-size: 11px; padding-left: 2px;")
        return edit, unit_lbl
    return edit, None

def make_row(label_text, widget, unit_widget=None, help_text=None):
    """Creates a form row: label | widget [unit]"""
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setFixedWidth(190)
    lbl.setStyleSheet(f"color: {SLATE}; font-size: 12px; font-weight: 600;")
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    if unit_widget:
        row.addWidget(unit_widget)
    return row

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None, figsize=(6, 4)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear_axes(self):
        self.fig.clear()
        return self.fig.add_subplot(111)

    def refresh(self):
        self.canvas.draw_idle()

class SectionHeader(QWidget):
    def __init__(self, title, subtitle="", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {NAVY};")
        layout.addWidget(title_lbl)
        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setStyleSheet(f"font-size: 12px; color: {SLATE};")
            layout.addWidget(sub_lbl)

class InheritedChip(QWidget):
    edit_requested = pyqtSignal(str)  # panel name

    def __init__(self, label, value_str, source_panel, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        icon = QLabel("🔗")
        lbl = QLabel(f"{label}: {value_str}")
        lbl.setStyleSheet(f"color: {SLATE}; font-size: 11px;")
        edit_btn = QPushButton(f"edit in {source_panel}")
        edit_btn.setObjectName("secondary")
        edit_btn.setFixedHeight(22)
        edit_btn.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        edit_btn.clicked.connect(lambda: self.edit_requested.emit(source_panel))
        layout.addWidget(icon)
        layout.addWidget(lbl)
        layout.addStretch()
        layout.addWidget(edit_btn)

# ─────────────────────────────────────────────────────────────────────────────
#  IPR PANEL
# ─────────────────────────────────────────────────────────────────────────────
class IPRPanel(QDialog):
    applied = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("IPR Data — Inflow Performance")
        self.setMinimumSize(1000, 680)
        self._debounce = QTimer(); self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._recompute)
        self._setup_ui()
        self._load_from_state()

    def _setup_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(20)

        # Left: inputs
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(380)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(14)

        left_layout.addWidget(SectionHeader("IPR Data", "Inflow Performance Relationship"))

        # Model
        model_row = QHBoxLayout()
        model_lbl = QLabel("IPR Model")
        model_lbl.setFixedWidth(190)
        model_lbl.setStyleSheet(f"color: {SLATE}; font-size: 12px; font-weight: 600;")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Composite", "Vogel", "Darcy", "Fetkovich", "Composite Fetkovich"])
        self.model_combo.currentTextChanged.connect(self._on_model_change)
        model_row.addWidget(model_lbl); model_row.addWidget(self.model_combo, 1)
        left_layout.addLayout(model_row)

        # Pr
        self.pr_edit, pr_unit = make_input("e.g. 2500", unit="psia")
        self.pr_edit.textChanged.connect(self._schedule_recompute)
        left_layout.addLayout(make_row("Reservoir Pressure (Pr)", self.pr_edit, pr_unit))

        # Pb
        self.pb_edit, pb_unit = make_input("e.g. 1800", unit="psia")
        self.pb_edit.textChanged.connect(self._schedule_recompute)
        self.pb_row_widget = QWidget()
        pb_row_l = make_row("Bubble Point Pressure (Pb)", self.pb_edit, pb_unit)
        self.pb_row_widget.setLayout(pb_row_l)
        left_layout.addWidget(self.pb_row_widget)

        # Qo_test
        self.q_test_edit, qt_unit = make_input("e.g. 800", unit="STB/day")
        self.q_test_edit.textChanged.connect(self._schedule_recompute)
        left_layout.addLayout(make_row("Test Rate (Qo_test)", self.q_test_edit, qt_unit))

        # Pwf_test
        self.pwf_test_edit, pwft_unit = make_input("e.g. 1200", unit="psia")
        self.pwf_test_edit.textChanged.connect(self._schedule_recompute)
        left_layout.addLayout(make_row("Test Pwf", self.pwf_test_edit, pwft_unit))

        # Fetkovich-specific inputs
        self.q_test2_edit, qt2_unit = make_input("e.g. 600", unit="STB/day")
        self.q_test2_edit.textChanged.connect(self._schedule_recompute)
        self.q_test2_row_widget = QWidget(); self.q_test2_row_widget.setLayout(make_row("Test Rate 2", self.q_test2_edit, qt2_unit))
        left_layout.addWidget(self.q_test2_row_widget)

        self.pwf_test2_edit, pwft2_unit = make_input("e.g. 1000", unit="psia")
        self.pwf_test2_edit.textChanged.connect(self._schedule_recompute)
        self.pwf_test2_row_widget = QWidget(); self.pwf_test2_row_widget.setLayout(make_row("Test Pwf 2", self.pwf_test2_edit, pwft2_unit))
        left_layout.addWidget(self.pwf_test2_row_widget)

        self.n_edit, n_unit = make_input("e.g. 0.8", unit="")
        self.n_row_widget = QWidget(); self.n_row_widget.setLayout(make_row("Fetkovich Exponent (n)", self.n_edit, n_unit))
        left_layout.addWidget(self.n_row_widget)

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {BLUE_M};"); left_layout.addWidget(sep)

        # Outputs
        left_layout.addWidget(make_label("COMPUTED RESULTS"))
        self.J_lbl = QLabel("—"); self.J_lbl.setObjectName("kpi_value")
        self.qbp_lbl = QLabel("—"); self.qbp_lbl.setObjectName("kpi_value")
        self.qmax_lbl = QLabel("—"); self.qmax_lbl.setObjectName("kpi_value")

        for title, lbl, unit in [
            ("Productivity Index J", self.J_lbl, "STB/day/psi"),
            ("Flow Rate @ Bubble Point", self.qbp_lbl, "STB/day"),
            ("Absolute Open Flow (AOF)", self.qmax_lbl, "STB/day"),
        ]:
            kpi_box = QFrame(); kpi_box.setObjectName("card")
            kpi_lay = QVBoxLayout(kpi_box); kpi_lay.setContentsMargins(12, 10, 12, 10)
            klbl = QLabel(title); klbl.setObjectName("kpi_label")
            kpi_lay.addWidget(klbl); kpi_lay.addWidget(lbl)
            u_lbl = QLabel(unit); u_lbl.setStyleSheet(f"font-size: 11px; color: {SLATE};")
            kpi_lay.addWidget(u_lbl)
            left_layout.addWidget(kpi_box)

        self.validation_lbl = QLabel("")
        self.validation_lbl.setStyleSheet(f"color: {WARNING}; font-size: 11px;")
        self.validation_lbl.setWordWrap(True)
        left_layout.addWidget(self.validation_lbl)
        left_layout.addStretch()

        # Apply
        apply_btn = QPushButton("💾  Apply & Save")
        apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(self._apply)
        left_layout.addWidget(apply_btn)

        left_scroll.setWidget(left_widget)
        main.addWidget(left_scroll)

        # Right: chart
        right_layout = QVBoxLayout()
        right_layout.addWidget(make_label("IPR CURVE"))
        self.chart = MatplotlibWidget(figsize=(6, 5))
        right_layout.addWidget(self.chart, 1)
        main.addLayout(right_layout, 1)

    def _on_model_change(self, model):
        is_fetkovich = model == "Fetkovich"
        is_comp_fetkovich = model == "Composite Fetkovich"

        self.pb_row_widget.setVisible(model != "Darcy")
        self.q_test2_row_widget.setVisible(is_fetkovich)
        self.pwf_test2_row_widget.setVisible(is_fetkovich)
        self.n_row_widget.setVisible(is_comp_fetkovich)
        self._schedule_recompute()

    def _schedule_recompute(self):
        self._debounce.start(250)

    def _load_from_state(self):
        s = self.state
        models = ["Composite", "Vogel", "Darcy", "Fetkovich", "Composite Fetkovich"]
        idx = models.index(s.ipr_model) if s.ipr_model in models else 0
        self.model_combo.setCurrentIndex(idx)
        for edit, val in [
            (self.pr_edit, s.Pr), (self.pb_edit, s.Pb),
            (self.q_test_edit, s.Qo_test), (self.pwf_test_edit, s.Pwf_test),
        ]:
            if val is not None:
                edit.setText(str(val))

    def _get_inputs(self):
        def safe(edit):
            try: return float(edit.text())
            except: return None
        return {
            "model": self.model_combo.currentText(),
            "Pr": safe(self.pr_edit), "Pb": safe(self.pb_edit),
            "q_test": safe(self.q_test_edit), "Pwf_test": safe(self.pwf_test_edit),
            "q_test2": safe(self.q_test2_edit), "Pwf_test2": safe(self.pwf_test2_edit),
            "n": safe(self.n_edit),
        }

    def _recompute(self):
        inputs = self._get_inputs()
        model, Pr, Pb, q_test, Pwf_test = inputs["model"], inputs["Pr"], inputs["Pb"], inputs["q_test"], inputs["Pwf_test"]
        Pb = Pb or (Pr * 0.7 if Pr else None)
        if not all([Pr, q_test, Pwf_test]):
            self.validation_lbl.setText("Fill Pr, Test Rate, and Test Pwf to compute.")
            return
        if Pwf_test >= Pr:
            self.validation_lbl.setText("⚠ Pwf_test must be < Pr.")
            return
        self.validation_lbl.setText("")

        try:
            if model == "Composite":
                ipr = composite_ipr(Pr, Pb, q_test, Pwf_test)
            elif model == "Darcy":
                ipr = darcy_ipr(Pr, Pb, q_test, Pwf_test)
            elif model == "Vogel":
                ipr = vogel_ipr(Pr, Pb, q_test, Pwf_test)
            elif model == "Fetkovich":
                q2, p2 = inputs["q_test2"], inputs["Pwf_test2"]
                if not all([q2, p2]):
                    self.validation_lbl.setText("Fetkovich requires a second test point.")
                    return
                ipr = fetkovich_ipr(Pr, Pb, q_test, Pwf_test, q2, p2)
            elif model == "Composite Fetkovich":
                # Calculate J from the linear part of a composite curve
                temp_comp_ipr = composite_ipr(Pr, Pb, q_test, Pwf_test)
                J = temp_comp_ipr.J
                n = inputs.get("n") or 1.0
                ipr = composite_fetkovich_ipr(Pr, Pb, q_test, Pwf_test, J=J, n=n)
            else:
                raise ValueError(f"Unknown IPR model: {model}")

            self.J_lbl.setText(f"{ipr.J:.4f}")
            self.qmax_lbl.setText(f"{ipr.q_max:.1f}")
            if hasattr(ipr, "q_bp") and ipr.q_bp is not None:
                self.qbp_lbl.setText(f"{ipr.q_bp:.1f}")
            else:
                self.qbp_lbl.setText("N/A")

            self._plot_ipr(ipr, model, Pb, q_test, Pwf_test)
        except Exception as e:
            self.validation_lbl.setText(f"Calculation error: {e}")

    def _plot_ipr(self, ipr, model, Pb, q_test, Pwf_test):
        ax = self.chart.clear_axes()
        q_range = np.linspace(0, ipr.q_max, 200)
        p_range = [ipr.calculate_Pwf(q) for q in q_range]
        ax.plot(q_range, p_range, color=BLUE, linewidth=2.5, label="IPR Curve")
        ax.scatter([q_test], [Pwf_test], color=WARNING, zorder=6, s=80, label="Test Point", marker="D")

        if model == "Composite" and Pb and hasattr(ipr, "q_bp"):
            ax.axhline(Pb, color=GOLD, linestyle="--", alpha=0.7, linewidth=1.5)
            ax.scatter([ipr.q_bp], [Pb], color=GOLD, zorder=6, s=80, marker="x", linewidths=2.5, label=f"Bubble Point (Pb = {Pb:.0f} psia)")
            ax.annotate("← Darcy  |  Vogel →", xy=(ipr.q_bp, Pb), xytext=(ipr.q_bp + ipr.q_max * 0.05, Pb + 50),
                        fontsize=8, color=SLATE)

        ax.set_xlabel("Liquid Rate, q (STB/day)")
        ax.set_ylabel("Flowing BHP, Pwf (psia)")
        ax.set_title(f"{model} IPR", fontsize=12, fontweight="bold", color=NAVY)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, framealpha=0.9)
        self.chart.refresh()

    def _apply(self):
        inputs = self._get_inputs()
        Pr, q_test, Pwf_test = inputs["Pr"], inputs["q_test"], inputs["Pwf_test"]
        if not all([Pr, q_test, Pwf_test]):
            QMessageBox.warning(self, "Incomplete", "Pr, Test Rate, and Test Pwf are required.")
            return
        if Pwf_test >= Pr:
            QMessageBox.warning(self, "Validation Error", "Pwf_test must be less than Pr.")
            return
        self.state.ipr_model = inputs["model"]
        self.state.Pr = Pr
        self.state.Pb = inputs["Pb"]
        self.state.Qo_test = q_test
        self.state.Pwf_test = Pwf_test
        self.state.ipr_saved = True
        self.state.save()
        self.applied.emit()
        self.accept()

# ─────────────────────────────────────────────────────────────────────────────
#  PVT PANEL
# ─────────────────────────────────────────────────────────────────────────────
class PVTPanel(QDialog):
    applied = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("PVT Data — Fluid Properties")
        self.setMinimumSize(1100, 700)
        self._debounce = QTimer(); self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._recompute)
        self._setup_ui()
        self._load_from_state()

    def _setup_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(20)

        # Left
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(360); left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_w = QWidget(); left_l = QVBoxLayout(left_w); left_l.setSpacing(10)
        left_l.addWidget(SectionHeader("PVT Data", "Black-Oil Fluid Properties"))

        # Fluid Composition
        left_l.addWidget(make_label("FLUID COMPOSITION"))

        self.sg_gas_edit, u1 = make_input("0.65", unit="air=1")
        self.sg_gas_edit.textChanged.connect(self._schedule)
        left_l.addLayout(make_row("Gas Specific Gravity", self.sg_gas_edit, u1))

        # Oil: toggle API vs SG
        oil_row = QHBoxLayout()
        self.oil_api_rb = QCheckBox("Use API gravity")
        self.oil_api_rb.toggled.connect(self._toggle_oil_input)
        self.sg_oil_edit, u2 = make_input("0.85", unit="water=1")
        self.sg_oil_edit.textChanged.connect(self._schedule)
        self.api_edit, u3 = make_input("°API", unit="°API")
        self.api_edit.textChanged.connect(self._schedule)
        self.api_edit.setVisible(False); u3.setVisible(False) if u3 else None
        self.api_unit_lbl = u3
        oil_row.addWidget(self.oil_api_rb)
        left_l.addLayout(oil_row)
        left_l.addLayout(make_row("Oil Specific Gravity", self.sg_oil_edit, u2))
        self.api_row_lbl = make_row("API Gravity", self.api_edit, self.api_unit_lbl)
        api_row_w = QWidget(); api_row_w.setLayout(self.api_row_lbl)
        api_row_w.setVisible(False); self.api_row_w = api_row_w
        left_l.addWidget(api_row_w)

        self.sg_water_edit, u4 = make_input("1.07", unit="water=1")
        self.sg_water_edit.textChanged.connect(self._schedule)
        left_l.addLayout(make_row("Water Specific Gravity", self.sg_water_edit, u4))

        self.wc_edit, u5 = make_input("0.33", unit="fraction 0–1")
        self.wc_edit.textChanged.connect(self._schedule)
        left_l.addLayout(make_row("Watercut", self.wc_edit, u5))

        self.gor_edit, u6 = make_input("e.g. 500", unit="scf/STB")
        self.gor_edit.textChanged.connect(self._schedule)
        left_l.addLayout(make_row("Producing GOR", self.gor_edit, u6))

        # Pb — inherited from IPR if available
        self.pb_chip_widget = QWidget()
        pb_chip_l = QHBoxLayout(self.pb_chip_widget); pb_chip_l.setContentsMargins(0, 0, 0, 0)
        self.pb_inherited_lbl = QLabel("")
        pb_chip_l.addWidget(self.pb_inherited_lbl)
        left_l.addWidget(self.pb_chip_widget)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet(f"color: {BLUE_M};")
        left_l.addWidget(sep)
        left_l.addWidget(make_label("EVALUATION NODE"))

        self.T_edit, u7 = make_input("180", unit="°F")
        self.T_edit.textChanged.connect(self._schedule)
        left_l.addLayout(make_row("Temperature", self.T_edit, u7))

        self.P_min_edit, u8 = make_input("14.7", unit="psia")
        self.P_min_edit.textChanged.connect(self._schedule)
        left_l.addLayout(make_row("Pressure Min", self.P_min_edit, u8))

        self.P_max_edit, u9 = make_input("5000", unit="psia")
        self.P_max_edit.textChanged.connect(self._schedule)
        left_l.addLayout(make_row("Pressure Max", self.P_max_edit, u9))

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine); sep2.setStyleSheet(f"color: {BLUE_M};")
        left_l.addWidget(sep2)

        # Phase diagnosis badge
        self.phase_badge = make_chip("—", "chip_blue")
        self.phase_badge.setFixedHeight(28)
        phase_row = QHBoxLayout()
        phase_row.addWidget(make_label("PHASE DIAGNOSIS"))
        phase_row.addStretch()
        phase_row.addWidget(self.phase_badge)
        left_l.addLayout(phase_row)
        left_l.addStretch()

        apply_btn = QPushButton("💾  Apply & Save")
        apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(self._apply)
        left_l.addWidget(apply_btn)

        left_scroll.setWidget(left_w)
        main.addWidget(left_scroll)

        # Right: chart + curve toggles
        right_l = QVBoxLayout()
        right_l.addWidget(make_label("PVT CURVES — select properties to display:"))

        toggles_frame = QFrame(); tgl = QHBoxLayout(toggles_frame); tgl.setSpacing(6)
        self.curve_checks = {}
        for name in ["Bo", "Bg", "Rs", "Z", "μ_oil", "μ_gas", "ρ_oil"]:
            cb = QCheckBox(name); cb.setChecked(name in ["Bo", "Rs"])
            cb.toggled.connect(self._schedule)
            tgl.addWidget(cb)
            self.curve_checks[name] = cb
        tgl.addStretch()
        right_l.addWidget(toggles_frame)

        self.chart = MatplotlibWidget(figsize=(6.5, 5))
        right_l.addWidget(self.chart, 1)
        main.addLayout(right_l, 1)

    def _toggle_oil_input(self, use_api):
        self.sg_oil_edit.setVisible(not use_api)
        self.api_row_w.setVisible(use_api)

    def _schedule(self):
        self._debounce.start(300)

    def _load_from_state(self):
        s = self.state
        for edit, val in [
            (self.sg_gas_edit, s.sg_gas), (self.sg_oil_edit, s.sg_oil),
            (self.sg_water_edit, s.sg_water), (self.wc_edit, s.wc),
            (self.gor_edit, s.gor), (self.T_edit, s.T_pvt),
            (self.P_min_edit, s.P_min_pvt), (self.P_max_edit, s.P_max_pvt),
        ]:
            if val is not None:
                edit.setText(str(val))
        if s.oil_api:
            self.oil_api_rb.setChecked(True)
            self.api_edit.setText(str(s.oil_api))

        if s.Pb:
            txt = f"🔗 Bubble Point Pb = {s.Pb:.0f} psia (from IPR panel)"
            self.pb_inherited_lbl.setText(txt)
            self.pb_inherited_lbl.setStyleSheet(f"color:{SLATE}; font-size:11px;")

    def _get_pvt(self):
        def safe(e, default=0.0):
            try: return float(e.text())
            except: return default

        sg_gas = safe(self.sg_gas_edit, 0.65)
        sg_oil = safe(self.sg_oil_edit, 0.84)
        oil_api = safe(self.api_edit, 0) if self.oil_api_rb.isChecked() else None
        sg_water = safe(self.sg_water_edit, 1.03)
        wc = max(0.0, min(0.9999, safe(self.wc_edit, 0.0)))
        gor = safe(self.gor_edit, 500.0)
        T = safe(self.T_edit, 180.0)
        P_min = safe(self.P_min_edit, 14.7)
        P_max = safe(self.P_max_edit, 5000.0)
        Pb = self.state.Pb or 2000.0
        return sg_gas, sg_oil, oil_api, sg_water, wc, gor, T, P_min, P_max, Pb

    def _recompute(self):
        sg_gas, sg_oil, oil_api, sg_water, wc, gor, T, P_min, P_max, Pb = self._get_pvt()
        try:
            pvt = BlackOilPVT(sg_gas=sg_gas, sg_oil=sg_oil, oil_api=oil_api,
                              sg_water=sg_water, watercut=wc)
            P_range = np.linspace(P_min, P_max, 100)
            Rsb = pvt.calc_true_rsb(Pb, T)
            if Rsb <= 0: Rsb = gor

            curves = {}
            for P in P_range:
                Z = pvt.calculate_dak_z_factor(P, T, sg_gas)
                Rs = pvt.calc_rs(P, T, Pb, Rsb)
                Bo = pvt.calc_bo(P, T, Rs, Pb)
                Bg = pvt.calc_bg(P, T, Z)
                mu_o = pvt.calc_viscosity_oil(P, T, Rs, Pb)
                mu_g = pvt.calc_viscosity_gas(P, T, Z)
                rho_o = pvt.calc_density_oil(Rs, Bo)
                for k, v in [("Bo", Bo), ("Bg", Bg * 1000), ("Rs", Rs), ("Z", Z),
                              ("μ_oil", mu_o), ("μ_gas", mu_g * 100), ("ρ_oil", rho_o)]:
                    curves.setdefault(k, []).append(v)

            # Phase diagnosis
            cur_P = (P_min + P_max) / 2
            phase = "Undersaturated (above Pb)" if cur_P >= Pb else "Two-Phase (below Pb)"
            badge_style = "chip_blue" if cur_P >= Pb else "chip_gold"
            self.phase_badge.setText(phase)
            self.phase_badge.setObjectName(badge_style)
            self.phase_badge.setStyle(self.phase_badge.style())

            # Plot selected curves
            ax = self.chart.clear_axes()
            colors_map = {"Bo": BLUE, "Bg": "#00897B", "Rs": NAVY, "Z": "#6A1B9A",
                          "μ_oil": "#E65100", "μ_gas": "#37474F", "ρ_oil": "#558B2F"}
            units_map = {"Bo": "bbl/STB", "Bg": "ft³/scf ×10³", "Rs": "scf/STB",
                         "Z": "dimensionless", "μ_oil": "cp", "μ_gas": "cp ×10²", "ρ_oil": "lbm/ft³"}
            any_plotted = False
            for name, cb in self.curve_checks.items():
                if cb.isChecked() and name in curves:
                    ax.plot(P_range, curves[name], color=colors_map[name], linewidth=2,
                            label=f"{name} [{units_map[name]}]")
                    any_plotted = True

            ax.axvline(Pb, color=GOLD, linestyle="--", alpha=0.7, linewidth=1.5, label=f"Pb={Pb:.0f} psia")
            if not any_plotted:
                ax.text(0.5, 0.5, "Select properties above to plot", ha="center", va="center",
                        transform=ax.transAxes, color=SLATE, fontsize=12)
            ax.set_xlabel("Pressure (psia)"); ax.set_ylabel("Property Value")
            ax.set_title("PVT Properties vs. Pressure", fontsize=12, fontweight="bold", color=NAVY)
            ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
            self.chart.refresh()

        except Exception as e:
            print(f"[PVTPanel] recompute error: {traceback.format_exc()}")

    def _apply(self):
        sg_gas, sg_oil, oil_api, sg_water, wc, gor, T, P_min, P_max, Pb = self._get_pvt()
        self.state.sg_gas = sg_gas; self.state.sg_oil = sg_oil
        self.state.oil_api = oil_api; self.state.sg_water = sg_water
        self.state.wc = wc; self.state.gor = gor
        self.state.T_pvt = T; self.state.P_min_pvt = P_min; self.state.P_max_pvt = P_max
        self.state.pvt_saved = True
        self.state.save()
        self.applied.emit()
        self.accept()

# ─────────────────────────────────────────────────────────────────────────────
#  VLP PANEL
# ─────────────────────────────────────────────────────────────────────────────
class VLPPanel(QDialog):
    applied = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("VLP Data — Wellbore Lift Performance")
        self.setMinimumSize(1050, 700)
        self._vlp_data = None
        self._setup_ui()
        self._load_from_state()

    def _setup_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(20)

        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(380); left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_w = QWidget(); left_l = QVBoxLayout(left_w); left_l.setSpacing(10)

        left_l.addWidget(SectionHeader("VLP Data", "Wellbore Geometry & Lift Curve"))

        # Correlation
        corr_row = QHBoxLayout()
        corr_lbl = QLabel("VLP Correlation"); corr_lbl.setFixedWidth(190)
        corr_lbl.setStyleSheet(f"color:{SLATE}; font-size:12px; font-weight:600;")
        self.corr_combo = QComboBox()
        self.corr_combo.addItems(["Hagedorn-Brown", "Beggs-Brill", "Duns and Ros"])
        corr_row.addWidget(corr_lbl); corr_row.addWidget(self.corr_combo, 1)
        left_l.addLayout(corr_row)

        left_l.addWidget(make_label("WELLBORE GEOMETRY"))

        fields = [
            ("Tubing ID", "tubing_id_edit", "in", "2.441"),
            ("Tubing OD", "tubing_od_edit", "in", "2.875"),
            ("Casing ID", "casing_id_edit", "in", "5.500"),
            ("Roughness", "roughness_edit", "in", "0.0006"),
            ("Deviation Angle", "theta_edit", "°", "0"),
            ("Total Depth", "depth_edit", "ft", ""),
            ("Depth Step", "dz_step_edit", "ft", "50"),
        ]
        for lbl_text, attr, unit, default in fields:
            edit, unit_lbl = make_input(default, unit=unit)
            setattr(self, attr, edit)
            left_l.addLayout(make_row(lbl_text, edit, unit_lbl))

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet(f"color:{BLUE_M};")
        left_l.addWidget(sep)
        left_l.addWidget(make_label("SURFACE / THERMAL"))

        thermal_fields = [
            ("Wellhead Pressure (THP)", "thp_edit", "psia", ""),
            ("Surface Temperature", "T_surf_edit", "°F", ""),
            ("Bottomhole Temperature", "T_bh_edit", "°F", ""),
        ]
        for lbl_text, attr, unit, default in thermal_fields:
            edit, unit_lbl = make_input(default, unit=unit)
            setattr(self, attr, edit)
            left_l.addLayout(make_row(lbl_text, edit, unit_lbl))

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine); sep2.setStyleSheet(f"color:{BLUE_M};")
        left_l.addWidget(sep2)
        left_l.addWidget(make_label("RATE SWEEP"))

        sweep_fields = [
            ("Min Rate", "q_min_edit", "STB/day", "50"),
            ("Max Rate", "q_max_edit", "STB/day", "5000"),
            ("Rate Step", "q_step_edit", "STB/day", "100"),
        ]
        for lbl_text, attr, unit, default in sweep_fields:
            edit, unit_lbl = make_input(default, unit=unit)
            setattr(self, attr, edit)
            left_l.addLayout(make_row(lbl_text, edit, unit_lbl))

        self.vlp_error_lbl = QLabel("")
        self.vlp_error_lbl.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        self.vlp_error_lbl.setWordWrap(True)
        left_l.addWidget(self.vlp_error_lbl)
        left_l.addStretch()

        compute_btn = QPushButton("▶  Compute VLP Curve")
        compute_btn.setObjectName("secondary")
        compute_btn.clicked.connect(self._run_vlp)
        left_l.addWidget(compute_btn)

        self.vlp_progress = QProgressBar()
        self.vlp_progress.setVisible(False)
        left_l.addWidget(self.vlp_progress)

        apply_btn = QPushButton("💾  Apply & Save")
        apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(self._apply)
        left_l.addWidget(apply_btn)

        left_scroll.setWidget(left_w)
        main.addWidget(left_scroll)

        # Right: VLP chart
        right_l = QVBoxLayout()
        right_l.addWidget(make_label("VLP CURVE"))

        # PRD §6.4 — Sticky-modifier note (calibration / gas lift active)
        self.vlp_modifier_chip = QLabel("")
        self.vlp_modifier_chip.setStyleSheet(
            f"color:{BLUE}; font-size:11px; font-weight:600; "
            f"background:{BLUE_L}; border-radius:8px; padding:2px 10px;"
        )
        self.vlp_modifier_chip.setVisible(False)
        right_l.addWidget(self.vlp_modifier_chip)

        self.chart = MatplotlibWidget(figsize=(6, 5))
        right_l.addWidget(self.chart, 1)
        main.addLayout(right_l, 1)

    def _load_from_state(self):
        s = self.state
        models = ["Hagedorn-Brown", "Beggs-Brill", "Duns and Ros"]
        if s.vlp_model in models:
            self.corr_combo.setCurrentText(s.vlp_model)

        map_fields = [
            (self.tubing_id_edit, (s.tubing_id * 12) if s.tubing_id else None), (self.tubing_od_edit, (s.tubing_od * 12) if s.tubing_od else None),
            (self.casing_id_edit, (s.casing_id * 12) if s.casing_id else None), (self.roughness_edit, (s.roughness * 12) if s.roughness else None),
            (self.theta_edit, s.theta), (self.depth_edit, s.depth),
            (self.dz_step_edit, s.dz_step), (self.thp_edit, s.thp),
            (self.T_surf_edit, s.T_surface), (self.T_bh_edit, s.T_bh),
            (self.q_min_edit, s.q_min), (self.q_max_edit, s.q_max_sweep),
            (self.q_step_edit, s.q_step),
        ]
        for edit, val in map_fields:
            if val is not None:
                edit.setText(str(val))

        # PRD §6.4 — update sticky-modifier chip
        self._update_modifier_chip()

    def _update_modifier_chip(self):
        """Show an inline note whenever calibration or gas lift is silently shaping VLP curves."""
        s = self.state
        parts = []
        if s.calib_holdup_factor != 1.0 or s.calib_friction_factor != 1.0:
            parts.append("Calibrated")
        if getattr(s, "gl_applied", False):
            parts.append("Gas-lift applied")
        if parts:
            self.vlp_modifier_chip.setText("⚡ " + " · ".join(parts))
            self.vlp_modifier_chip.setVisible(True)
        else:
            self.vlp_modifier_chip.setVisible(False)

    def _collect(self):
        def safe(e, default=None):
            try: return float(e.text())
            except: return default
        return {
            "vlp_model": self.corr_combo.currentText(),
            "tubing_id": safe(self.tubing_id_edit) / 12.0 if safe(self.tubing_id_edit) is not None else None,
            "tubing_od": safe(self.tubing_od_edit) / 12.0 if safe(self.tubing_od_edit) is not None else None,
            "casing_id": safe(self.casing_id_edit) / 12.0 if safe(self.casing_id_edit) is not None else None,
            "roughness": safe(self.roughness_edit) / 12.0 if safe(self.roughness_edit) is not None else None,
            "theta": safe(self.theta_edit, 0.0),
            "depth": safe(self.depth_edit),
            "dz_step": safe(self.dz_step_edit, 50.0),
            "thp": safe(self.thp_edit),
            "T_surface": safe(self.T_surf_edit),
            "T_bh": safe(self.T_bh_edit),
            "q_min": safe(self.q_min_edit, 50.0),
            "q_max_sweep": safe(self.q_max_edit, 5000.0),
            "q_step": safe(self.q_step_edit, 100.0),
        }

    def _run_vlp(self):
        vals = self._collect()
        req = ["tubing_id", "depth", "thp", "T_surface", "T_bh"]
        missing = [k for k in req if not vals[k]]
        if missing:
            self.vlp_error_lbl.setText(f"⚠ Required: {', '.join(missing)}")
            return
        self.vlp_error_lbl.setText("")

        # Build a temp state
        temp = copy.deepcopy(self.state)
        for k, v in vals.items():
            setattr(temp, k, v)

        self.vlp_progress.setVisible(True)
        self.vlp_progress.setValue(0)
        worker = VLPWorker(temp)
        worker.signals.finished.connect(self._on_vlp_done)
        worker.signals.error.connect(self._on_vlp_error)
        worker.signals.progress.connect(lambda p, _: self.vlp_progress.setValue(p))
        QThreadPool.globalInstance().start(worker)

    def _on_vlp_done(self, data):
        self.vlp_progress.setVisible(False)
        self._vlp_data = data
        rates = data["rates"]; pwfs = data["pwfs"]
        ax = self.chart.clear_axes()
        valid = [(r, p) for r, p in zip(rates, pwfs) if not (isinstance(p, float) and (p != p))]
        if valid:
            r_v, p_v = zip(*valid)
            ax.plot(r_v, p_v, color=BLUE, linewidth=2.5, label="VLP Curve")
        ax.set_xlabel("Liquid Rate, q (STB/day)")
        ax.set_ylabel("Flowing BHP, Pwf (psia)")
        ax.set_title("VLP — Vertical Lift Performance", fontsize=12, fontweight="bold", color=NAVY)
        if valid:
            ax.legend()
        self.chart.refresh()

    def _on_vlp_error(self, msg):
        self.vlp_progress.setVisible(False)
        self.vlp_error_lbl.setText(f"Error: {msg}")

    def _apply(self):
        vals = self._collect()
        req = ["tubing_id", "depth", "thp", "T_surface", "T_bh"]
        missing = [k for k in req if not vals[k]]
        if missing:
            QMessageBox.warning(self, "Incomplete", f"Required fields: {', '.join(missing)}")
            return
        T_bh = vals["T_bh"]; T_surf = vals["T_surface"]
        if T_bh and T_surf and T_bh < T_surf:
            QMessageBox.warning(self, "Validation", "Bottomhole temperature must be ≥ surface temperature.")
            return
        for k, v in vals.items():
            setattr(self.state, k, v)
        self.state.vlp_saved = True
        self.state.save()
        self.applied.emit()
        self.accept()

# ─────────────────────────────────────────────────────────────────────────────
#  SENSITIVITY PANEL
# ─────────────────────────────────────────────────────────────────────────────
SENS_PARAMS = {
    "Reservoir Pressure (Pr)": "Pr",
    "Bubble Point (Pb)": "Pb",
    "Test Rate (Qo_test)": "Qo_test",
    "GOR": "gor",
    "Watercut (wc)": "wc",
    "Gas SG (sg_gas)": "sg_gas",
    "THP": "thp",
    "Depth": "depth",
    "Tubing ID": "tubing_id",
    "Roughness": "roughness",
    "T_bh": "T_bh",
}

class SensSlot(QGroupBox):
    def __init__(self, index: int, color: str, parent=None):
        super().__init__(f"Slot {index + 1}", parent)
        self.color = color
        self.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {color};
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 700;
                color: {color};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        layout = QVBoxLayout(self)
        self.enabled_cb = QCheckBox("Enable this slot")
        self.enabled_cb.toggled.connect(self._toggle)
        layout.addWidget(self.enabled_cb)

        param_row = QHBoxLayout()
        param_lbl = QLabel("Parameter:"); param_lbl.setFixedWidth(80)
        self.param_combo = QComboBox()
        self.param_combo.addItems(list(SENS_PARAMS.keys()))
        param_row.addWidget(param_lbl); param_row.addWidget(self.param_combo, 1)
        layout.addLayout(param_row)

        vals_row = QHBoxLayout()
        self.min_edit = QLineEdit(); self.min_edit.setPlaceholderText("Min")
        self.max_edit = QLineEdit(); self.max_edit.setPlaceholderText("Max")
        self.steps_edit = QLineEdit(); self.steps_edit.setPlaceholderText("Steps")
        self.steps_edit.setText("5")
        for e in [self.min_edit, self.max_edit, self.steps_edit]:
            e.setFixedWidth(70)
        vals_row.addWidget(QLabel("Min:")); vals_row.addWidget(self.min_edit)
        vals_row.addWidget(QLabel("Max:")); vals_row.addWidget(self.max_edit)
        vals_row.addWidget(QLabel("N:")); vals_row.addWidget(self.steps_edit)
        layout.addLayout(vals_row)

        self._toggle(False)

    def _toggle(self, state):
        for w in [self.param_combo, self.min_edit, self.max_edit, self.steps_edit]:
            w.setEnabled(state)

    def get_slot(self):
        if not self.enabled_cb.isChecked():
            return None
        param_key = self.param_combo.currentText()
        param = SENS_PARAMS[param_key]
        try:
            mn = float(self.min_edit.text())
            mx = float(self.max_edit.text())
            n = int(self.steps_edit.text())
        except Exception:
            return None
        if mn >= mx or n < 2:
            return None
        return {"param": param, "values": np.linspace(mn, mx, n).tolist()}

class SensitivityPanel(QDialog):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Sensitivity Analysis — Multi-Parameter Sweep")
        self.setMinimumSize(1100, 720)
        self._results = None
        self._setup_ui()

    def _setup_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(20)

        left_l = QVBoxLayout()
        left_l.setFixedWidth = 400
        left_l.addWidget(SectionHeader("Sensitivity Analysis", "Vary up to 3 parameters simultaneously"))

        self.slots = []
        for i in range(3):
            slot = SensSlot(i, SENS_PALETTE[i])
            self.slots.append(slot)
            left_l.addWidget(slot)

        self.sens_error_lbl = QLabel("")
        self.sens_error_lbl.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        self.sens_error_lbl.setWordWrap(True)
        left_l.addWidget(self.sens_error_lbl)
        left_l.addStretch()

        self.run_btn = QPushButton("▶  Run Sensitivity")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setToolTip("Configure and enable at least one slot to run.")
        left_l.addWidget(self.run_btn)

        self.progress = QProgressBar(); self.progress.setVisible(False)
        left_l.addWidget(self.progress)

        clear_btn = QPushButton("✕  Clear")
        clear_btn.setObjectName("secondary")
        clear_btn.clicked.connect(self._clear)
        left_l.addWidget(clear_btn)

        left_w = QWidget(); left_w.setLayout(left_l); left_w.setFixedWidth(400)
        main.addWidget(left_w)

        # Right
        right_l = QVBoxLayout()
        right_l.addWidget(make_label("SENSITIVITY CHART — IPR & VLP families by slot"))
        self.chart = MatplotlibWidget(figsize=(7, 5))
        right_l.addWidget(self.chart, 2)

        right_l.addWidget(make_label("RESULTS TABLE — Operating Points"))
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        right_l.addWidget(self.table, 1)
        main.addLayout(right_l, 1)

    def _run(self):
        active = [s.get_slot() for s in self.slots]
        active = [s for s in active if s]
        if not active:
            self.sens_error_lbl.setText("Enable at least one slot with valid range.")
            return
        self.sens_error_lbl.setText("")
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        worker = SensitivityWorker(self.state, active)
        worker.signals.finished.connect(self._on_done)
        worker.signals.error.connect(self._on_error)
        worker.signals.progress.connect(lambda p, m: self.progress.setValue(p))
        QThreadPool.globalInstance().start(worker)

    def _on_done(self, results):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._results = results
        self._plot(results)
        self._populate_table(results)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.sens_error_lbl.setText(f"Error: {msg[:200]}")

    def _plot(self, results):
        ax = self.chart.clear_axes()
        for slot_idx, slot_data in enumerate(results):
            param = slot_data["param"]
            color_base = SENS_PALETTE[slot_idx % len(SENS_PALETTE)]
            slot_results = slot_data["results"]
            cmap = plt.cm.Blues if slot_idx == 0 else (plt.cm.Greens if slot_idx == 1 else plt.cm.Purples)
            n = len(slot_results)
            for vi, entry in enumerate(slot_results):
                if "error" in entry: continue
                frac = 0.4 + 0.6 * vi / max(n - 1, 1)
                color = cmap(frac)
                ax.plot(entry["rates_vlp"], entry["p_vlp"], color=color, linewidth=1.5, alpha=0.8)
                if vi == 0 or vi == n - 1:
                    ax.plot(entry["q_ipr"], entry["p_ipr"], color=color, linewidth=1.5, linestyle="--", alpha=0.6)
                if entry.get("op_rate"):
                    ax.scatter([entry["op_rate"]], [entry["op_pwf"]], color=color, zorder=8, s=60, marker="o")

        ax.set_xlabel("Rate, q (STB/day)"); ax.set_ylabel("Pwf (psia)")
        ax.set_title("Sensitivity — VLP Families", fontsize=12, fontweight="bold", color=NAVY)
        # Legend patches
        patches = [mpatches.Patch(color=SENS_PALETTE[i % len(SENS_PALETTE)],
                                   label=results[i]["param"]) for i in range(len(results))]
        ax.legend(handles=patches, fontsize=9)
        self.chart.refresh()

    def _populate_table(self, results):
        rows = []
        for slot_data in results:
            for entry in slot_data["results"]:
                if "error" in entry: continue
                rows.append({
                    "param": slot_data["param"],
                    "value": f"{entry['val']:.4g}",
                    "q*": f"{entry['op_rate']:.1f}" if entry.get("op_rate") else "—",
                    "Pwf*": f"{entry['op_pwf']:.1f}" if entry.get("op_pwf") else "—",
                })
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value", "q* (STB/day)", "Pwf* (psia)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for ri, row in enumerate(rows):
            for ci, col in enumerate(["param", "value", "q*", "Pwf*"]):
                self.table.setItem(ri, ci, QTableWidgetItem(row[col]))

    def _clear(self):
        ax = self.chart.clear_axes()
        ax.text(0.5, 0.5, "Run sensitivity to see curves", ha="center", va="center",
                transform=ax.transAxes, color=SLATE, fontsize=12)
        self.chart.refresh()
        self.table.setRowCount(0)

    def _save_plot(self):
        if not self._results: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Sensitivity Plot", f"{self.state.well_id or 'sensitivity'}_plot.png", "PNG Files (*.png)")
        if not path: return
        try:
            self.chart.fig.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Exported", f"Plot saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not save plot: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION PANEL
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationPanel(QDialog):
    factors_applied = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("VLP Calibration")
        self.setMinimumSize(1100, 720)
        self._calib_factors = None
        self._setup_ui()

    def _setup_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(20)

        # Left: Data input and results
        left_l = QVBoxLayout()
        left_l.addWidget(SectionHeader("VLP Calibration", "Match model to measured data"))

        # Data table
        left_l.addWidget(make_label("MEASURED PRESSURE SURVEY"))
        self.table = QTableWidget(5, 2)
        self.table.setHorizontalHeaderLabels(["Depth (ft)", "Pressure (psia)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # Add sample data
        sample_data = [("0", str(self.state.thp or "200")), ("2000", "850"), ("4000", "1300"), ("6000", "1650"), (str(self.state.depth or "8000"), "1950")]
        for r, (d, p) in enumerate(sample_data):
            self.table.setItem(r, 0, QTableWidgetItem(d))
            self.table.setItem(r, 1, QTableWidgetItem(p))
        left_l.addWidget(self.table)

        # PRD §6.10 — table tools with CSV Upload
        table_tools = QHBoxLayout()
        add_row_btn = QPushButton("＋ Add Row"); add_row_btn.setObjectName("secondary")
        add_row_btn.clicked.connect(lambda: self.table.insertRow(self.table.rowCount()))
        rem_row_btn = QPushButton("－ Remove Row"); rem_row_btn.setObjectName("secondary")
        rem_row_btn.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()))
        upload_csv_btn = QPushButton("📂 Upload CSV"); upload_csv_btn.setObjectName("secondary")
        upload_csv_btn.setToolTip("Upload a measured pressure-survey CSV (columns: depth_ft, pressure_psia)")
        upload_csv_btn.clicked.connect(self._upload_csv)
        table_tools.addWidget(add_row_btn); table_tools.addWidget(rem_row_btn)
        table_tools.addStretch()
        table_tools.addWidget(upload_csv_btn)
        left_l.addLayout(table_tools)

        # CSV path display
        self.csv_path_lbl = QLabel("")
        self.csv_path_lbl.setStyleSheet(f"color:{SLATE}; font-size:10px; font-style:italic;")
        if self.state.calib_csv_path:
            self.csv_path_lbl.setText(f"Last upload: {os.path.basename(self.state.calib_csv_path)}")
        left_l.addWidget(self.csv_path_lbl)

        # Results
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet(f"color: {BLUE_M};")
        left_l.addWidget(sep)
        left_l.addWidget(make_label("CALIBRATION RESULTS"))

        self.holdup_factor_lbl = QLabel("—"); self.holdup_factor_lbl.setObjectName("kpi_value")
        self.friction_factor_lbl = QLabel("—"); self.friction_factor_lbl.setObjectName("kpi_value")
        
        results_grid = QGridLayout()
        results_grid.addWidget(make_label("Holdup Factor", style="kpi_label"), 0, 0)
        results_grid.addWidget(self.holdup_factor_lbl, 1, 0)
        results_grid.addWidget(make_label("Friction Factor", style="kpi_label"), 0, 1)
        results_grid.addWidget(self.friction_factor_lbl, 1, 1)
        left_l.addLayout(results_grid)

        self.calib_error_lbl = QLabel("")
        self.calib_error_lbl.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        self.calib_error_lbl.setWordWrap(True)
        left_l.addWidget(self.calib_error_lbl)
        left_l.addStretch()
        
        # Action buttons
        action_l = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run Calibration")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        
        self.apply_btn = QPushButton("💾 Apply Factors")
        self.apply_btn.setObjectName("secondary")
        self.apply_btn.clicked.connect(self._apply_factors)
        self.apply_btn.setEnabled(False)
        
        self.clear_btn = QPushButton("✕ Clear Calibration")
        self.clear_btn.setObjectName("secondary")
        self.clear_btn.clicked.connect(self._clear_factors)

        action_l.addWidget(self.run_btn, 1)
        action_l.addWidget(self.apply_btn)
        left_l.addLayout(action_l)
        left_l.addWidget(self.clear_btn)

        self.progress = QProgressBar(); self.progress.setVisible(False)
        left_l.addWidget(self.progress)

        left_w = QWidget()
        left_w.setLayout(left_l)
        left_w.setFixedWidth(380)
        main.addWidget(left_w)

        # Right: Chart
        right_l = QVBoxLayout()
        right_l.addWidget(make_label("CALIBRATION PLOT"))
        self.chart = MatplotlibWidget(figsize=(6, 5))
        right_l.addWidget(self.chart, 1)
        main.addLayout(right_l, 1)

        # Set initial state for clear button
        is_calibrated = bool(self.state.calib_holdup_factor != 1.0 or self.state.calib_friction_factor != 1.0)
        self.clear_btn.setVisible(is_calibrated)

    def _run(self):
        # Collect data from table
        data = []
        for r in range(self.table.rowCount()):
            try:
                depth = float(self.table.item(r, 0).text())
                pressure = float(self.table.item(r, 1).text())
                data.append((depth, pressure))
            except (ValueError, AttributeError):
                continue # Skip empty/invalid rows
        
        if len(data) < 2:
            self.calib_error_lbl.setText("⚠ Need at least 2 valid data points.")
            return

        self.calib_error_lbl.setText("")
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        worker = CalibrationWorker(self.state, data)
        worker.signals.finished.connect(self._on_done)
        worker.signals.error.connect(self._on_error)
        worker.signals.progress.connect(lambda p, m: self.progress.setValue(p))
        QThreadPool.globalInstance().start(worker)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.calib_error_lbl.setText(f"Error: {msg}")

    def _on_done(self, data):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        result = data["result"]

        if result.success:
            self.apply_btn.setEnabled(True)
            h_factor, f_factor = result.x
            self._calib_factors = (h_factor, f_factor)
            self.holdup_factor_lbl.setText(f"{h_factor:.4f}")
            self.friction_factor_lbl.setText(f"{f_factor:.4f}")
        else:
            self.apply_btn.setEnabled(False)
            self.calib_error_lbl.setText(f"Calibration failed: {result.message}")

        self._plot(data)

    def _plot(self, data):
        ax = self.chart.clear_axes()
        
        # Measured data
        ax.plot(data["measured_pressures"], data["measured_depths"], 'ko', markersize=8, label="Measured Data")

        # Original traverse
        if data.get("original_traverse"):
            orig_d, orig_p = data["original_traverse"]
            ax.plot(orig_p, orig_d, color=WARNING, linestyle='--', linewidth=2, label="Original Model")

        # Calibrated traverse
        if data.get("calibrated_traverse"):
            calib_d, calib_p = data["calibrated_traverse"]
            ax.plot(calib_p, calib_d, color=SUCCESS, linewidth=2.5, label="Calibrated Model")

        ax.invert_yaxis()
        ax.set_xlabel("Pressure (psia)")
        ax.set_ylabel("Depth (ft)")
        ax.set_title("Pressure Traverse Calibration", fontsize=12, fontweight="bold", color=NAVY)
        if data.get("original_traverse") or data.get("calibrated_traverse"):
            ax.legend()
        self.chart.refresh()

    def _apply_factors(self):
        if not self._calib_factors:
            QMessageBox.warning(self, "No Factors", "Run a successful calibration first.")
            return
        
        h_factor, f_factor = self._calib_factors
        self.state.calib_holdup_factor = h_factor
        self.state.calib_friction_factor = f_factor
        self.state.save()
        self.factors_applied.emit()
        QMessageBox.information(self, "Factors Applied", 
            f"Holdup ({h_factor:.4f}) and Friction ({f_factor:.4f}) factors have been saved.\n"
            "All subsequent VLP and Nodal runs will use this calibration.")
        self.clear_btn.setVisible(True)
        self.accept()

    def _clear_factors(self):
        self.state.calib_holdup_factor = 1.0
        self.state.calib_friction_factor = 1.0
        self.state.save()
        self.factors_applied.emit()
        QMessageBox.information(self, "Calibration Cleared", "VLP calibration factors have been reset to 1.0.")
        self.clear_btn.setVisible(False)

    def _upload_csv(self):
        """PRD §6.10 — Upload a measured pressure-survey CSV to populate the table."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Pressure Survey CSV", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        # Confirm if table already has user data
        has_data = any(
            self.table.item(r, 0) and self.table.item(r, 0).text().strip()
            for r in range(self.table.rowCount())
        )
        if has_data:
            ans = QMessageBox.question(
                self, "Replace Table Data",
                "The table already has data. Replace it with the CSV contents?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        # Parse CSV
        rows = []
        warnings = []
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header_skipped = False
                for line_num, row in enumerate(reader, start=1):
                    if not row or all(c.strip() == "" for c in row):
                        continue
                    # Skip header row (non-numeric first cell)
                    try:
                        d = float(row[0].strip())
                    except (ValueError, IndexError):
                        if not header_skipped:
                            header_skipped = True
                            continue
                        warnings.append(f"Row {line_num}: could not parse depth '{row[0] if row else ''}'")
                        continue
                    try:
                        p = float(row[1].strip())
                    except (ValueError, IndexError):
                        warnings.append(f"Row {line_num}: could not parse pressure '{row[1] if len(row) > 1 else ''}'")
                        continue
                    # Depth range check
                    max_depth = self.state.depth or float("inf")
                    if d < 0 or d > max_depth:
                        warnings.append(f"Row {line_num}: depth {d:.0f} ft outside [0, {max_depth:.0f}] — skipped")
                        continue
                    rows.append((d, p))
        except Exception as e:
            QMessageBox.warning(self, "CSV Error", f"Could not read file:\n{e}")
            return

        if len(rows) < 2:
            QMessageBox.warning(
                self, "Insufficient Data",
                "Need at least 2 valid (depth, pressure) rows after parsing."
                + ("\n\nWarnings:\n" + "\n".join(warnings) if warnings else "")
            )
            return

        # Auto-sort by ascending depth
        rows.sort(key=lambda x: x[0])

        # Populate table
        self.table.setRowCount(len(rows))
        for r, (d, p) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(f"{d:.2f}"))
            self.table.setItem(r, 1, QTableWidgetItem(f"{p:.2f}"))

        # Store path for display
        self.state.calib_csv_path = path
        self.csv_path_lbl.setText(f"Loaded: {os.path.basename(path)} ({len(rows)} rows)")
        self.state.save()

        # Show warnings if any
        if warnings:
            QMessageBox.information(
                self, "CSV Loaded with Warnings",
                f"{len(rows)} valid rows loaded.\n\nSkipped rows:\n" + "\n".join(warnings)
            )
        else:
            QMessageBox.information(
                self, "CSV Loaded",
                f"{len(rows)} data points loaded from {os.path.basename(path)}.\n"
                "You can edit rows before clicking Run Calibration."
            )

# ─────────────────────────────────────────────────────────────────────────────
#  GAS LIFT WORKER
# ─────────────────────────────────────────────────────────────────────────────
class GasLiftWorker(BaseWorker):
    """Async worker that calls compute_glpc for depth / rate / GLR sweeps."""
    def __init__(self, state: AppState, sweep_param: str, sweep_values: list,
                 gl_inj_depth=None, gl_inj_rate_mscf=None, gl_sg_gas=None):
        super().__init__()
        self._state        = copy.deepcopy(state)
        self._sweep_param  = sweep_param
        self._sweep_values = list(sweep_values)
        self._inj_depth    = gl_inj_depth
        self._inj_rate     = gl_inj_rate_mscf
        self._sg_gas       = gl_sg_gas

    @pyqtSlot()
    def run(self):
        try:
            result = compute_glpc(
                state=self._state,
                sweep_param=self._sweep_param,
                sweep_values=self._sweep_values,
                build_ipr_fn=build_ipr,
                build_vlp_fn=build_vlp,
                build_pvt_fn=build_pvt,
                get_fp_fn=get_fp,
                find_op_fn=find_operating_points,
                gl_inj_depth=self._inj_depth,
                gl_inj_rate_mscf=self._inj_rate,
                gl_sg_gas=self._sg_gas,
                progress_callback=lambda p, m: self.signals.progress.emit(p, m),
            )
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  GAS LIFT PANEL
# ─────────────────────────────────────────────────────────────────────────────
class GasLiftPanel(QDialog):
    design_applied = pyqtSignal()
    design_reset   = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Gas Lift Analysis")
        self.setMinimumSize(1150, 760)
        self._depth_result = None
        self._rate_result  = None
        self._glr_result   = None
        self._opt_depth    = None
        self._opt_rate     = None
        self._opt_glr      = None
        self._setup_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Title bar
        title_bar = QFrame()
        title_bar.setStyleSheet(f"background:{NAVY};")
        tb_l = QHBoxLayout(title_bar)
        tb_l.setContentsMargins(20, 10, 20, 10)
        t_lbl = QLabel("Gas Lift Analysis")
        t_lbl.setStyleSheet(f"color:{WHITE}; font-size:18px; font-weight:700;")
        tb_l.addWidget(t_lbl)
        tb_l.addStretch()
        self.active_lbl = make_chip(
            "⚡ Active" if self.state.gl_applied else "● Off",
            "chip_success" if self.state.gl_applied else "chip_gray")
        tb_l.addWidget(self.active_lbl)
        outer.addWidget(title_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs, 1)
        self._build_depth_tab()
        self._build_rate_tab()
        self._build_glr_tab()
        self._build_summary_tab()

        # Bottom action bar
        bot = QFrame()
        bot.setStyleSheet(f"background:{OFF_W}; border-top:1px solid {BLUE_M};")
        bot_l = QHBoxLayout(bot)
        bot_l.setContentsMargins(20, 10, 20, 10)
        self.apply_btn = QPushButton("✅  Apply Design")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self._apply_design)
        self.apply_btn.setEnabled(False)
        self.reset_btn = QPushButton("✕  Reset Gas Lift")
        self.reset_btn.setObjectName("secondary")
        self.reset_btn.clicked.connect(self._reset_design)
        self.reset_btn.setEnabled(self.state.gl_applied)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("secondary")
        close_btn.clicked.connect(self.reject)
        bot_l.addWidget(self.apply_btn)
        bot_l.addWidget(self.reset_btn)
        bot_l.addStretch()
        bot_l.addWidget(close_btn)
        outer.addWidget(bot)

    def _tab_pane(self):
        """Return (tab_widget, left_widget, left_layout, right_layout)."""
        tab = QWidget()
        lay = QHBoxLayout(tab)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(16)
        left = QWidget(); left.setFixedWidth(340)
        ll = QVBoxLayout(left); ll.setSpacing(10)
        rl = QVBoxLayout()
        lay.addWidget(left)
        lay.addLayout(rl, 1)
        return tab, left, ll, rl

    def _build_depth_tab(self):
        tab, _, ll, rl = self._tab_pane()
        ll.addWidget(SectionHeader("Optimum Injection Depth",
                                   "Sweep depth at a fixed injection rate"))
        self.d_min_e,  u = make_input("e.g. 1000",  unit="ft");       ll.addLayout(make_row("Depth Min",           self.d_min_e,  u))
        self.d_max_e,  u = make_input("e.g. 8000",  unit="ft");       ll.addLayout(make_row("Depth Max",           self.d_max_e,  u))
        self.d_step_e, u = make_input("500",         unit="ft");       ll.addLayout(make_row("Depth Step",          self.d_step_e, u))
        self.d_rate_e, u = make_input("500",         unit="Mscf/day"); ll.addLayout(make_row("Fixed Inj. Rate",     self.d_rate_e, u))
        self.d_pinj_e, u = make_input("optional",    unit="psia");     ll.addLayout(make_row("Avail. Inj. Pressure",self.d_pinj_e, u))
        self.d_err = QLabel(""); self.d_err.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        self.d_err.setWordWrap(True); ll.addWidget(self.d_err)
        ll.addStretch()
        self.d_run = QPushButton("▶  Run Depth Sweep"); self.d_run.setObjectName("primary")
        self.d_run.clicked.connect(self._run_depth); ll.addWidget(self.d_run)
        self.d_prog = QProgressBar(); self.d_prog.setVisible(False); ll.addWidget(self.d_prog)
        self.d_opt_chip = make_chip("Run sweep to find optimum", "chip_gray"); ll.addWidget(self.d_opt_chip)

        rl.addWidget(make_label("LIQUID RATE vs. INJECTION DEPTH"))
        self.d_chart = MatplotlibWidget(figsize=(6, 4)); rl.addWidget(self.d_chart, 1)
        self.tabs.addTab(tab, "1 · Optimum Depth")

        # Pre-fill
        if self.state.gl_depth_min:  self.d_min_e.setText(str(self.state.gl_depth_min))
        if self.state.gl_depth_max:  self.d_max_e.setText(str(self.state.gl_depth_max))
        if self.state.gl_depth_step: self.d_step_e.setText(str(self.state.gl_depth_step))

    def _build_rate_tab(self):
        tab, _, ll, rl = self._tab_pane()
        ll.addWidget(SectionHeader("Optimum Injection Rate",
                                   "Gas Lift Performance Curve (GLPC)"))
        self.r_depth_e,  u = make_input("e.g. 5000",  unit="ft");         ll.addLayout(make_row("Injection Depth",       self.r_depth_e,  u))
        self.r_sg_e,     u = make_input("0.65",        unit="air=1");      ll.addLayout(make_row("Injection Gas SG",      self.r_sg_e,     u))
        self.r_qmin_e,   u = make_input("100",          unit="Mscf/day");  ll.addLayout(make_row("Min Inj. Rate",          self.r_qmin_e,   u))
        self.r_qmax_e,   u = make_input("3000",         unit="Mscf/day");  ll.addLayout(make_row("Max Inj. Rate",          self.r_qmax_e,   u))
        self.r_qstep_e,  u = make_input("100",          unit="Mscf/day");  ll.addLayout(make_row("Rate Step",              self.r_qstep_e,  u))
        self.r_pinj_e,   u = make_input("optional",     unit="psia");      ll.addLayout(make_row("Avail. Inj. Pressure",   self.r_pinj_e,   u))
        self.r_qavail_e, u = make_input("optional",     unit="Mscf/day");  ll.addLayout(make_row("Compressor Ceiling",     self.r_qavail_e, u))
        self.r_econ_e,   u = make_input("0.5",          unit="STB/Mscf");  ll.addLayout(make_row("Econ Slope Threshold",   self.r_econ_e,   u))
        self.r_err = QLabel(""); self.r_err.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        self.r_err.setWordWrap(True); ll.addWidget(self.r_err)
        ll.addStretch()
        self.r_run = QPushButton("▶  Run Rate Sweep (GLPC)"); self.r_run.setObjectName("primary")
        self.r_run.clicked.connect(self._run_rate); ll.addWidget(self.r_run)
        self.r_prog = QProgressBar(); self.r_prog.setVisible(False); ll.addWidget(self.r_prog)
        self.r_opt_chip = make_chip("Run sweep to find optimum", "chip_gray"); ll.addWidget(self.r_opt_chip)

        rl.addWidget(make_label("GAS LIFT PERFORMANCE CURVE — Liquid Rate vs. Injection Gas Rate"))
        self.r_chart = MatplotlibWidget(figsize=(6, 4)); rl.addWidget(self.r_chart, 1)
        self.tabs.addTab(tab, "2 · Optimum Rate")

        if self.state.gl_inj_depth:   self.r_depth_e.setText(str(self.state.gl_inj_depth))
        self.r_sg_e.setText(str(self.state.gl_sg_gas or self.state.sg_gas))
        self.r_qmin_e.setText(str(self.state.gl_q_min))
        self.r_qmax_e.setText(str(self.state.gl_q_max))
        self.r_qstep_e.setText(str(self.state.gl_q_step))
        if self.state.gl_p_inj:       self.r_pinj_e.setText(str(self.state.gl_p_inj))
        if self.state.gl_q_available: self.r_qavail_e.setText(str(self.state.gl_q_available))
        self.r_econ_e.setText(str(self.state.gl_econ_slope))

    def _build_glr_tab(self):
        tab, _, ll, rl = self._tab_pane()
        ll.addWidget(SectionHeader("Optimum GLR",
                                   "Total GLR sweep above the injection point"))
        self.g_min_e,  u = make_input("e.g. 500",  unit="scf/STB"); ll.addLayout(make_row("GLR Min",  self.g_min_e,  u))
        self.g_max_e,  u = make_input("e.g. 5000", unit="scf/STB"); ll.addLayout(make_row("GLR Max",  self.g_max_e,  u))
        self.g_step_e, u = make_input("500",        unit="scf/STB"); ll.addLayout(make_row("GLR Step", self.g_step_e, u))
        self.g_err = QLabel(""); self.g_err.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        self.g_err.setWordWrap(True); ll.addWidget(self.g_err)
        ll.addStretch()
        self.g_run = QPushButton("▶  Run GLR Sweep"); self.g_run.setObjectName("primary")
        self.g_run.clicked.connect(self._run_glr); ll.addWidget(self.g_run)
        self.g_prog = QProgressBar(); self.g_prog.setVisible(False); ll.addWidget(self.g_prog)
        self.g_opt_chip = make_chip("Run sweep to find optimum", "chip_gray"); ll.addWidget(self.g_opt_chip)

        rl.addWidget(make_label("LIQUID RATE vs. TOTAL GLR"))
        self.g_chart = MatplotlibWidget(figsize=(6, 4)); rl.addWidget(self.g_chart, 1)
        self.tabs.addTab(tab, "3 · Optimum GLR")

        if self.state.gl_glr_min:  self.g_min_e.setText(str(self.state.gl_glr_min))
        if self.state.gl_glr_max:  self.g_max_e.setText(str(self.state.gl_glr_max))
        if self.state.gl_glr_step: self.g_step_e.setText(str(self.state.gl_glr_step))

    def _build_summary_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab); lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(16)
        lay.addWidget(SectionHeader("Summary & Apply Design",
                                    "Consolidate best picks from Tabs 1–3, then apply to all VLP computations"))

        grid = QGridLayout(); grid.setSpacing(12)
        for ci, h in enumerate(["Parameter", "Optimum Value", "Unit"]):
            lbl = QLabel(h.upper()); lbl.setObjectName("section"); grid.addWidget(lbl, 0, ci)

        self.sum_depth_lbl   = QLabel("—"); self.sum_depth_lbl.setObjectName("kpi_value")
        self.sum_rate_lbl    = QLabel("—"); self.sum_rate_lbl.setObjectName("kpi_value")
        self.sum_glr_lbl     = QLabel("—"); self.sum_glr_lbl.setObjectName("kpi_value")
        self.sum_q_lbl       = QLabel("—"); self.sum_q_lbl.setObjectName("kpi_value")
        self.sum_improve_lbl = QLabel("—"); self.sum_improve_lbl.setObjectName("kpi_value")

        for ri, (name, lbl, unit) in enumerate([
            ("Injection Depth",   self.sum_depth_lbl,   "ft"),
            ("Injection Rate",    self.sum_rate_lbl,    "Mscf/day"),
            ("Total GLR",         self.sum_glr_lbl,     "scf/STB"),
            ("Resulting q*",      self.sum_q_lbl,       "STB/day"),
            ("Rate Improvement",  self.sum_improve_lbl, "vs no-lift"),
        ], start=1):
            grid.addWidget(QLabel(name), ri, 0)
            grid.addWidget(lbl, ri, 1)
            grid.addWidget(QLabel(unit), ri, 2)

        lay.addLayout(grid)
        lay.addStretch()

        self.sum_status_lbl = QLabel("Run sweeps in Tabs 1–3 to populate this summary.")
        self.sum_status_lbl.setStyleSheet(f"color:{SLATE}; font-style:italic;")
        lay.addWidget(self.sum_status_lbl)

        self.applied_chip = make_chip(
            "⚡ Gas Lift Active" if self.state.gl_applied else "○ Not Applied",
            "chip_success" if self.state.gl_applied else "chip_gray")
        self.applied_chip.setFixedHeight(28)
        lay.addWidget(self.applied_chip, alignment=Qt.AlignmentFlag.AlignLeft)

        self.tabs.addTab(tab, "4 · Summary")

        if self.state.gl_applied:
            if self.state.gl_opt_depth: self.sum_depth_lbl.setText(f"{self.state.gl_opt_depth:.0f}")
            if self.state.gl_opt_rate:  self.sum_rate_lbl.setText(f"{self.state.gl_opt_rate:.0f}")

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _sf(self, edit, default=None):
        try: return float(edit.text())
        except: return default

    def _gl_error(self, msg, err_lbl, run_btn, prog):
        run_btn.setEnabled(True); prog.setVisible(False)
        err_lbl.setText(f"Error: {str(msg)[:200]}")

    # ── Run slots ────────────────────────────────────────────────────────────
    def _run_depth(self):
        d_min, d_max, d_step = self._sf(self.d_min_e), self._sf(self.d_max_e), self._sf(self.d_step_e, 500.0)
        if d_min is None or d_max is None or d_min >= d_max:
            self.d_err.setText("⚠ Valid Depth Min < Max required."); return
        self.d_err.setText("")
        depths = np.arange(d_min, d_max + d_step, d_step).tolist()
        p_inj  = self._sf(self.d_pinj_e)
        self.d_run.setEnabled(False); self.d_prog.setVisible(True); self.d_prog.setValue(0)
        w = GasLiftWorker(self.state, "depth", depths,
                          gl_inj_rate_mscf=self._sf(self.d_rate_e, 500.0),
                          gl_sg_gas=self.state.gl_sg_gas or self.state.sg_gas)
        w.signals.finished.connect(lambda r: self._on_depth_done(r, p_inj))
        w.signals.error.connect(lambda e: self._gl_error(e, self.d_err, self.d_run, self.d_prog))
        w.signals.progress.connect(lambda p, _: self.d_prog.setValue(p))
        QThreadPool.globalInstance().start(w)

    def _on_depth_done(self, result, p_inj):
        self.d_run.setEnabled(True); self.d_prog.setVisible(False)
        self._depth_result = result
        opt = find_optimum(result)
        if opt["opt_value"] is not None:
            self._opt_depth = opt["opt_value"]
            self.d_opt_chip.setText(f"✓ Opt. Depth: {opt['opt_value']:.0f} ft  (q*={opt['opt_q']:.0f} STB/d)")
            self.d_opt_chip.setObjectName("chip_success"); self.d_opt_chip.setStyle(self.d_opt_chip.style())
            self.r_depth_e.setText(f"{opt['opt_value']:.0f}")  # pre-fill Tab 2
            self._update_summary()
        self._plot_sweep(self.d_chart, result, "Injection Depth (ft)", opt, p_inj)

    def _run_rate(self):
        inj_depth = self._sf(self.r_depth_e)
        if inj_depth is None:
            self.r_err.setText("⚠ Injection Depth required (run Tab 1 or enter manually)."); return
        q_min = self._sf(self.r_qmin_e, 100.0); q_max = self._sf(self.r_qmax_e, 3000.0)
        q_step = self._sf(self.r_qstep_e, 100.0)
        sg_gas = self._sf(self.r_sg_e, self.state.sg_gas)
        econ   = self._sf(self.r_econ_e, 0.5)
        p_inj  = self._sf(self.r_pinj_e)
        rates  = np.arange(q_min, q_max + q_step, q_step).tolist()
        self.r_err.setText("")
        self.r_run.setEnabled(False); self.r_prog.setVisible(True); self.r_prog.setValue(0)
        w = GasLiftWorker(self.state, "rate", rates, gl_inj_depth=inj_depth, gl_sg_gas=sg_gas)
        w.signals.finished.connect(lambda r: self._on_rate_done(r, econ, p_inj, sg_gas))
        w.signals.error.connect(lambda e: self._gl_error(e, self.r_err, self.r_run, self.r_prog))
        w.signals.progress.connect(lambda p, _: self.r_prog.setValue(p))
        QThreadPool.globalInstance().start(w)

    def _on_rate_done(self, result, econ, p_inj, sg_gas):
        self.r_run.setEnabled(True); self.r_prog.setVisible(False)
        self._rate_result = result
        opt = find_optimum(result, econ_slope=econ)
        if opt["opt_value"] is not None:
            self._opt_rate = opt["opt_value"]
            self.r_opt_chip.setText(f"✓ Opt. Rate: {opt['opt_value']:.0f} Mscf/d  (q*={opt['opt_q']:.0f} STB/d)")
            self.r_opt_chip.setObjectName("chip_success"); self.r_opt_chip.setStyle(self.r_opt_chip.style())
            self.apply_btn.setEnabled(True)
            self._update_summary()
        self._plot_rate_chart(result, opt, p_inj, sg_gas)

    def _run_glr(self):
        g_min, g_max, g_step = self._sf(self.g_min_e), self._sf(self.g_max_e), self._sf(self.g_step_e, 500.0)
        if g_min is None or g_max is None or g_min >= g_max:
            self.g_err.setText("⚠ Valid GLR Min < Max required."); return
        glrs      = np.arange(g_min, g_max + g_step, g_step).tolist()
        inj_depth = self._sf(self.r_depth_e) or (self.state.depth or 5000) * 0.55
        self.g_err.setText("")
        self.g_run.setEnabled(False); self.g_prog.setVisible(True); self.g_prog.setValue(0)
        w = GasLiftWorker(self.state, "glr", glrs, gl_inj_depth=inj_depth,
                          gl_sg_gas=self.state.gl_sg_gas or self.state.sg_gas)
        w.signals.finished.connect(self._on_glr_done)
        w.signals.error.connect(lambda e: self._gl_error(e, self.g_err, self.g_run, self.g_prog))
        w.signals.progress.connect(lambda p, _: self.g_prog.setValue(p))
        QThreadPool.globalInstance().start(w)

    def _on_glr_done(self, result):
        self.g_run.setEnabled(True); self.g_prog.setVisible(False)
        self._glr_result = result
        opt = find_optimum(result)
        if opt["opt_value"] is not None:
            self._opt_glr = opt["opt_value"]
            self.g_opt_chip.setText(f"✓ Opt. GLR: {opt['opt_value']:.0f} scf/STB  (q*={opt['opt_q']:.0f} STB/d)")
            self.g_opt_chip.setObjectName("chip_success"); self.g_opt_chip.setStyle(self.g_opt_chip.style())
            self._update_summary()
        self._plot_sweep(self.g_chart, result, "Total GLR (scf/STB)", opt)

    # ── Chart helpers ────────────────────────────────────────────────────────
    def _plot_sweep(self, chart: MatplotlibWidget, result: dict,
                    xlabel: str, opt: dict,
                    p_inj_surface=None, sg_gas_inj=0.65, depth_total=None):
        ax = chart.clear_axes()
        sv = result["sweep_values"]; qr = result["q_results"]

        feasible = [True] * len(sv)
        if p_inj_surface:
            feasible = injection_feasibility_mask(
                result, p_inj_surface, sg_gas_inj,
                state_depth=depth_total or self.state.depth or 8000)

        fx, fy, ix, iy = [], [], [], []
        for i, (x, y) in enumerate(zip(sv, qr)):
            if y is None: continue
            if feasible[i]: fx.append(x); fy.append(y)
            else:           ix.append(x); iy.append(y)

        if fx: ax.plot(fx, fy, color=BLUE, linewidth=2.5, label="Feasible")
        if ix: ax.plot(ix, iy, color=BLUE, linewidth=2.0, linestyle="--",
                       alpha=0.35, label="Infeasible (inj. pressure insufficient)")
        if result.get("baseline_q"):
            ax.axhline(result["baseline_q"], color=WARNING, linestyle=":", linewidth=1.5,
                       label=f"No-lift baseline: {result['baseline_q']:.0f} STB/d")
        if opt["opt_value"] is not None and opt["opt_q"] is not None:
            ax.scatter([opt["opt_value"]], [opt["opt_q"]], color=SUCCESS,
                       zorder=9, s=150, marker="*", label=f"Optimum: {opt['opt_value']:.0f}")
        ax.set_xlabel(xlabel); ax.set_ylabel("Liquid Rate (STB/day)")
        ax.set_title(f"Gas Lift — {xlabel}", fontsize=12, fontweight="bold", color=NAVY)
        ax.legend(fontsize=9)
        chart.refresh()

    def _plot_rate_chart(self, result: dict, opt: dict, p_inj, sg_gas):
        ax = self.r_chart.clear_axes()
        sv = result["sweep_values"]; qr = result["q_results"]

        feasible = [True] * len(sv)
        if p_inj:
            feasible = injection_feasibility_mask(
                result, p_inj, sg_gas, state_depth=self.state.depth or 8000)

        fx, fy, ix, iy = [], [], [], []
        for i, (x, y) in enumerate(zip(sv, qr)):
            if y is None: continue
            if feasible[i]: fx.append(x); fy.append(y)
            else:           ix.append(x); iy.append(y)

        if fx: ax.plot(fx, fy, color=BLUE, linewidth=2.5, label="GLPC")
        if ix: ax.plot(ix, iy, color=BLUE, linewidth=2.0, linestyle="--",
                       alpha=0.35, label="Infeasible")

        if opt.get("method") == "econ_slope" and opt["opt_value"]:
            ax.axvline(opt["opt_value"], color=GOLD, linestyle="--", alpha=0.8,
                       label=f"Econ Optimum: {opt['opt_value']:.0f} Mscf/d")

        q_avail = self._sf(self.r_qavail_e)
        if q_avail:
            ax.axvline(q_avail, color=WARNING, linestyle=":", linewidth=2,
                       label=f"Compressor ceiling: {q_avail:.0f} Mscf/d")

        if result.get("baseline_q"):
            ax.axhline(result["baseline_q"], color=WARNING, linestyle=":", linewidth=1.5,
                       label=f"No-lift: {result['baseline_q']:.0f} STB/d")

        if opt["opt_value"] is not None and opt["opt_q"] is not None:
            ax.scatter([opt["opt_value"]], [opt["opt_q"]], color=SUCCESS,
                       zorder=9, s=150, marker="*", label=f"q*={opt['opt_q']:.0f} STB/d")

        ax.set_xlabel("Injection Gas Rate (Mscf/day)")
        ax.set_ylabel("Liquid Rate (STB/day)")
        ax.set_title("Gas Lift Performance Curve (GLPC)", fontsize=12, fontweight="bold", color=NAVY)
        ax.legend(fontsize=9)
        self.r_chart.refresh()

    def _update_summary(self):
        self.tabs.setTabText(3, "4 · Summary ✓")
        if self._opt_depth is not None: self.sum_depth_lbl.setText(f"{self._opt_depth:.0f}")
        if self._opt_rate  is not None: self.sum_rate_lbl.setText(f"{self._opt_rate:.0f}")
        if self._opt_glr   is not None: self.sum_glr_lbl.setText(f"{self._opt_glr:.0f}")

        best_q, baseline_q = None, None
        for r in [self._depth_result, self._rate_result, self._glr_result]:
            if not r: continue
            if r.get("baseline_q"): baseline_q = r["baseline_q"]
            valid_q = [v for v in r.get("q_results", []) if v is not None]
            if valid_q:
                cand = max(valid_q)
                if best_q is None or cand > best_q: best_q = cand

        if best_q:
            self.sum_q_lbl.setText(f"{best_q:.0f}")
        if best_q and baseline_q and baseline_q > 0:
            pct = (best_q - baseline_q) / baseline_q * 100
            self.sum_improve_lbl.setText(f"+{pct:.1f}%")
            self.sum_improve_lbl.setStyleSheet(f"font-size:20px; font-weight:700; color:{SUCCESS};")

        self.sum_status_lbl.setText("Design ready — click 'Apply Design' below to activate.")
        self.apply_btn.setEnabled(True)

    # ── Apply / Reset ────────────────────────────────────────────────────────
    def _apply_design(self):
        depth = self._opt_depth or (self.state.depth * 0.55 if self.state.depth else 4000.0)
        rate  = self._opt_rate or 500.0
        sg    = self._sf(self.r_sg_e, self.state.sg_gas)

        self.state.gl_applied   = True
        self.state.gl_opt_depth = depth
        self.state.gl_opt_rate  = rate
        self.state.gl_sg_gas    = sg
        self.state.save()

        for w in [self.active_lbl, self.applied_chip]:
            w.setText("⚡ Active" if w is self.active_lbl else "⚡ Gas Lift Active")
            w.setObjectName("chip_success"); w.setStyle(w.style())
        self.reset_btn.setEnabled(True)

        QMessageBox.information(self, "Design Applied",
            f"Gas Lift design applied:\n"
            f"  Injection Depth: {depth:.0f} ft\n"
            f"  Injection Rate:  {rate:.0f} Mscf/day\n\n"
            "All VLP curves (VLP Panel, Sensitivity, Nodal Analysis) "
            "will now reflect gas-lifted performance.")
        self.design_applied.emit()

    def _reset_design(self):
        self.state.gl_applied   = False
        self.state.gl_opt_depth = None
        self.state.gl_opt_rate  = None
        self.state.save()
        for w in [self.active_lbl, self.applied_chip]:
            w.setText("● Off" if w is self.active_lbl else "○ Not Applied")
            w.setObjectName("chip_gray"); w.setStyle(w.style())
        self.reset_btn.setEnabled(False)
        QMessageBox.information(self, "Gas Lift Reset",
            "Gas lift design cleared. All VLP curves reverted to natural (no-lift) flow.")
        self.design_reset.emit()


# ─────────────────────────────────────────────────────────────────────────────
#  CHOKE PANEL
# ─────────────────────────────────────────────────────────────────────────────
class ChokePanel(QDialog):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Choke Panel — Optimum Choke Sizing")
        self.setMinimumSize(1100, 740)
        self._perf_results = None
        self._setup_ui()
        self._load_from_state()

    def _setup_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(20)

        # Left: inputs
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(380); left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_w = QWidget(); ll = QVBoxLayout(left_w); ll.setSpacing(10)

        ll.addWidget(SectionHeader("Choke Panel", "Surface choke sizing & rate-check"))

        # Mode
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Mode:"); mode_lbl.setFixedWidth(80)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Rate-Check (single bean)", "Sizing (candidate list)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        mode_row.addWidget(mode_lbl); mode_row.addWidget(self.mode_combo, 1)
        ll.addLayout(mode_row)

        # Correlation
        corr_row = QHBoxLayout()
        corr_lbl = QLabel("Correlation:"); corr_lbl.setFixedWidth(130)
        corr_lbl.setStyleSheet(f"color:{SLATE}; font-size:12px; font-weight:600;")
        self.corr_combo = QComboBox()
        self.corr_combo.addItems(["gilbert", "ros", "achong", "baxendell", "sachdeva"])
        corr_row.addWidget(corr_lbl); corr_row.addWidget(self.corr_combo, 1)
        ll.addLayout(corr_row)

        ll.addWidget(make_label("INPUTS"))

        self.thp_e,   u = make_input("from VLP",  unit="psia");      ll.addLayout(make_row("Upstream Pressure (Pwh)", self.thp_e,   u))
        self.pdown_e, u = make_input("e.g. 100",  unit="psia");      ll.addLayout(make_row("Downstream Pressure",     self.pdown_e, u))
        self.glr_e,   u = make_input("from state", unit="scf/STB");  ll.addLayout(make_row("GLR at Wellhead",         self.glr_e,   u))

        # Rate-check mode
        self.bean_e, u = make_input("32", unit="1/64 in")
        self.bean_row_w = QWidget(); self.bean_row_w.setLayout(make_row("Bean Size", self.bean_e, u))
        ll.addWidget(self.bean_row_w)

        # Sizing mode
        self.cand_e, u = make_input("16,20,24,28,32,36,40,48,64", unit="1/64 in")
        self.cand_row_w = QWidget(); self.cand_row_w.setLayout(make_row("Candidate Sizes", self.cand_e, u))
        ll.addWidget(self.cand_row_w)

        self.tgt_q_e, u = make_input("e.g. 1500", unit="STB/day")
        self.tgt_row_w = QWidget(); self.tgt_row_w.setLayout(make_row("Target Plateau Rate", self.tgt_q_e, u))
        ll.addWidget(self.tgt_row_w)

        self.t_up_e,  u = make_input("100",  unit="°F");     ll.addLayout(make_row("Upstream Temperature",  self.t_up_e,  u))
        self.cfac_e,  u = make_input("100",  unit="");        ll.addLayout(make_row("API RP 14E c-factor",   self.cfac_e,  u))

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setStyleSheet(f"color:{BLUE_M};")
        ll.addWidget(sep)
        ll.addWidget(make_label("FLUID PROPERTIES"))
        self.sg_gas_e, u = make_input("0.65", unit="air=1");   ll.addLayout(make_row("Gas SG",    self.sg_gas_e, u))
        self.sg_oil_e, u = make_input("0.84", unit="water=1"); ll.addLayout(make_row("Oil SG",    self.sg_oil_e, u))
        self.wc_e,     u = make_input("0.0",  unit="fraction");ll.addLayout(make_row("Watercut",  self.wc_e,     u))

        self.choke_err = QLabel("")
        self.choke_err.setStyleSheet(f"color:{WARNING}; font-size:11px;")
        self.choke_err.setWordWrap(True)
        ll.addWidget(self.choke_err)
        ll.addStretch()

        calc_btn = QPushButton("▶  Calculate")
        calc_btn.setObjectName("primary")
        calc_btn.clicked.connect(self._calculate)
        ll.addWidget(calc_btn)

        left_scroll.setWidget(left_w)
        main.addWidget(left_scroll)

        # Right: results
        rl = QVBoxLayout()
        rl.addWidget(make_label("CHOKE PERFORMANCE"))

        # Status badges
        badges = QHBoxLayout()
        self.flow_badge    = make_chip("—", "chip_gray"); self.flow_badge.setFixedHeight(28)
        self.eros_badge    = make_chip("—", "chip_gray"); self.eros_badge.setFixedHeight(28)
        self.hydrate_badge = make_chip("—", "chip_gray"); self.hydrate_badge.setFixedHeight(28)
        self.rec_badge     = make_chip("—", "chip_gray"); self.rec_badge.setFixedHeight(28)
        for b in [self.flow_badge, self.eros_badge, self.hydrate_badge, self.rec_badge]:
            badges.addWidget(b)
        badges.addStretch()
        rl.addLayout(badges)

        self.choke_chart = MatplotlibWidget(figsize=(6, 3.5))
        rl.addWidget(self.choke_chart, 1)

        rl.addWidget(make_label("CANDIDATE RESULTS"))
        self.choke_table = QTableWidget()
        self.choke_table.setAlternatingRowColors(True)
        self.choke_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.choke_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        rl.addWidget(self.choke_table, 1)

        main.addLayout(rl, 1)
        self._on_mode_change(0)

    def _on_mode_change(self, idx):
        sizing = (idx == 1)
        self.bean_row_w.setVisible(not sizing)
        self.cand_row_w.setVisible(sizing)
        self.tgt_row_w.setVisible(sizing)

    def _load_from_state(self):
        s = self.state
        models = ["gilbert", "ros", "achong", "baxendell", "sachdeva"]
        self.corr_combo.setCurrentIndex(models.index(s.choke_model) if s.choke_model in models else 0)
        if s.thp:         self.thp_e.setText(str(s.thp))
        if s.choke_p_down:self.pdown_e.setText(str(s.choke_p_down))
        glr = s.gor or 500.0
        self.glr_e.setText(str(glr))
        self.bean_e.setText(str(s.choke_size_64))
        self.cand_e.setText(s.choke_sizes_list)
        if s.choke_target_q: self.tgt_q_e.setText(str(s.choke_target_q))
        self.cfac_e.setText(str(s.choke_c_factor))
        if s.T_surface: self.t_up_e.setText(str(s.T_surface))
        self.sg_gas_e.setText(str(s.sg_gas))
        self.sg_oil_e.setText(str(s.sg_oil))
        self.wc_e.setText(str(s.wc))

    def _sf(self, edit, default=None):
        try: return float(edit.text())
        except: return default

    def _calculate(self):
        self.choke_err.setText("")
        model  = self.corr_combo.currentText()
        p_up   = self._sf(self.thp_e)
        p_down = self._sf(self.pdown_e)
        glr    = self._sf(self.glr_e)
        t_up   = self._sf(self.t_up_e, 100.0)
        c_fac  = self._sf(self.cfac_e, 100.0)
        sg_gas = self._sf(self.sg_gas_e, 0.65)
        sg_oil = self._sf(self.sg_oil_e, 0.84)
        wc     = self._sf(self.wc_e, 0.0)

        if not all([p_up, p_down, glr]):
            self.choke_err.setText("⚠ Upstream pressure, downstream pressure, and GLR are required."); return
        if p_down >= p_up:
            self.choke_err.setText("⚠ Warning: Downstream pressure ≥ Upstream — critical flow may not hold.")

        is_sizing = (self.mode_combo.currentIndex() == 1)
        if is_sizing:
            try:
                beans = [float(x.strip()) for x in self.cand_e.text().split(",") if x.strip()]
                if not beans: raise ValueError
            except Exception:
                self.choke_err.setText("⚠ Invalid candidate sizes (e.g. 16,24,32,40)."); return
        else:
            beans = [self._sf(self.bean_e, 32.0)]

        rho_liq = 62.4 * sg_oil * (1.0 - wc) + 62.4 * 1.07 * wc

        try:
            results = choke_performance_curve(
                bean_sizes_64=beans, glr=glr, p_up=p_up, p_down=p_down,
                model=model, sg_gas=sg_gas, sg_oil=sg_oil, wc=wc,
                t_up=t_up, c_factor=c_fac, rho_liq_lbm_ft3=rho_liq)
        except Exception as e:
            self.choke_err.setText(f"Calculation error: {e}"); return

        self._perf_results = results

        # Save key state fields
        self.state.choke_model    = model
        self.state.choke_p_down   = p_down
        self.state.choke_c_factor = c_fac
        self.state.choke_sizes_list = self.cand_e.text()
        self.state.save()

        # Update badges
        if results:
            crit = results[0]["is_critical"]
            self.flow_badge.setText("🔴 Critical Flow" if crit else "🟡 Subcritical Flow")
            self.flow_badge.setObjectName("chip_success" if crit else "chip_gold")
            self.flow_badge.setStyle(self.flow_badge.style())

            all_pass = all(r["erosional"]["pass_check"] for r in results)
            self.eros_badge.setText("✅ Erosion OK" if all_pass else "⚠ Erosion Risk")
            self.eros_badge.setObjectName("chip_success" if all_pass else "chip_warning")
            self.eros_badge.setStyle(self.eros_badge.style())

            jt = joule_thomson_estimate(p_up, p_down, t_up)
            self.hydrate_badge.setText(
                f"⚠ Hydrate Risk ({jt['t_down']:.0f}°F)" if jt["hydrate_risk"]
                else f"✅ No Hydrate ({jt['t_down']:.0f}°F)")
            self.hydrate_badge.setObjectName("chip_warning" if jt["hydrate_risk"] else "chip_success")
            self.hydrate_badge.setStyle(self.hydrate_badge.style())

        tgt_q = self._sf(self.tgt_q_e) if is_sizing else None
        if is_sizing and tgt_q:
            rec = recommend_bean_size(results, tgt_q)
            if rec:
                self.rec_badge.setText(f"★ Rec: {rec['bean_64']:.0f}/64\" ({rec['q_pred']:.0f} STB/d)")
                self.rec_badge.setObjectName("chip_blue")
            else:
                self.rec_badge.setText("No candidate meets target")
        elif not is_sizing and results:
            self.rec_badge.setText(f"Predicted Rate: {results[0]['q_pred']:.0f} STB/d")
            self.rec_badge.setObjectName("chip_blue")
        self.rec_badge.setStyle(self.rec_badge.style())

        self._plot_choke(results, tgt_q)
        self._populate_table(results, tgt_q)

    def _plot_choke(self, results, target_q=None):
        ax = self.choke_chart.clear_axes()
        beans  = [r["bean_64"] for r in results]
        rates  = [r["q_pred"]  for r in results]
        colors = [SUCCESS if r["erosional"]["pass_check"] else WARNING for r in results]

        x = range(len(beans))
        ax.bar(x, rates, color=colors, alpha=0.80, edgecolor=NAVY, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b:.0f}/64"' for b in beans], rotation=30, ha="right")
        if target_q:
            ax.axhline(target_q, color=BLUE, linestyle="--", linewidth=1.8,
                       label=f"Target: {target_q:.0f} STB/d")
            ax.legend(fontsize=9)
        ax.set_xlabel('Bean Size (1/64 in)')
        ax.set_ylabel('Predicted Liquid Rate (STB/day)')
        ax.set_title("Choke Performance — Rate vs. Bean Size",
                     fontsize=12, fontweight="bold", color=NAVY)
        # Legend for erosional
        import matplotlib.patches as _mp
        patches = [_mp.Patch(color=SUCCESS, label="Erosion OK"),
                   _mp.Patch(color=WARNING, label="Erosion Risk")]
        ax.legend(handles=patches, fontsize=9, loc="upper left")
        self.choke_chart.refresh()

    def _populate_table(self, results, target_q=None):
        rec_bean = None
        if target_q:
            rec = recommend_bean_size(results, target_q)
            if rec: rec_bean = rec["bean_64"]

        hdrs = ["Bean (1/64 in)", "Rate (STB/day)", "Pwh Pred (psia)",
                "Flow Regime", "V_act (ft/s)", "V_eros (ft/s)", "Erosion", "Recommended"]
        self.choke_table.setColumnCount(len(hdrs))
        self.choke_table.setHorizontalHeaderLabels(hdrs)
        self.choke_table.setRowCount(len(results))

        for ri, r in enumerate(results):
            eros   = r["erosional"]
            regime = "Critical" if r["is_critical"] else "Subcritical"
            is_rec = (rec_bean is not None and r["bean_64"] == rec_bean)
            vals = [
                f"{r['bean_64']:.0f}",
                f"{r['q_pred']:.1f}",
                f"{r['p_up_pred']:.1f}",
                regime,
                f"{eros['v_actual']:.2f}",
                f"{eros['v_erosional']:.2f}",
                "✅ OK" if eros["pass_check"] else "⚠ Risk",
                "★ Recommended" if is_rec else "",
            ]
            for ci, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if is_rec: item.setBackground(QColor(BLUE_L))
                if ci == 6 and not eros["pass_check"]: item.setForeground(QColor(WARNING))
                self.choke_table.setItem(ri, ci, item)


# ─────────────────────────────────────────────────────────────────────────────
#  COMING SOON DIALOG
# ─────────────────────────────────────────────────────────────────────────────
class ComingSoonDialog(QDialog):
    def __init__(self, feature_name, description, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{feature_name} — Coming Soon")
        self.setFixedSize(480, 340)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        badge = make_chip("🚧  On the Roadmap", "chip_gold")
        badge.setFixedHeight(32); badge.setFixedWidth(160)
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignLeft)

        title = QLabel(feature_name)
        title.setStyleSheet(f"font-size: 22px; font-weight: 700; color: {NAVY};")
        layout.addWidget(title)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size: 13px; color: {SLATE}; line-height: 150%;")
        layout.addWidget(desc)

        # Placeholder visual — grayed out fake chart area
        placeholder = QFrame()
        placeholder.setFixedHeight(100)
        placeholder.setStyleSheet(f"""
            QFrame {{
                background: {OFF_W};
                border: 1.5px dashed {BLUE_M};
                border-radius: 8px;
            }}
        """)
        ph_layout = QVBoxLayout(placeholder)
        ph_lbl = QLabel("[ Feature Preview Coming Soon ]")
        ph_lbl.setStyleSheet(f"color: {BLUE_M}; font-size: 12px;")
        ph_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_layout.addWidget(ph_lbl)
        layout.addWidget(placeholder)

        close_btn = QPushButton("Got it")
        close_btn.setObjectName("primary")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

# ─────────────────────────────────────────────────────────────────────────────
#  NODAL ANALYSIS SCREEN
# ─────────────────────────────────────────────────────────────────────────────
class NodalAnalysisScreen(QWidget):
    back_requested = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._nodal_data = None
        self._setup_ui()

    def _setup_ui(self):
        main_l = QVBoxLayout(self)
        main_l.setContentsMargins(20, 16, 20, 16)
        main_l.setSpacing(12)

        # Top bar
        top_bar = QHBoxLayout()
        back_btn = QPushButton("← Home")
        back_btn.setObjectName("secondary")
        back_btn.clicked.connect(self.back_requested.emit)
        top_bar.addWidget(back_btn)
        title = QLabel("Nodal Analysis")
        title.setStyleSheet(f"font-size:20px; font-weight:700; color:{NAVY};")
        top_bar.addWidget(title)
        top_bar.addStretch()
        self.run_btn = QPushButton("▶  Run Analysis")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        top_bar.addWidget(self.run_btn)
        main_l.addLayout(top_bar)

        # Progress
        self.progress = QProgressBar(); self.progress.setVisible(False)
        main_l.addWidget(self.progress)

        # Operating Point banner
        self.banner = QFrame(); self.banner.setObjectName("card")
        banner_l = QHBoxLayout(self.banner)
        banner_l.setContentsMargins(16, 12, 16, 12)

        self.op_kpis = {}
        for key, label, unit in [
            ("q_star", "Operating Rate", "STB/day"),
            ("pwf_star", "Flowing BHP", "psia"),
            ("stability", "Stability", ""),
            ("drawdown", "Drawdown", "psia"),
            ("pi", "PI (J)", "STB/day/psi"),
        ]:
            kpi_frame = QFrame()
            kpi_l = QVBoxLayout(kpi_frame)
            kpi_l.setContentsMargins(10, 4, 10, 4)
            klbl = QLabel(label.upper()); klbl.setObjectName("kpi_label")
            val_lbl = QLabel("—"); val_lbl.setObjectName("kpi_value")
            kpi_l.addWidget(klbl); kpi_l.addWidget(val_lbl)
            if unit:
                u_lbl = QLabel(unit); u_lbl.setStyleSheet(f"font-size:10px; color:{SLATE};")
                kpi_l.addWidget(u_lbl)
            banner_l.addWidget(kpi_frame)
            if key != "pi":
                sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet(f"color:{BLUE_M};"); banner_l.addWidget(sep)
            self.op_kpis[key] = val_lbl

        main_l.addWidget(self.banner)

        # Main split: chart (A) | tabs (B)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Pane A: Chart
        self.chart_widget = MatplotlibWidget(figsize=(7, 5))
        splitter.addWidget(self.chart_widget)

        # Pane B: Tabs
        self.tab_widget = QTabWidget()

        # Tab 1: Traverse table
        traverse_tab = QWidget()
        tt_l = QVBoxLayout(traverse_tab)
        self.traverse_table = QTableWidget()
        self.traverse_table.setAlternatingRowColors(True)
        self.traverse_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.traverse_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.traverse_table.setSortingEnabled(True)
        tt_l.addWidget(self.traverse_table)
        self.tab_widget.addTab(traverse_tab, "📊  Pressure Traverse")

        # Tab 2: PVT @ op
        pvt_tab = QWidget()
        pvt_tab_l = QVBoxLayout(pvt_tab)
        self.pvt_grid = QGridLayout()
        pvt_scroll = QScrollArea(); pvt_scroll.setWidgetResizable(True)
        pvt_grid_w = QWidget(); pvt_grid_w.setLayout(self.pvt_grid)
        pvt_scroll.setWidget(pvt_grid_w)
        pvt_tab_l.addWidget(pvt_scroll)
        self.tab_widget.addTab(pvt_tab, "🧪  PVT @ Operating Point")
        
        # Tab 3: Traverse Chart
        traverse_chart_tab = QWidget()
        tc_l = QVBoxLayout(traverse_chart_tab)
        self.traverse_chart_widget = MatplotlibWidget(figsize=(7, 5))
        tc_l.addWidget(self.traverse_chart_widget)
        self.tab_widget.addTab(traverse_chart_tab, "📈 Traverse Chart")

        splitter.addWidget(self.tab_widget)
        splitter.setSizes([600, 500])
        main_l.addWidget(splitter, 1)

        # Bottom toolbar
        bottom_bar = QHBoxLayout()
        self.export_btn = QPushButton("📥  Export CSV ▾")
        self.export_btn.setObjectName("secondary")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._show_export_menu)
        bottom_bar.addWidget(self.export_btn)

        rerun_btn = QPushButton("↺  Re-run")
        rerun_btn.setObjectName("secondary")
        rerun_btn.clicked.connect(self._run)
        bottom_bar.addWidget(rerun_btn)

        open_sens_btn = QPushButton("→  Open in Sensitivity")
        open_sens_btn.setObjectName("secondary")
        open_sens_btn.clicked.connect(self._open_sensitivity)
        bottom_bar.addWidget(open_sens_btn)

        # PRD §6.9 — Use choke as outflow boundary toggle
        self.choke_outflow_cb = QCheckBox("Use choke as outflow boundary")
        self.choke_outflow_cb.setToolTip(
            "Replaces fixed THP with a choke-derived upstream pressure (rate-dependent).\n"
            "Configure bean size and downstream pressure in the Choke panel first."
        )
        choke_ready = bool(
            getattr(self.state, "choke_size_64", None) and
            getattr(self.state, "choke_p_down", None) and
            getattr(self.state, "thp", None)
        )
        self.choke_outflow_cb.setEnabled(choke_ready)
        bottom_bar.addWidget(self.choke_outflow_cb)

        # Save plot button
        self.save_plot_btn = QPushButton("🖼️ Save Plot")
        self.save_plot_btn.setObjectName("secondary")
        self.save_plot_btn.setEnabled(False)
        self.save_plot_btn.clicked.connect(self._save_plot)
        bottom_bar.addWidget(self.save_plot_btn)

        bottom_bar.addStretch()

        self.error_lbl = QLabel("")
        self.error_lbl.setStyleSheet(f"color:{WARNING}; font-size:12px;")
        self.error_lbl.setWordWrap(True)
        bottom_bar.addWidget(self.error_lbl)

        main_l.addLayout(bottom_bar)

    def _run(self):
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.error_lbl.setText("")

        # PRD §6.9 — refresh choke-outflow toggle availability
        choke_ready = bool(
            getattr(self.state, "choke_size_64", None) and
            getattr(self.state, "choke_p_down", None) and
            getattr(self.state, "thp", None)
        )
        self.choke_outflow_cb.setEnabled(choke_ready)
        if not choke_ready:
            self.choke_outflow_cb.setChecked(False)

        # Build a modified state for the worker when choke outflow is active
        run_state = self.state
        if choke_ready and self.choke_outflow_cb.isChecked():
            run_state = self._build_choke_outflow_state()

        worker = NodalWorker(run_state)
        worker.signals.finished.connect(self._on_done)
        worker.signals.error.connect(self._on_error)
        worker.signals.progress.connect(lambda p, m: self.progress.setValue(p))
        QThreadPool.globalInstance().start(worker)

    def _build_choke_outflow_state(self):
        """PRD §6.9 — Compute choke-derived effective THP at the current operating rate
        and return a cloned state with thp replaced by that value.
        
        Uses a single-point choke estimate at the test rate (or q_min as fallback)
        to set an effective THP before handing off to the standard NodalWorker.
        This is additive — no engine modules are modified.
        """
        import copy as _copy
        s = _copy.deepcopy(self.state)
        try:
            glr = s.gor or 500.0
            p_up = s.thp or 500.0
            p_down = s.choke_p_down
            bean = s.choke_size_64
            model = s.choke_model or "gilbert"
            sg_gas = s.sg_gas
            sg_oil = s.sg_oil
            wc = s.wc
            t_up = s.T_surface or 100.0
            c_factor = s.choke_c_factor or 100.0
            rho_liq = 62.4 * sg_oil * (1.0 - wc) + 62.4 * 1.07 * wc

            # Compute choke-predicted upstream pressure for the single bean
            results = choke_performance_curve(
                bean_sizes_64=[bean], glr=glr, p_up=p_up, p_down=p_down,
                model=model, sg_gas=sg_gas, sg_oil=sg_oil, wc=wc, t_up=t_up,
                c_factor=c_factor, rho_liq=rho_liq
            )
            if results:
                # p_up_pred is the upstream (wellhead) pressure implied by the choke at this rate
                effective_thp = results[0].get("p_up_pred", p_up)
                s.thp = float(effective_thp)
        except Exception as exc:
            print(f"[Nodal] choke outflow THP estimate failed: {exc} — using fixed THP")
        return s

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.error_lbl.setText(f"⚠ {msg[:300]}")

    def _on_done(self, data):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._nodal_data = data
        result: NodalResult = data["result"]

        # Update banner KPIs
        if result.success and result.stable_point:
            pt = result.stable_point
            self.op_kpis["q_star"].setText(f"{pt.rate:.1f}")
            self.op_kpis["pwf_star"].setText(f"{pt.pwf:.1f}")
            self.op_kpis["drawdown"].setText(f"{pt.drawdown:.1f}")
            self.op_kpis["pi"].setText(f"{pt.productivity_index:.4f}")
            stab = pt.stability.value
            self.op_kpis["stability"].setText(stab)
            stab_color = SUCCESS if stab == "Stable" else (WARNING if stab == "Unstable" else GOLD)
            self.op_kpis["stability"].setStyleSheet(f"font-size:20px; font-weight:700; color:{stab_color};")
        else:
            for key, lbl in self.op_kpis.items():
                lbl.setText("—")
            reason = result.failure_reason if not result.success else ""
            if reason:
                self.error_lbl.setText(f"⚠ No operating point found.\n{reason}")

        self._plot_nodal(data)
        self._populate_traverse(data.get("traverse"))
        self._populate_pvt_at_op(data.get("pvt_at_op"))
        self.export_btn.setEnabled(True)
        self.save_plot_btn.setEnabled(True)

    def _plot_nodal(self, data):
        ax = self.chart_widget.clear_axes()
        result: NodalResult = data["result"]

        q_ipr = data["q_ipr"]; p_ipr = data["p_ipr"]
        rates_vlp = data["rates_vlp"]; p_vlp = data["p_vlp"]

        ax.plot(q_ipr, p_ipr, color=BLUE, linewidth=2.5, label="IPR Curve")
        valid_vlp = [(r, p) for r, p in zip(rates_vlp, p_vlp) if not (isinstance(p, float) and p != p)]
        if valid_vlp:
            rv, pv = zip(*valid_vlp)
            ax.plot(rv, pv, color="#00897B", linewidth=2.5, label="VLP Curve")

        # Plot intersection markers
        if result.success:
            for pt in result.all_points:
                if pt.stability == StabilityType.STABLE:
                    marker, color, zorder = "o", SUCCESS, 9
                    ax.scatter([pt.rate], [pt.pwf], color=color, zorder=zorder, s=120, marker=marker,
                               label=f"Stable: q={pt.rate:.0f} STB/d")
                elif pt.stability == StabilityType.UNSTABLE:
                    marker, color, zorder = "^", WARNING, 9
                    ax.scatter([pt.rate], [pt.pwf], color=color, zorder=zorder, s=120, marker=marker,
                               label=f"Unstable: q={pt.rate:.0f} STB/d")
                else:
                    marker, color, zorder = "D", GOLD, 9
                    ax.scatter([pt.rate], [pt.pwf], color=color, zorder=zorder, s=100, marker=marker,
                               label=f"Indeterminate: q={pt.rate:.0f} STB/d")
        else:
            ax.text(0.5, 0.5, result.failure_reason, ha="center", va="center",
                    transform=ax.transAxes, color=WARNING, fontsize=10, wrap=True,
                    bbox=dict(facecolor="#FFEBEE", edgecolor=WARNING, boxstyle="round,pad=0.4"))

        ax.set_xlabel("Liquid Rate, q (STB/day)")
        ax.set_ylabel("Flowing BHP, Pwf (psia)")
        ax.set_title("Nodal Analysis — IPR × VLP Intersection", fontsize=12, fontweight="bold", color=NAVY)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, framealpha=0.9)

        self.chart_widget.refresh()

    def _populate_traverse(self, traverse):
        if not traverse:
            return
        depths = traverse["depths"]
        pressures = traverse["pressures"]
        profiles = traverse["profiles"]
        headers = ["Depth (ft)", "Pressure (psia)", "Holdup", "Friction Factor",
                   "Hydrostatic (psi/ft)", "Frictional (psi/ft)", "Total Grad (psi/ft)"]
        self.traverse_table.setColumnCount(len(headers))
        self.traverse_table.setHorizontalHeaderLabels(headers)
        n = len(depths)
        self.traverse_table.setRowCount(n)

        for i in range(n):
            vals = [
                f"{depths[i]:.1f}", f"{pressures[i]:.2f}",
                f"{profiles['holdup'][i]:.4f}" if i < len(profiles['holdup']) else "—",
                f"{profiles['friction_factor'][i]:.6f}" if i < len(profiles['friction_factor']) else "—",
                f"{profiles['hydrostatic_loss'][i]:.6f}" if i < len(profiles['hydrostatic_loss']) else "—",
                f"{profiles['frictional_loss'][i]:.6f}" if i < len(profiles['frictional_loss']) else "—",
                f"{profiles['total_gradient'][i]:.6f}" if i < len(profiles['total_gradient']) else "—",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.traverse_table.setItem(i, j, item)
        
        # Update traverse chart tab
        if traverse:
            ax = self.traverse_chart_widget.clear_axes()
            ax.plot(traverse["pressures"], traverse["depths"], color="#6A1B9A", linewidth=2.5)
            ax.invert_yaxis()
            ax.set_xlabel("Pressure (psia)")
            ax.set_ylabel("Depth (ft)")
            ax.set_title("Pressure Traverse at Operating Point", fontsize=12, fontweight="bold", color=NAVY)
            ax.grid(True, linestyle="--", alpha=0.7)
            self.traverse_chart_widget.refresh()


    def _populate_pvt_at_op(self, pvt_data):
        # Clear grid
        while self.pvt_grid.count():
            item = self.pvt_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not pvt_data:
            return

        display_map = {
            "gor": ("Solution GOR (Rs)", "scf/STB"),
            "Bo": ("Oil FVF (Bo)", "bbl/STB"),
            "Bg": ("Gas FVF (Bg)", "ft³/scf"),
            "Bw": ("Water FVF (Bw)", "bbl/STB"),
            "Z": ("Z-Factor", "—"),
            "rho_l": ("Liquid Density", "lbm/ft³"),
            "rho_g": ("Gas Density", "lbm/ft³"),
            "mu_l": ("Liquid Viscosity", "cp"),
            "mu_g": ("Gas Viscosity", "cp"),
            "Pb": ("Bubble Point", "psia"),
            "glr": ("GLR", "scf/STB"),
            "Pr": ("Pressure", "psia"),
            "Tr": ("Temperature", "°F"),
        }
        row = 0
        for key, (label, unit) in display_map.items():
            if key in pvt_data:
                val = pvt_data[key]
                try:
                    val_str = f"{float(val):.4g}"
                except Exception:
                    val_str = str(val)
                lbl = QLabel(f"{label}:"); lbl.setStyleSheet(f"font-weight:600; color:{SLATE}; font-size:12px;")
                val_lbl = QLabel(val_str); val_lbl.setStyleSheet(f"font-weight:700; color:{NAVY}; font-size:13px;")
                unit_lbl = QLabel(unit); unit_lbl.setStyleSheet(f"color:{SLATE}; font-size:11px;")
                self.pvt_grid.addWidget(lbl, row, 0)
                self.pvt_grid.addWidget(val_lbl, row, 1)
                self.pvt_grid.addWidget(unit_lbl, row, 2)
                row += 1

    def _save_plot(self):
        if not self._nodal_data: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG", f"{self.state.well_id or 'nodal'}_plot.png", "PNG Files (*.png)")
        if not path: return
        try:
            self.chart_widget.fig.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Exported", f"Plot saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Could not save plot: {e}")

    def _show_export_menu(self):
        menu = QMenu(self)
        acts = [
            ("Operating Point(s)", self._export_op),
            ("PVT @ Operating Point", self._export_pvt_op),
            ("IPR Curve", self._export_ipr),
            ("VLP Curve", self._export_vlp),
            ("Pressure Traverse", self._export_traverse),
            ("Export All", self._export_all),
        ]
        for label, fn in acts:
            act = QAction(label, self)
            act.triggered.connect(fn)
            menu.addAction(act)
        menu.exec(self.export_btn.mapToGlobal(QPoint(0, self.export_btn.height())))

    def _get_save_path(self, default="ipm_export.csv"):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default, "CSV Files (*.csv)")
        return path

    def _export_op(self):
        if not self._nodal_data: return
        result: NodalResult = self._nodal_data["result"]
        path = self._get_save_path("operating_point.csv")
        if not path: return
        lines = [
            f"# Well ID: {self.state.well_id}, Field: {self.state.field_name}\n",
            "# Operating Points\n", "stability,q_star,pwf,drawdown,drawdown_pct,PI,message\n"
        ]
        for pt in result.all_points:
            lines.append(f"{pt.stability.value},{pt.rate:.4f},{pt.pwf:.4f},{pt.drawdown:.4f},{pt.drawdown_pct:.2f},{pt.productivity_index:.6f},{pt.message}\n")
        with open(path, "w") as f:
            f.writelines(lines)
        QMessageBox.information(self, "Exported", f"Saved to {path}")

    def _export_pvt_op(self):
        if not self._nodal_data or not self._nodal_data.get("pvt_at_op"): return
        pvt = self._nodal_data["pvt_at_op"]
        path = self._get_save_path("pvt_at_op.csv")
        if not path: return
        with open(path, "w") as f:
            f.write(f"# Well ID: {self.state.well_id}, Field: {self.state.field_name}\n")
            f.write("# PVT @ Operating Point\n")
            for k, v in pvt.items():
                f.write(f"{k},{v}\n")
        QMessageBox.information(self, "Exported", f"Saved to {path}")

    def _export_ipr(self):
        if not self._nodal_data: return
        path = self._get_save_path("ipr_curve.csv")
        if not path: return
        with open(path, "w") as f:
            f.write(f"# Well ID: {self.state.well_id}, Field: {self.state.field_name}\n")
            f.write("# IPR Curve\n")
            f.write("q_STB_day,Pwf_psia\n")
            for q, p in zip(self._nodal_data["q_ipr"], self._nodal_data["p_ipr"]):
                f.write(f"{q:.4f},{p:.4f}\n")
        QMessageBox.information(self, "Exported", f"Saved to {path}")

    def _export_vlp(self):
        if not self._nodal_data: return
        path = self._get_save_path("vlp_curve.csv")
        if not path: return
        with open(path, "w") as f:
            f.write(f"# Well ID: {self.state.well_id}, Field: {self.state.field_name}\n")
            f.write("# VLP Curve\n")
            f.write("q_STB_day,Pwf_psia\n")
            for q, p in zip(self._nodal_data["rates_vlp"], self._nodal_data["p_vlp"]):
                f.write(f"{q:.4f},{p:.4f}\n")
        QMessageBox.information(self, "Exported", f"Saved to {path}")

    def _export_traverse(self):
        if not self._nodal_data or not self._nodal_data.get("traverse"): return
        traverse = self._nodal_data["traverse"]
        path = self._get_save_path("pressure_traverse.csv")
        if not path: return
        with open(path, "w") as f:
            f.write(f"# Well ID: {self.state.well_id}, Field: {self.state.field_name}\n")
            f.write("# Pressure Traverse\n")
            f.write("Depth_ft,Pressure_psia,Holdup,FrictionFactor,Hydrostatic_psi_ft,Frictional_psi_ft,TotalGrad_psi_ft\n")
            depths = traverse["depths"]; pressures = traverse["pressures"]; prof = traverse["profiles"]
            for i in range(len(depths)):
                f.write(f"{depths[i]:.2f},{pressures[i]:.4f},"
                        f"{prof['holdup'][i] if i<len(prof['holdup']) else ''},"
                        f"{prof['friction_factor'][i] if i<len(prof['friction_factor']) else ''},"
                        f"{prof['hydrostatic_loss'][i] if i<len(prof['hydrostatic_loss']) else ''},"
                        f"{prof['frictional_loss'][i] if i<len(prof['frictional_loss']) else ''},"
                        f"{prof['total_gradient'][i] if i<len(prof['total_gradient']) else ''}\n")
        QMessageBox.information(self, "Exported", f"Saved to {path}")

    def _export_all(self):
        path = self._get_save_path("ipm_export_all.csv")
        if not path or not self._nodal_data: return
        data = self._nodal_data
        result: NodalResult = data["result"]
        with open(path, "w") as f:
            f.write(f"# Well ID: {self.state.well_id}, Field: {self.state.field_name}\n\n")
            # Operating Points
            f.write("# === OPERATING POINTS ===\n")
            f.write("stability,q_star,pwf,drawdown,drawdown_pct,PI\n")
            for pt in result.all_points:
                f.write(f"{pt.stability.value},{pt.rate:.4f},{pt.pwf:.4f},{pt.drawdown:.4f},{pt.drawdown_pct:.2f},{pt.productivity_index:.6f}\n")
            f.write("\n")
            # PVT @ op
            if data.get("pvt_at_op"):
                f.write("# === PVT @ OPERATING POINT ===\n")
                for k, v in data["pvt_at_op"].items():
                    f.write(f"{k},{v}\n")
                f.write("\n")
            # IPR Curve
            f.write("# === IPR CURVE ===\nq_STB_day,Pwf_psia\n")
            for q, p in zip(data["q_ipr"], data["p_ipr"]):
                f.write(f"{q:.4f},{p:.4f}\n")
            f.write("\n")
            # VLP Curve
            f.write("# === VLP CURVE ===\nq_STB_day,Pwf_psia\n")
            for q, p in zip(data["rates_vlp"], data["p_vlp"]):
                f.write(f"{q:.4f},{p:.4f}\n")
            f.write("\n")
            # Traverse
            if data.get("traverse"):
                traverse = data["traverse"]
                f.write("# === PRESSURE TRAVERSE ===\nDepth_ft,Pressure_psia,Holdup,FrictionFactor,Hydrostatic_psi_ft,Frictional_psi_ft,TotalGrad_psi_ft\n")
                prof = traverse["profiles"]
                for i in range(len(traverse["depths"])):
                    f.write(f"{traverse['depths'][i]:.2f},{traverse['pressures'][i]:.4f},"
                            f"{prof['holdup'][i] if i<len(prof['holdup']) else ''},"
                            f"{prof['friction_factor'][i] if i<len(prof['friction_factor']) else ''},"
                            f"{prof['hydrostatic_loss'][i] if i<len(prof['hydrostatic_loss']) else ''},"
                            f"{prof['frictional_loss'][i] if i<len(prof['frictional_loss']) else ''},"
                            f"{prof['total_gradient'][i] if i<len(prof['total_gradient']) else ''}\n")
        QMessageBox.information(self, "Exported", f"All sections saved to {path}")

    def _open_sensitivity(self):
        dlg = SensitivityPanel(self.state, self)
        dlg.exec()

# ─────────────────────────────────────────────────────────────────────────────
#  HOME SCREEN
# ─────────────────────────────────────────────────────────────────────────────
class HomeCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, icon, title, description, is_primary=False, is_disabled=False, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.is_primary = is_primary
        self.is_disabled = is_disabled
        self.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor if is_disabled else Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(130 if is_primary else 110)

        if is_primary:
            self.setStyleSheet(f"""
                QFrame#card {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {BLUE}, stop:1 #0D47A1);
                    border: none;
                    border-radius: 12px;
                }}
            """)
        elif is_disabled:
            self.setStyleSheet(f"""
                QFrame#card {{
                    background-color: #F5F5F5;
                    border: 1.5px dashed #CFD8DC;
                    border-radius: 10px;
                }}
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)

        top_row = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 24px;")
        top_row.addWidget(icon_lbl)

        if is_disabled:
            soon_chip = make_chip("Coming Soon", "chip_gray")
            top_row.addStretch()
            top_row.addWidget(soon_chip)
        elif is_primary:
            top_row.addStretch()

        layout.addLayout(top_row)

        title_lbl = QLabel(title)
        txt_color = WHITE if is_primary else (SLATE if is_disabled else NAVY)
        title_lbl.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {txt_color}; background: transparent;")
        layout.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_color = BLUE_L if is_primary else ("#B0BEC5" if is_disabled else SLATE)
        desc_lbl.setStyleSheet(f"font-size: 11px; color: {desc_color}; background: transparent;")
        desc_lbl.setWordWrap(True)
        layout.addWidget(desc_lbl)

        self.badge = None

    def add_badge(self, text, style="chip_gray"):
        if not self.badge:
            self.badge = make_chip(text, style)
            # Add to layout
            self.layout().addWidget(self.badge, alignment=Qt.AlignmentFlag.AlignRight)
        else:
            self.badge.setText(text)
            self.badge.setObjectName(style)
            self.badge.setStyle(self.badge.style())

    def mousePressEvent(self, event):
        if not self.is_disabled:
            self.clicked.emit()

class HomeScreen(QWidget):
    open_ipr = pyqtSignal()
    open_pvt = pyqtSignal()
    open_vlp = pyqtSignal()
    open_sensitivity = pyqtSignal()
    open_nodal = pyqtSignal()
    open_calibration = pyqtSignal()
    open_gaslift = pyqtSignal()
    open_choke = pyqtSignal()  # PRD §6.1 — Choke card
    well_info_saved = pyqtSignal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._setup_ui()
        self.refresh_badges()

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(32, 24, 32, 24)
        main.setSpacing(20)

        # Header
        header = QHBoxLayout()
        title_lbl = QLabel("PROPMAS")
        title_lbl.setStyleSheet(f"font-size: 28px; font-weight: 700; color: {NAVY};")
        subtitle_lbl = QLabel("Analysis - Modelling - Production")
        subtitle_lbl.setStyleSheet(f"font-size: 13px; color: {SLATE};")
        header_left = QVBoxLayout()
        header_left.addWidget(title_lbl)
        header_left.addWidget(subtitle_lbl)
        header.addLayout(header_left)
        header.addStretch()

        # Reset all
        reset_btn = QPushButton("⟳ Reset All Inputs")
        reset_btn.setObjectName("secondary")
        reset_btn.clicked.connect(self._reset_all)
        header.addWidget(reset_btn)
        main.addLayout(header)

        # Well Info
        well_info_frame = QFrame(); well_info_frame.setObjectName("card")
        well_info_l = QHBoxLayout(well_info_frame)
        well_info_l.setSpacing(10)
        well_info_l.addWidget(make_label("Well ID:", style=""))
        self.well_id_edit = QLineEdit()
        self.well_id_edit.textChanged.connect(lambda t: setattr(self.state, 'well_id', t))
        well_info_l.addWidget(self.well_id_edit)
        well_info_l.addWidget(make_label("Field Name:", style=""))
        self.field_name_edit = QLineEdit()
        self.field_name_edit.textChanged.connect(lambda t: setattr(self.state, 'field_name', t))
        well_info_l.addWidget(self.field_name_edit)
        save_info_btn = QPushButton("Save Info"); save_info_btn.setObjectName("secondary")
        # save_info_btn.clicked.connect(self.state.save)
        save_info_btn.clicked.connect(self._save_well_info)

        well_info_l.addWidget(save_info_btn)
        main.addWidget(well_info_frame)

        # Readiness progress bar
        prog_frame = QFrame(); prog_frame.setObjectName("card")
        prog_l = QVBoxLayout(prog_frame); prog_l.setContentsMargins(16, 12, 16, 12)
        prog_label = QLabel("Workflow Readiness")
        prog_label.setStyleSheet(f"font-weight:700; color:{NAVY}; font-size:12px;")
        prog_l.addWidget(prog_label)

        steps_row = QHBoxLayout()
        self.step_labels = {}
        for step, key in [("IPR Data", "ipr"), ("PVT Data", "pvt"), ("VLP Data", "vlp"), ("Nodal Ready", "nodal")]:
            lbl = make_chip(f"○ {step}", "chip_gray")
            self.step_labels[key] = lbl
            steps_row.addWidget(lbl)
            if key != "nodal":
                arr = QLabel("→"); arr.setStyleSheet(f"color:{BLUE_M}; font-size:14px;"); steps_row.addWidget(arr)
        prog_l.addLayout(steps_row)
        main.addWidget(prog_frame)

        # Card grid: row 1 — data panels (IPR, PVT, VLP, Sensitivity, Choke)
        row1 = QHBoxLayout(); row1.setSpacing(14)
        self.ipr_card = HomeCard("📉", "IPR Data",
            "Inflow Performance Relationship — reservoir model & test data")
        self.pvt_card = HomeCard("🧪", "PVT Data",
            "Black-Oil fluid properties — composition & PVT correlations")
        self.vlp_card = HomeCard("🔧", "VLP Data",
            "Wellbore geometry & vertical lift performance")
        self.sens_card = HomeCard("📊", "Sensitivity Analysis",
            "Vary up to 3 parameters — rate & pressure impact")
        self.choke_card = HomeCard("🎛️", "Choke",
            "Surface choke sizing & rate-check — optimum bean size")  # PRD §6.1
        row1.addWidget(self.ipr_card)
        row1.addWidget(self.pvt_card)
        row1.addWidget(self.vlp_card)
        row1.addWidget(self.sens_card)
        row1.addWidget(self.choke_card)
        main.addLayout(row1)

        # Row 2: Nodal (primary CTA) + Calibration + Gas Lift
        row2 = QHBoxLayout(); row2.setSpacing(14)
        self.nodal_card = HomeCard("▶", "Nodal Analysis",
            "IPR × VLP intersection — find the well operating point", is_primary=True)
        self.nodal_card.setFixedHeight(140)

        self.calib_card = HomeCard("📌", "Calibration",
            "Match VLP model to measured pressure-survey data", is_disabled=False)
        self.gaslift_card = HomeCard("⛽", "Gas Lift Analysis",
            "Gas lift performance & injection optimization", is_disabled=False)  # PRD §6.1 — fully implemented

        row2.addWidget(self.nodal_card, 2)
        row2.addWidget(self.calib_card, 1)
        row2.addWidget(self.gaslift_card, 1)
        main.addLayout(row2)
        main.addStretch()

        # Status strip
        status_frame = QFrame()
        status_frame.setStyleSheet(f"background:{OFF_W}; border-top:1px solid {BLUE_M};")
        status_l = QHBoxLayout(status_frame)
        status_l.setContentsMargins(16, 8, 16, 8)
        self.status_lbl = QLabel("Complete IPR, PVT, and VLP panels before running Nodal Analysis.")
        self.status_lbl.setStyleSheet(f"font-size:11px; color:{SLATE};")
        status_l.addWidget(self.status_lbl)
        main.addWidget(status_frame)

        # Add About Us button to the status bar
        status_l.addStretch()
        about_btn = QPushButton("About Us")
        about_btn.setObjectName("secondary")
        about_btn.setToolTip("Show information about the application and developer.")
        about_btn.clicked.connect(self._show_about_dialog)
        status_l.addWidget(about_btn)



        # Connect signals
        self.ipr_card.clicked.connect(self.open_ipr.emit)
        self.pvt_card.clicked.connect(self.open_pvt.emit)
        self.vlp_card.clicked.connect(self.open_vlp.emit)
        self.sens_card.clicked.connect(self.open_sensitivity.emit)
        self.choke_card.clicked.connect(self.open_choke.emit)  # PRD §6.1
        self.nodal_card.clicked.connect(self._on_nodal_click)
        self.calib_card.clicked.connect(self.open_calibration.emit)
        self.gaslift_card.clicked.connect(self.open_gaslift.emit)

        self.well_id_edit.setText(self.state.well_id or "")
        self.field_name_edit.setText(self.state.field_name or "")

    def _show_about_dialog(self):
        """Displays the 'About Us' information dialog."""
        about_text = (
            "Hello, I am Nob007 (Soumik Dutta), this software is an open-source implementation of "
            "Nodal Analysis and Gas Lift Design, 'PROPMAS' - Petroleum Production Modelling and Analysis System.\n\n"
            "Feel free to report any issues and thank you for using the software!"
        )
        QMessageBox.information(self, "About PROPMAS", about_text)


    def _on_nodal_click(self):
        if self.state.nodal_ready:
            self.open_nodal.emit()
        else:
            missing = []
            if not self.state.ipr_complete: missing.append("IPR")
            if not self.state.pvt_complete: missing.append("PVT")
            if not self.state.vlp_complete: missing.append("VLP")

            QMessageBox.information(
                self, "Incomplete Data",
                f"Please complete these panels first: {', '.join(missing)}\n\n")


    def refresh_badges(self):
        s = self.state
        # IPR / PVT / VLP completion badges
        for card, complete in [
            (self.ipr_card, s.ipr_complete),
            (self.pvt_card, s.pvt_complete),
            (self.vlp_card, s.vlp_complete),
        ]:
            if complete:
                card.add_badge("✓ Saved", "chip_success")
            else:
                card.add_badge("○ Not saved", "chip_gray")

        # PRD §6.1 — Calibration "Active" indicator
        calib_active = s.calib_holdup_factor != 1.0 or s.calib_friction_factor != 1.0
        if calib_active:
            self.calib_card.add_badge("⚡ Active", "chip_success")
        else:
            self.calib_card.add_badge("○ Not applied", "chip_gray")

        # PRD §6.1 — Gas Lift "Active" indicator
        if getattr(s, "gl_applied", False):
            self.gaslift_card.add_badge("⚡ Active", "chip_success")
        else:
            self.gaslift_card.add_badge("○ Not applied", "chip_gray")

        # Readiness progress pills
        for key, complete in [("ipr", s.ipr_complete), ("pvt", s.pvt_complete),
                               ("vlp", s.vlp_complete), ("nodal", s.nodal_ready)]:
            lbl = self.step_labels[key]
            label_text = {"ipr": "IPR Data", "pvt": "PVT Data",
                          "vlp": "VLP Data", "nodal": "Nodal Ready"}[key]
            if complete:
                lbl.setText(f"✓ {label_text}"); lbl.setObjectName("chip_success")
            else:
                lbl.setText(f"○ {label_text}"); lbl.setObjectName("chip_gray")
            lbl.setStyle(lbl.style())

        parts = []
        if s.ipr_complete: parts.append("IPR ✓")
        if s.pvt_complete: parts.append("PVT ✓")
        if s.vlp_complete: parts.append("VLP ✓")
        missing = []
        if not s.ipr_complete: missing.append("IPR (missing fields)")
        if not s.pvt_complete: missing.append("PVT (not saved)")
        if not s.vlp_complete: missing.append("VLP (missing fields)")
        if missing:
            self.status_lbl.setText(f"Status: {' · '.join(parts)} — Still needed: {', '.join(missing)}")
        else:
            self.status_lbl.setText("✅ All data complete — Ready to run Nodal Analysis!")
            self.status_lbl.setStyleSheet(f"font-size:11px; color:{SUCCESS}; font-weight:700;")

    def _reset_all(self):
        confirm = QMessageBox.question(
            self, "Reset All Inputs",
            "This will clear ALL saved inputs and session data. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            new_state = AppState()
            # Copy reference attributes
            for attr in vars(new_state):
                setattr(self.state, attr, getattr(new_state, attr))
            try:
                os.remove(SESSION_FILE)
            except Exception:
                pass
            self.refresh_badges()
            self.well_id_edit.setText(self.state.well_id or "")
            self.field_name_edit.setText(self.state.field_name or "")

    def _save_well_info(self):
        self.state.save()
        self.well_info_saved.emit()
        QMessageBox.information(self, "Saved", "Well information has been saved for this session.")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.state.load()
        self.setWindowTitle("PROPMAS - Analysis - Modelling - Production")
        self.setMinimumSize(1200, 800)
        self._setup_ui()

    def _setup_ui(self):
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_screen = HomeScreen(self.state)
        self.nodal_screen = NodalAnalysisScreen(self.state)

        self.stack.addWidget(self.home_screen)    # index 0
        self.stack.addWidget(self.nodal_screen)   # index 1

        # Home → panels
        self.home_screen.open_ipr.connect(self._open_ipr)
        self.home_screen.open_pvt.connect(self._open_pvt)
        self.home_screen.open_vlp.connect(self._open_vlp)
        self.home_screen.open_sensitivity.connect(self._open_sensitivity)
        self.home_screen.open_nodal.connect(self._goto_nodal)
        self.home_screen.open_calibration.connect(self._open_calibration)
        self.home_screen.open_gaslift.connect(self._open_gaslift)
        self.home_screen.open_choke.connect(self._open_choke)  # PRD §6.1
        self.home_screen.well_info_saved.connect(self._schedule_nav_update)

        # Nodal → home
        self.nodal_screen.back_requested.connect(self._goto_home)

        self.stack.setCurrentIndex(0)

        # Navbar
        nav = QWidget()
        nav.setFixedHeight(44)
        nav.setStyleSheet(f"background:{NAVY};")
        nav_l = QHBoxLayout(nav)
        nav_l.setContentsMargins(16, 0, 16, 0)
        app_name = QLabel("PROPMAS")
        app_name.setStyleSheet(f"color:{WHITE}; font-size:16px; font-weight:700;")
        nav_l.addWidget(app_name)
        nav_l.addStretch()
        self.nav_well = QLabel(f"Well: {self.state.well_id or 'Untitled'}")
        self.nav_well.setStyleSheet(f"color:{BLUE_L}; font-size:12px;")
        nav_l.addWidget(self.nav_well)

        container = QWidget()
        cont_l = QVBoxLayout(container)
        cont_l.setContentsMargins(0, 0, 0, 0)
        cont_l.setSpacing(0)
        cont_l.addWidget(nav)
        cont_l.addWidget(self.stack, 1)
        self.setCentralWidget(container)

        self.state_change_timer = QTimer(self)
        self.state_change_timer.setSingleShot(True)
        self.state_change_timer.timeout.connect(self._update_nav_well)

    def _goto_home(self):
        self.home_screen.refresh_badges()
        self.stack.setCurrentIndex(0)

    def _goto_nodal(self):
        self.stack.setCurrentIndex(1)
        self.nodal_screen._run()

    def _open_ipr(self):
        dlg = IPRPanel(self.state, self)
        dlg.applied.connect(self.home_screen.refresh_badges)
        dlg.applied.connect(self._schedule_nav_update)
        dlg.exec()

    def _open_pvt(self):
        dlg = PVTPanel(self.state, self)
        dlg.applied.connect(self.home_screen.refresh_badges)
        dlg.applied.connect(self._schedule_nav_update)
        dlg.exec()

    def _open_vlp(self):
        dlg = VLPPanel(self.state, self)
        dlg.applied.connect(self.home_screen.refresh_badges)
        dlg.applied.connect(self._schedule_nav_update)
        dlg.exec()

    def _open_sensitivity(self):
        if not self.state.ipr_complete or not self.state.vlp_complete:
            QMessageBox.information(
                self, "Setup Required",
                "Please complete and save the IPR, PVT, and VLP panels before running Sensitivity."
            )
            return
        dlg = SensitivityPanel(self.state, self)
        dlg.exec()

    def _open_calibration(self):
        if not self.state.vlp_complete:
            QMessageBox.information(
                self, "Setup Required", "Please complete the VLP panel before running Calibration."
            )
            return
        dlg = CalibrationPanel(self.state, self)
        dlg.factors_applied.connect(self.home_screen.refresh_badges)
        dlg.exec()

    def _open_gaslift(self):
        if not self.state.vlp_complete or not self.state.ipr_complete:
            QMessageBox.information(
                self, "Setup Required",
                "Please complete and save IPR, PVT, and VLP panels before running Gas Lift Analysis."
            )
            return
        dlg = GasLiftPanel(self.state, self)
        dlg.design_applied.connect(self.home_screen.refresh_badges)
        dlg.design_reset.connect(self.home_screen.refresh_badges)
        dlg.exec()

    def _open_choke(self):
        dlg = ChokePanel(self.state, self)
        dlg.exec()

    def _schedule_nav_update(self):
        self.state_change_timer.start(100)

    def _update_nav_well(self):
        self.nav_well.setText(f"Well: {self.state.well_id or 'Untitled'}")

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    app.setApplicationName("PROPMAS")
    app.setOrganizationName("Nob007 (Soumik Dutta)")

    # High-DPI
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    except Exception:
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

# Developed by: Nob007 (Soumik Dutta)
# This application is for educational and demonstrative purposes.
# Always validate results with commercial software and engineering judgment.
