"""
gui/main.py
===========
IPR × VLP Nodal Analysis — PyQt6 Desktop Application
White + Blue Accent Theme | Spec v1.0 | Production-Ready

Layout (spec §2):
    TitleBar (56 px)
    ├── QSplitter (Horizontal)
    │   ├── Sidebar (QScrollArea, 320 px)  — 5 collapsible sections (§3)
    │   └── ChartWidget                    — matplotlib canvas + legend + mini-map (§5, §7)
    └── QStatusBar (48 px)                 — operating-point result (§6)
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from scipy.optimize import brentq

# ── path so imports from core/ work regardless of cwd ─────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.pvt import BlackOilPVT
from core.ipr import composite_ipr
from core.ipr import darcy_ipr
from core.ipr import vogel_ipr
from core.vlp import HagedornBrown

from core.solver_other import find_operating_points

# ── Qt ────────────────────────────────────────────────────────────────────────
from PyQt6.QtCore import (
    QObject, QPropertyAnimation, QRunnable, QSize, Qt,
    QThreadPool, pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import QColor, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QListView, QMainWindow, QPushButton, QScrollArea, QSizePolicy, # <-- Added QListView
    QSpinBox, QSplitter, QStatusBar, QToolButton, QVBoxLayout,
    QWidget,
)

# ── matplotlib ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

# ══════════════════════════════════════════════════════════════════════════════
#  §14  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

C_NAVY       = "#0D2B55"
C_BLUE       = "#1565C0"
C_BLUE_HOV   = "#1976D2"
C_BLUE_LIGHT = "#E3F0FB"
C_BLUE_MID   = "#90CAF9"
C_WHITE      = "#FFFFFF"
C_OFF_WHITE  = "#F7FAFD"
C_INK        = "#1A2840"
C_SLATE      = "#4A6080"
C_RED        = "#E53935"
C_GOLD       = "#F9A825"
C_BORDER     = "#C2D6EC"

# ══════════════════════════════════════════════════════════════════════════════
#  §8  QSS STYLESHEET
# ══════════════════════════════════════════════════════════════════════════════

QSS = f"""
/* ── Global ────────────────────────────────────────────── */
QWidget {{
    background: {C_WHITE};
    color: {C_INK};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 12px;
}}

/* ── Sidebar scroll area ────────────────────────────────── */
QScrollArea {{ border: none; background: {C_OFF_WHITE}; }}
QScrollBar:vertical {{
    width: 6px; background: #F0F4F8; border-radius: 3px;
}}
QScrollBar::handle:vertical {{ background: {C_BLUE_MID}; border-radius: 3px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Collapsible section headers ────────────────────────── */
QToolButton#section_header {{
    background: {C_BLUE_LIGHT}; color: {C_NAVY};
    border: none; border-left: 3px solid {C_BLUE};
    padding: 8px 12px; font-weight: 600;
    text-align: left; border-radius: 0px;
}}
QToolButton#section_header:hover {{ background: #BBDEFB; }}

/* ── Input widgets ──────────────────────────────────────── */
QDoubleSpinBox, QSpinBox, QComboBox {{
    border: 1px solid {C_BORDER}; border-radius: 5px;
    padding: 4px 8px; background: {C_WHITE};
    color: {C_INK};
    selection-background-color: {C_BLUE};
    selection-color: {C_WHITE};
    min-height: 26px;
}}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
    border: 1.5px solid {C_BLUE};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView, QComboBox QListView {{
    border: 1px solid {C_BORDER};
    background: {C_WHITE};
    color: {C_INK};
    selection-background-color: {C_BLUE_LIGHT};
    selection-color: {C_NAVY};
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    min-height: 24px;
    color: {C_INK};
}}
QComboBox QAbstractItemView::item:selected {{
    background: {C_BLUE_LIGHT};
    color: {C_NAVY};
}}

/* ── Read-only output labels ─────────────────────────────── */
QLabel#ro_label {{
    color: {C_BLUE}; font-weight: 600;
    background: {C_BLUE_LIGHT}; border: 1px solid {C_BORDER};
    border-radius: 5px; padding: 4px 8px; min-height: 26px;
}}
QLabel#field_label {{ color: {C_SLATE}; font-size: 11px; }}

/* ── Run button ─────────────────────────────────────────── */
QPushButton#run_btn {{
    background: {C_BLUE}; color: {C_WHITE};
    border: none; border-radius: 6px;
    padding: 6px 18px; font-weight: 700; font-size: 13px;
    min-width: 140px; min-height: 34px;
}}
QPushButton#run_btn:hover   {{ background: {C_BLUE_HOV}; }}
QPushButton#run_btn:pressed {{ background: #0D47A1; }}
QPushButton#run_btn:disabled{{ background: {C_BLUE_MID}; color: #FFFFFF; }}

/* ── Sensitivity buttons ─────────────────────────────────── */
QPushButton#sens_btn {{
    background: {C_BLUE_LIGHT}; color: {C_BLUE};
    border: 1px solid {C_BLUE_MID}; border-radius: 5px;
    padding: 5px 12px; font-weight: 600;
}}
QPushButton#sens_btn:hover    {{ background: #BBDEFB; }}
QPushButton#sens_btn:disabled {{ color: {C_SLATE}; }}

/* ── Separator line ──────────────────────────────────────── */
QFrame#sep {{ background: {C_BLUE_MID}; max-height: 1px; border: none; }}

/* ── Status bar ──────────────────────────────────────────── */
QStatusBar {{
    background: {C_BLUE_LIGHT}; color: {C_INK};
    border-top: 1px solid {C_BLUE_MID}; font-size: 12px;
    min-height: 48px;
}}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  SENSITIVITY PALETTE  (blue → teal, 10-step sequential)
# ══════════════════════════════════════════════════════════════════════════════

SENS_PALETTE = [
    "#42A5F5", "#29B6F6", "#26C6DA", "#00ACC1", "#00838F",
    "#0097A7", "#006064", "#4DD0E1", "#80DEEA", "#B2EBF2",
]

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _dspin(val: float, lo: float, hi: float, step: float, dec: int) -> QDoubleSpinBox:
    """Convenience constructor for QDoubleSpinBox."""
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(dec)
    return s


def _row(label: str, widget: QWidget, unit: str = "") -> QWidget:
    """Build a horizontal form row: [label] [widget] [unit]."""
    w = QWidget()
    w.setStyleSheet("background: transparent;")
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 2, 0, 2)
    h.setSpacing(6)

    lbl = QLabel(label)
    lbl.setObjectName("field_label")
    lbl.setFixedWidth(92)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    h.addWidget(lbl)
    h.addWidget(widget, 1)

    if unit:
        u = QLabel(unit)
        u.setStyleSheet(
            f"color:{C_SLATE}; font-size:10px; background:transparent; min-width:54px;")
        h.addWidget(u)

    return w


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE HELPERS  (wrap core/ functions safely)
# ══════════════════════════════════════════════════════════════════════════════

def _build_pvt_vlp(p: dict) -> tuple[BlackOilPVT, HagedornBrown]:
    """Construct a fresh PVT + HagedornBrown pair from a parameters dict."""
    # wc  = p["wor"] / (1.0 + p["wor"]) if p["wor"] > 0 else 0.0
    pvt = BlackOilPVT(
        sg_gas=p["sg_gas"], sg_oil=p["sg_oil"],
        sg_water=p["sg_water"], watercut=p["wc"]
    )
    # Initial fluid properties at tubing head conditions
    fp0 = pvt.fluid_properties_dict(p["thp"], p["T_surface"], p["gor"])
    if p["model"] == "Hagedron-Brown":
        vlp = HagedornBrown(
            tubing_id=p["tubing_id"], tubing_od=p["tubing_od"],
            casing_id=p["casing_id"], roughness=p["roughness"],
            pvt_model=pvt, fluid_properties=fp0,
            watercut=p["wc"], theta=p["theta"],
        )
    # vlp = HagedornBrown(
    #     tubing_id=p["tubing_id"], tubing_od=p["tubing_od"],
    #     casing_id=p["casing_id"], roughness=p["roughness"],
    #     pvt_model=pvt, fluid_properties=fp0,
    #     watercut=p["wc"], theta=p["theta"],
    # )
    return pvt, vlp


def _vlp_pwf(vlp: HagedornBrown, p: dict, q: float) -> float:
    """Run a pressure traverse and return the bottomhole pressure (psia)."""
    _, pressures = vlp.calculate_pressure_traverse(
        Pth=p["thp"],
        surface_temp=p["T_surface"],
        bottomhole_temp=p["T_bh"],
        total_depth=p["depth"],
        step_size=p["dz_step"],
        Ql=q,
    )
    return float(pressures[-1])


def _build_ipr(p: dict) -> composite_ipr:
    """Return the correct IPR model based on model selection."""
    model = p.get("ipr_model", "Composite")
    Pr, Pb = p["Pr"], p["Pb"]
    qt, pwft = p["Qo_test"], p["Pwf_test"]

    if model == "Darcy":
        # Force Pb >> Pr so the entire curve stays in the linear Darcy region
        return darcy_ipr(Pr=Pr, Pb=Pr * 1.001, q_test=qt, Pwf_test=pwft)
    elif model == "Composite":
        # Pure Vogel: set Pb = Pr so the entire curve uses the Vogel equation
        return composite_ipr(Pr=Pr, Pb=Pr, q_test=qt, Pwf_test=pwft)
    elif model == "Vogel":
        return vogel_ipr(Pr=Pr, Pb=Pb, q_test=qt, Pwf_test=pwft)
    else:
        # "Composite Vogel-Darcy" and "Fetkovitch" both use composite IPR
        return composite_ipr(Pr=Pr, Pb=Pb, q_test=qt, Pwf_test=pwft)


# ══════════════════════════════════════════════════════════════════════════════
#  §12  WORKER THREADS
# ══════════════════════════════════════════════════════════════════════════════

class _Signals(QObject):
    result = pyqtSignal(object)
    error  = pyqtSignal(str)
    status = pyqtSignal(str)


class AnalysisWorker(QRunnable):
    """Computes IPR curve, VLP curve, operating point, and pressure traverse."""

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.p = params
        self.signals = _Signals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            p = self.p
            self.signals.status.emit("⏳  Building IPR curve…")

            # ── IPR ──────────────────────────────────────────────────────────
            ipr = _build_ipr(p)
            rates_ipr = np.linspace(0.0, ipr.q_max, 200)
            pwf_ipr   = [ipr.calculate_Pwf(float(q)) for q in rates_ipr]

            # ── VLP sweep ────────────────────────────────────────────────────
            self.signals.status.emit("⏳  Building VLP curve…")
            _, vlp = _build_pvt_vlp(p)

            n_vlp     = max(int((p["q_max"] - p["q_min"]) / p["q_step"]) + 1, 20)
            rates_vlp = np.linspace(p["q_min"], p["q_max"], n_vlp)
            pwf_vlp   = []
            for i, q in enumerate(rates_vlp):
                if i % 5 == 0:
                    self.signals.status.emit(
                        f"⏳  VLP sweep {i + 1}/{n_vlp}  (q = {q:.0f} STB/d)…")
                pwf_vlp.append(_vlp_pwf(vlp, p, float(q)))

            # ── Find operating point ─────────────────────────────────────────
            self.signals.status.emit("⏳  Finding operating points…")
            sol: dict = {
                "success": False, "operating_rate": None,
                "operating_pwf": None, "message": "", "all_points": []
            }
            traverse_depths: list = []
            traverse_pressures: list = []

            q_lo, q_hi = float(p["q_min"]), float(p["q_max"])

            try:
                vlp_params_dict = {
                    "Pth": p["thp"],
                    "surface_temp": p["T_surface"],
                    "bottomhole_temp": p["T_bh"],
                    "depth": p["depth"],
                    "step_size": p["dz_step"]
                }
                
                nodal_result = find_operating_points(
                    ipr_model=ipr,
                    vlp_model=vlp,
                    vlp_params=vlp_params_dict,
                    pr=p["Pr"],
                    q_min=q_lo,
                    q_max=q_hi,
                    xtol=0.1
                )

                if not nodal_result.success:
                    sol["message"] = nodal_result.failure_reason
                else:
                    op_point = nodal_result.stable_point
                    if op_point is None:
                        op_point = nodal_result.unstable_point
                    if op_point is None and nodal_result.all_points:
                        op_point = nodal_result.all_points[0]

                    if op_point:
                        q_star = op_point.rate
                        p_star = op_point.pwf
                        sol.update(
                            success=True, 
                            operating_rate=q_star,
                            operating_pwf=p_star, 
                            message="Converged.",
                            all_points=[(pt.rate, pt.pwf, pt.stability.value) for pt in nodal_result.all_points]
                        )

                        # Pressure traverse at operating rate (for mini-map)
                        self.signals.status.emit(
                            f"⏳  Computing pressure traverse at q* = {q_star:.1f} STB/d…")
                        traverse_depths, traverse_pressures = vlp.calculate_pressure_traverse(
                            Pth=p["thp"], surface_temp=p["T_surface"],
                            bottomhole_temp=p["T_bh"], total_depth=p["depth"],
                            step_size=p["dz_step"], Ql=q_star,
                        )
            except Exception as solve_err:
                sol["message"] = str(solve_err)

            self.signals.result.emit({
                "rates_ipr":          rates_ipr.tolist(),
                "pwf_ipr":            pwf_ipr,
                "rates_vlp":          rates_vlp.tolist(),
                "pwf_vlp":            pwf_vlp,
                "sol":                sol,
                "traverse_depths":    list(traverse_depths),
                "traverse_pressures": list(traverse_pressures),
            })

        except Exception:
            self.signals.error.emit(traceback.format_exc())


class SensWorker(QRunnable):
    """Sweeps one VLP parameter to overlay multiple curves."""

    _KEY_MAP = {
        "GOR": "gor", "THP": "thp", "Depth": "depth",
        "WOR": "wor", "Tubing ID": "tubing_id",
    }

    def __init__(self, params: dict, values: list[float], param_name: str) -> None:
        super().__init__()
        self.p, self.values, self.pname = params, values, param_name
        self.signals = _Signals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            key = self._KEY_MAP[self.pname]
            results = []
            total = len(self.values)

            for i, val in enumerate(self.values):
                self.signals.status.emit(
                    f"⏳  Sensitivity {i + 1}/{total} — {self.pname} = {val:.2f}…")
                p2 = dict(self.p)
                p2[key] = val

                _, vlp = _build_pvt_vlp(p2)
                n = max(int((p2["q_max"] - p2["q_min"]) / p2["q_step"]) + 1, 20)
                rates = np.linspace(p2["q_min"], p2["q_max"], n)
                pwfs  = [_vlp_pwf(vlp, p2, float(q)) for q in rates]

                results.append({
                    "label": f"{self.pname} = {val:.2f}",
                    "rates": rates.tolist(),
                    "pwf":   pwfs,
                })

            self.signals.result.emit(results)

        except Exception:
            self.signals.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  COLLAPSIBLE SECTION  (spec §3 — reusable widget)
# ══════════════════════════════════════════════════════════════════════════════

class CollapsibleSection(QWidget):
    """QToolButton header that toggles a body QWidget (▼ / ▶)."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._open = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header toggle button
        self.btn = QToolButton()
        self.btn.setObjectName("section_header")
        self.btn.setCheckable(True)
        self.btn.setChecked(True)
        self.btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn.setMinimumHeight(38)
        self._title = title
        self._refresh()
        self.btn.clicked.connect(self._toggle)

        # Body container
        self.body = QWidget()
        self.body.setStyleSheet("background: #FFFFFF;")
        self.bl = QVBoxLayout(self.body)
        self.bl.setContentsMargins(12, 8, 12, 10)
        self.bl.setSpacing(5)

        # 1-px separator below the body
        sep = QFrame()
        sep.setObjectName("sep")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)

        root.addWidget(self.btn)
        root.addWidget(self.body)
        root.addWidget(sep)

    def _refresh(self) -> None:
        arrow = "▼" if self._open else "▶"
        self.btn.setText(f"  {arrow}  {self._title}")

    def _toggle(self) -> None:
        self._open = not self._open
        self.body.setVisible(self._open)
        self._refresh()

    def add(self, widget: QWidget) -> None:
        self.bl.addWidget(widget)


# ══════════════════════════════════════════════════════════════════════════════
#  §7  MINI TRAVERSE MAP  (signature feature)
# ══════════════════════════════════════════════════════════════════════════════

class MiniMap(QFrame):
    """
    Semi-transparent 250×310 px pressure traverse mini-plot.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(QSize(250, 310))
        self.setStyleSheet(f"""
            QFrame {{
                background: rgba(255, 255, 255, 235);
                border: 1px solid {C_BLUE_MID};
                border-radius: 8px;
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        # Header row: title
        hdr = QHBoxLayout()
        title_lbl = QLabel("Traverse @ q*")
        title_lbl.setStyleSheet(
            f"font-size: 9pt; font-weight: 700; color: {C_BLUE}; "
            "background: transparent; border: none;")
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        lay.addLayout(hdr)

        # Mini matplotlib figure
        self.fig = Figure(figsize=(2.4, 2.8), dpi=80)
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_subplot(111)
        self._blank()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background: white; border: none;")
        lay.addWidget(self.canvas, 1)

    # ── public ─────────────────────────────────────────────────────────────

    def update_traverse(self, depths: list, pressures: list) -> None:
        """Redraw with actual traverse data."""
        ax = self.ax
        ax.clear()
        ax.set_facecolor(C_OFF_WHITE)
        ax.plot(pressures, depths, color=C_RED, linewidth=1.5)
        ax.invert_yaxis()
        ax.set_xlabel("P (psia)", fontsize=7, color=C_INK)
        ax.set_ylabel("Depth (ft)", fontsize=7, color=C_INK)
        ax.tick_params(labelsize=6, colors=C_INK)
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.5, color=C_BLUE_MID)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BORDER)
            sp.set_linewidth(0.5)
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def reset(self) -> None:
        """Restore placeholder state."""
        self._blank()
        self.canvas.draw_idle()

    # ── private ─────────────────────────────────────────────────────────────

    def _blank(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor(C_OFF_WHITE)
        self.ax.text(
            0.5, 0.5, "Run analysis\nto see traverse.",
            ha="center", va="center", transform=self.ax.transAxes,
            fontsize=8, color=C_SLATE)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout(pad=0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  §5  CHART WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class ChartWidget(QWidget):
    """
    Right pane: matplotlib plot canvas + legend bar + floating mini-map.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(600)
        self._sens_lines: list = []
        self._ipr_q: list = []
        self._ipr_p: list = []
        self._vlp_q: list = []
        self._vlp_p: list = []
        self._ipr_line = None
        self._vlp_line = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 0)
        root.setSpacing(0)

        # ── matplotlib figure ───────────────────────────────────────────────
        self.fig = Figure(facecolor=C_WHITE)
        self.ax  = self.fig.add_subplot(111)
        self._init_ax()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background: {C_WHITE};")

        # ── navigation toolbar ──────────────────────────────────────────────
        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(
            f"QToolBar {{ background: {C_OFF_WHITE}; border-bottom: 1px solid {C_BORDER}; }}"
            "QToolButton { background: transparent; border-radius: 4px; padding: 2px; }"
            f"QToolButton:hover {{ background: {C_BLUE_LIGHT}; }}")

        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, 1)

        # ── §5.2 legend bar ─────────────────────────────────────────────────
        self.legend_bar = QFrame()
        self.legend_bar.setFixedHeight(40)
        self.legend_bar.setStyleSheet(
            f"QFrame {{ border-top: 1px solid {C_BORDER}; background: {C_OFF_WHITE}; }}")
        self._legend_hl = QHBoxLayout(self.legend_bar)
        self._legend_hl.setContentsMargins(14, 0, 14, 0)
        self._legend_hl.setSpacing(16)
        self._build_legend()
        root.addWidget(self.legend_bar)

        # ── §7 floating mini-map (overlaid, repositioned in resizeEvent) ────
        self.mini = MiniMap(self)
        self.mini.raise_()

        # ── hover tooltip ───────────────────────────────────────────────────
        self._tip = QLabel("", self.canvas)
        self._tip.setStyleSheet(
            "background: rgba(13,43,85,0.88); color: white; "
            "border-radius: 4px; padding: 3px 8px; font-size: 11px;")
        self._tip.hide()
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("axes_leave_event",
                                lambda _e: self._tip.hide())

    # ── layout ──────────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cr = self.canvas.geometry()
        mm = self.mini
        mm.move(cr.right() - mm.width() - 12,
                cr.bottom() - mm.height() - 12)

    # ── axes ────────────────────────────────────────────────────────────────

    def _init_ax(self) -> None:
        ax = self.ax
        ax.set_facecolor(C_WHITE)
        ax.set_xlabel("Liquid Rate, q  (STB/day)",
                      fontsize=11, color=C_INK, labelpad=8)
        ax.set_ylabel("Bottomhole Flowing Pressure, Pwf  (psia)",
                      fontsize=11, color=C_INK, labelpad=8)
        ax.set_title("IPR × VLP — Nodal Analysis",
                     fontsize=13, fontweight="bold", color=C_NAVY, pad=10)
        ax.grid(True, linestyle="--", alpha=0.35, color=C_BLUE_MID, linewidth=0.7)
        ax.tick_params(colors=C_INK, labelsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BORDER)
        ax.set_xlim(0, 3500)
        ax.set_ylim(0, 4000)
        self.fig.tight_layout(pad=1.8)

    # ── legend ──────────────────────────────────────────────────────────────

    def _build_legend(self, extras: list[tuple[str, str]] | None = None) -> None:
        """Rebuild the legend bar. extras = [(colour, label), …] for sensitivity."""
        # Clear old items
        for i in reversed(range(self._legend_hl.count())):
            item = self._legend_hl.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()

        items: list[tuple[str, str, bool]] = [
            (C_RED,  "IPR Curve",        False),
            (C_BLUE, "VLP Curve",        False),
            (C_GOLD, "Operating Point ★", True),
        ]
        if extras:
            items += [(c, lbl, False) for c, lbl in extras]

        for col, lbl, is_star in items:
            sw = QLabel("★" if is_star else "")
            if is_star:
                sw.setStyleSheet(
                    f"color: {col}; font-size: 14px; "
                    "background: transparent; border: none;")
            else:
                sw.setFixedSize(12, 12)
                sw.setStyleSheet(
                    f"background: {col}; border-radius: 2px; "
                    "min-height: 12px; min-width: 12px;")
            tx = QLabel(lbl)
            tx.setStyleSheet(
                f"color: {C_INK}; font-size: 11px; "
                "background: transparent; border: none;")
            self._legend_hl.addWidget(sw)
            self._legend_hl.addWidget(tx)

        self._legend_hl.addStretch()

    # ── hover tooltip ────────────────────────────────────────────────────────

    def _on_hover(self, ev) -> None:
        if ev.inaxes != self.ax:
            self._tip.hide()
            return

        if ev.x is None or ev.y is None:
            self._tip.hide()
            return

        lines = []
        if self._ipr_line: lines.append((self._ipr_line, "IPR"))
        if self._vlp_line: lines.append((self._vlp_line, "VLP"))
        for sl in self._sens_lines:
            lines.append((sl, sl.get_label()))

        best_dist = 20.0  # snapping threshold in pixels
        best_data = None
        
        for line, label in lines:
            xdata, ydata = line.get_data()
            if len(xdata) == 0: continue
            
            xy = np.column_stack((xdata, ydata))
            xy_disp = self.ax.transData.transform(xy)
            
            dx = xy_disp[:, 0] - ev.x
            dy = xy_disp[:, 1] - ev.y
            dist = np.hypot(dx, dy)
            
            idx = np.argmin(dist)
            if dist[idx] < best_dist:
                best_dist = dist[idx]
                best_data = (xdata[idx], ydata[idx], label)

        if best_data is None:
            self._tip.hide()
            return
            
        q, p, label = best_data
        self._tip.setText(f"{label}:  q = {q:.0f} STB/d  |  Pwf = {p:.0f} psia")
        self._tip.adjustSize()
        
        cx = min(int(ev.x) + 14, self.canvas.width() - self._tip.width() - 4)
        cy = max(self.canvas.height() - int(ev.y) + 4, 4)
        self._tip.move(cx, cy)
        self._tip.show()

    # ── public drawing API ───────────────────────────────────────────────────

    def plot(self, data: dict) -> None:
        """Render IPR, VLP curves and operating point from worker result dict."""
        self.ax.clear()
        self._init_ax()
        self._sens_lines.clear()

        ri, pi = data["rates_ipr"], data["pwf_ipr"]
        rv, pv = data["rates_vlp"], data["pwf_vlp"]
        sol     = data["sol"]

        self._ipr_q, self._ipr_p = ri, pi
        self._vlp_q, self._vlp_p = rv, pv

        self._ipr_line, = self.ax.plot(ri, pi, color=C_RED,  linewidth=2.5,
                                       solid_capstyle="round", label="IPR")
        self._vlp_line, = self.ax.plot(rv, pv, color=C_BLUE, linewidth=2.5,
                                       solid_capstyle="round", label="VLP")

        if sol["success"]:
            if sol.get("all_points"):
                for q_val, p_val, stab in sol["all_points"]:
                    is_stable = stab == "Stable"
                    self.ax.plot(
                        q_val, p_val,
                        marker="*" if is_stable else "o", markersize=16 if is_stable else 8,
                        color=C_GOLD if is_stable else C_RED, zorder=10,
                        linestyle="None",
                        markeredgecolor="#C67C00" if is_stable else "#B71C1C", markeredgewidth=0.8,
                    )
            else:
                qs, ps = sol["operating_rate"], sol["operating_pwf"]
                self.ax.plot(
                    qs, ps,
                    marker="*", markersize=16, color=C_GOLD, zorder=10,
                    linestyle="None",
                    markeredgecolor="#C67C00", markeredgewidth=0.8,
                )

        # Auto-scale
        all_q = ri + rv
        all_p = pi + pv
        if all_q:
            self.ax.set_xlim(0, max(all_q) * 1.06)
        if all_p:
            self.ax.set_ylim(0, max(all_p) * 1.06)

        self.canvas.draw_idle()
        self._build_legend()

        # Update mini-map if traverse was computed
        if sol["success"] and data.get("traverse_depths"):
            self.mini.update_traverse(
                data["traverse_depths"], data["traverse_pressures"])
        else:
            self.mini.reset()

    def add_sens(self, results: list[dict]) -> None:
        """Overlay sensitivity VLP curves as dashed lines."""
        extra: list[tuple[str, str]] = []
        for i, r in enumerate(results):
            col = SENS_PALETTE[i % len(SENS_PALETTE)]
            line, = self.ax.plot(
                r["rates"], r["pwf"],
                color=col, linewidth=1.5, linestyle="--", label=r["label"])
            self._sens_lines.append(line)
            extra.append((col, r["label"]))
        self.canvas.draw_idle()
        self._build_legend(extra)

    def clear_sens(self) -> None:
        """Remove all sensitivity overlay curves."""
        for ln in self._sens_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._sens_lines.clear()
        self.canvas.draw_idle()
        self._build_legend()


# ══════════════════════════════════════════════════════════════════════════════
#  §3.1  IPR SECTION
# ══════════════════════════════════════════════════════════════════════════════

class IprSection(QWidget):
    live_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Model selector
        self.model = QComboBox()
        view = QListView()
        view.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.model.setView(view)
        self.model.addItems([
            "Composite", "Vogel", "Darcy", "Fetkovitch (soon)"])
        lay.addWidget(_row("IPR Model", self.model))

        # Inputs
        self.pr  = _dspin(2500.0, 0, 20000, 50,  2)
        self.pwf = _dspin(1000.0, 0, 20000, 50,  2)
        self.pb  = _dspin(2800.0, 0, 20000, 50,  2)
        self.qt  = _dspin(500.0,  0, 50000, 50,  2)

        lay.addWidget(_row("Pr",          self.pr,  "psia"))
        self._pb_row = _row("Pb",         self.pb,  "psia")

        lay.addWidget(_row("Pwf (test)",  self.pwf, "psia"))
        self._qt_row = _row("Qo (test)",  self.qt,  "STB/d")
        lay.addWidget(self._pb_row)
        lay.addWidget(self._qt_row)

        # Read-only outputs
        self.aof_lbl = QLabel("—")
        self.aof_lbl.setObjectName("ro_label")
        self.pi_lbl  = QLabel("—")
        self.pi_lbl.setObjectName("ro_label")
        lay.addWidget(_row("AOF",    self.aof_lbl, "STB/d"))
        lay.addWidget(_row("J (PI)", self.pi_lbl,  "STB/d/psi"))

        # Signals
        for sp in [self.pr, self.pwf, self.pb, self.qt]:
            sp.valueChanged.connect(self._live)
        self.model.currentIndexChanged.connect(self._on_model)
        self._on_model()

    def _on_model(self) -> None:
        darcy = self.model.currentText() == "Darcy"
        self._pb_row.setVisible(not darcy)
        # self._qt_row.setVisible(not darcy)
        self._live()

    def _live(self) -> None:
        """Update AOF and PI live without running the full analysis."""
        try:
            if self.model.currentText() == "Composite":
                ipr = composite_ipr(
                    Pr=self.pr.value(), Pb=self.pb.value(),
                    q_test=self.qt.value(), Pwf_test=self.pwf.value())
                self.aof_lbl.setText(f"{ipr.q_max:.1f}")
                self.pi_lbl.setText(f"{ipr.J:.4f}")
            elif self.model.currentText() == "Darcy":
                ipr = darcy_ipr(
                    Pr=self.pr.value(), Pb=self.pr.value() * 1.001,
                    q_test=self.qt.value(), Pwf_test=self.pwf.value())
                self.aof_lbl.setText(f"{ipr.q_max:.1f}")
                self.pi_lbl.setText(f"{ipr.J:.4f}")
            elif self.model.currentText() == "Vogel":
                ipr = vogel_ipr(
                    Pr=self.pr.value(), Pb=self.pb.value(),
                    q_test=self.qt.value(), Pwf_test=self.pwf.value())
                self.aof_lbl.setText(f"{ipr.q_max:.1f}")
                self.pi_lbl.setText(f"{ipr.J:.4f}")
            elif self.model.currentText() == "Fetkovitch (soon)":
                self.aof_lbl.setText("—")
                self.pi_lbl.setText("—")
        except Exception:
            self.aof_lbl.setText("—")
            self.pi_lbl.setText("—")
        self.live_changed.emit()

    def values(self) -> dict:
        return {
            "Pr":        self.pr.value(),
            "Pb":        self.pb.value(),
            "Qo_test":   self.qt.value(),
            "Pwf_test":  self.pwf.value(),
            "ipr_model": self.model.currentText(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  §3.2  FLUID SECTION
# ══════════════════════════════════════════════════════════════════════════════

class FluidSection(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.sg_o = _dspin(0.82,  0.60, 1.10, 0.005, 3)
        self.sg_g = _dspin(0.915, 0.55, 1.50, 0.005, 3)
        self.sg_w = _dspin(1.07,  0.98, 1.20, 0.005, 3)
        self.wc  = _dspin(0.0,   0.0,  50.0, 0.1,   2)

        lay.addWidget(_row("SG Oil",   self.sg_o, "(w=1)"))
        lay.addWidget(_row("SG Gas",   self.sg_g, "(air=1)"))
        lay.addWidget(_row("SG Water", self.sg_w, "(w=1)"))
        lay.addWidget(_row("Water Cut", self.wc, "fraction"))

        self.api_lbl = QLabel("—")
        self.api_lbl.setObjectName("ro_label")
        lay.addWidget(_row("API Gravity", self.api_lbl, "°API"))

        self.sg_o.valueChanged.connect(self._update_api)
        self._update_api()

    def _update_api(self) -> None:
        sg = self.sg_o.value()
        self.api_lbl.setText(f"{141.5 / sg - 131.5:.1f}" if sg > 0 else "—")

    def values(self) -> dict:
        return {
            "sg_oil":   self.sg_o.value(),
            "sg_gas":   self.sg_g.value(),
            "sg_water": self.sg_w.value(),
            "wc":      self.wc.value(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  §3.3  VLP SECTION
# ══════════════════════════════════════════════════════════════════════════════

class VlpSection(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.model = QComboBox()
        view = QListView()
        view.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.model.setView(view)
        self.model.addItems([
            "Hagedron-Brown", "Duns and Ross (soon)", "Beggs and Brill (soon)"])
        lay.addWidget(_row("VLP Model", self.model))

        self.tid   = _dspin(2.441, 0.5, 5.5,    0.1,  3)
        self.tod   = _dspin(2.875, 0.5, 6,    0.1,  3)
        self.cid   = _dspin(5.5, 0.10, 20.5,    0.1,  3)
        self.rough = _dspin(0.0006, 0.0,  1,    0.0001, 5)
        self.thp   = _dspin(346.6,  0.0,  5000.0,  10.0,   1)
        self.depth = _dspin(8244.0, 100,  30000,   50.0,   1)
        self.t_sf  = _dspin(80.0,   -20,  200,     1.0,    1)
        self.t_bh  = _dspin(130.0,  50,   400,     1.0,    1)
        self.gor   = _dspin(480.0,  0,    10000,   10.0,   1)
        self.theta = _dspin(0.0,    0,    90,      1.0,    1)

        lay.addWidget(_row("Tubing ID",  self.tid,   "in"))
        lay.addWidget(_row("Tubing OD",  self.tod,   "in"))
        lay.addWidget(_row("Casing ID",  self.cid,   "in"))
        lay.addWidget(_row("Roughness",  self.rough, "in"))
        lay.addWidget(_row("THP",        self.thp,   "psia"))
        lay.addWidget(_row("TVD",        self.depth, "ft"))
        lay.addWidget(_row("T Surface",  self.t_sf,  "°F"))
        lay.addWidget(_row("T BH",       self.t_bh,  "°F"))
        lay.addWidget(_row("GOR",        self.gor,   "scf/STB"))
        lay.addWidget(_row("Deviation",  self.theta, "°"))

    def values(self) -> dict:
        return {
            "model":     self.model.currentText(),
            "tubing_id": self.tid.value()/12,
            "tubing_od": self.tod.value()/12,
            "casing_id": self.cid.value()/12,
            "roughness": self.rough.value()/12,
            "thp":       self.thp.value(),
            "depth":     self.depth.value(),
            "T_surface": self.t_sf.value(),
            "T_bh":      self.t_bh.value(),
            "gor":       self.gor.value(),
            "theta":     self.theta.value(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  §3.4  RATE RANGE SECTION
# ══════════════════════════════════════════════════════════════════════════════

class RateSection(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.qmin  = _dspin(50.0,   1,    1000,  10,  1)
        self.qmax  = _dspin(3000.0, 100, 50000, 100,  1)
        self.qstep = _dspin(50.0,   1,    500,   10,  1)
        self.dz    = _dspin(200.0,  10,  1000,   10,  1)

        lay.addWidget(_row("q min",   self.qmin,  "STB/d"))
        lay.addWidget(_row("q max",   self.qmax,  "STB/d"))
        lay.addWidget(_row("q step",  self.qstep, "STB/d"))
        lay.addWidget(_row("dz step", self.dz,    "ft"))

        # Info label
        info = QLabel(
            "Points = ⌈(q_max − q_min) / q_step⌉ + 1")
        info.setStyleSheet(
            f"color: {C_SLATE}; font-size: 10px; "
            "font-style: italic; background: transparent;")
        info.setWordWrap(True)
        lay.addWidget(info)

    def values(self) -> dict:
        return {
            "q_min":  self.qmin.value(),
            "q_max":  self.qmax.value(),
            "q_step": self.qstep.value(),
            "dz_step": self.dz.value(),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  §3.5  SENSITIVITY SECTION
# ══════════════════════════════════════════════════════════════════════════════

class SensSection(QWidget):
    do_run   = pyqtSignal(str, float, float, int)
    do_clear = pyqtSignal()

    _DEFAULTS: dict[str, tuple[float, float]] = {
        "GOR":       (200.0,  1000.0),
        "THP":       (100.0,   500.0),
        "Depth":     (5000.0, 12000.0),
        "Water Cut": (0.0,      1.0),
        "Tubing ID": (0.15,     0.30),
    }

    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.param = QComboBox()
        self.param.addItems(list(self._DEFAULTS))
        self.param.currentIndexChanged.connect(self._prefill)
        lay.addWidget(_row("Parameter", self.param))

        self.smin  = _dspin(200.0, 0, 99999, 10, 2)
        self.smax  = _dspin(1000.0, 0, 99999, 10, 2)
        self.steps = QSpinBox()
        self.steps.setRange(2, 10)
        self.steps.setValue(4)

        lay.addWidget(_row("Min value", self.smin))
        lay.addWidget(_row("Max value", self.smax))
        lay.addWidget(_row("# Curves",  self.steps))

        btn_row = QHBoxLayout()
        self.run_btn   = QPushButton("Run Sensitivity")
        self.run_btn.setObjectName("sens_btn")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("sens_btn")
        self.run_btn.clicked.connect(
            lambda: self.do_run.emit(
                self.param.currentText(),
                self.smin.value(), self.smax.value(), self.steps.value()))
        self.clear_btn.clicked.connect(self.do_clear.emit)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.clear_btn)
        lay.addLayout(btn_row)

        self._prefill()

    def _prefill(self) -> None:
        lo, hi = self._DEFAULTS.get(self.param.currentText(), (0.0, 100.0))
        self.smin.setValue(lo)
        self.smax.setValue(hi)


# ══════════════════════════════════════════════════════════════════════════════
#  §4  TITLE BAR
# ══════════════════════════════════════════════════════════════════════════════

class TitleBar(QWidget):
    run_clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(56)
        self.setStyleSheet(f"background: {C_NAVY};")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(10)

        # Icon
        ico = QLabel("🛢")
        ico.setStyleSheet(
            f"color: {C_BLUE_MID}; font-size: 22px; background: transparent;")
        lay.addWidget(ico)

        # App name
        title = QLabel("Nodal Analysis")
        title.setStyleSheet(
            "color: #FFFFFF; font-size: 16pt; font-weight: 700; "
            "background: transparent;")
        lay.addWidget(title)

        # Subtitle
        sub = QLabel("Integrated Production Modelling")
        sub.setStyleSheet(
            "color: #000000; font-size: 10pt; "
            "background: transparent; margin-left: 6px;"
            "font-weight: 900;")
        lay.addWidget(sub)
        lay.addStretch()

        # Run button (spec §4)
        self.run_btn = QPushButton("▶  Run Analysis")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self.run_clicked.emit)
        lay.addWidget(self.run_btn)


# ══════════════════════════════════════════════════════════════════════════════
#  §3  SIDEBAR  (QScrollArea containing 5 CollapsibleSections)
# ══════════════════════════════════════════════════════════════════════════════

class Sidebar(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)

        wrap = QWidget()
        wrap.setStyleSheet(f"background: {C_OFF_WHITE};")
        vbox = QVBoxLayout(wrap)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Instantiate section content widgets
        self.ipr   = IprSection()
        self.fluid = FluidSection()
        self.vlp   = VlpSection()
        self.rate  = RateSection()
        self.sens  = SensSection()

        # Wrap each in a collapsible header (spec §3 icons approximate Unicode)
        for header, body in [
            ("⬤  IPR Model",         self.ipr),
            ("◎  Fluid Properties",   self.fluid),
            ("◉  VLP — Wellbore",     self.vlp),
            ("◈  Rate Range",         self.rate),
            ("◇  Sensitivity",        self.sens),
        ]:
            sec = CollapsibleSection(header)
            sec.add(body)
            vbox.addWidget(sec)

        vbox.addStretch()
        self.setWidget(wrap)

    def all_values(self) -> dict:
        """Collect and merge all section values into one flat dict."""
        d: dict = {}
        d.update(self.ipr.values())
        d.update(self.fluid.values())
        d.update(self.vlp.values())
        d.update(self.rate.values())
        return d


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "Integrated Production Modelling — Nodal Analysis")
        self.setMinimumSize(1280, 800)
        self._pool = QThreadPool.globalInstance()

        # ── Central widget root layout ────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── §4 Title bar ─────────────────────────────────────────────────────
        self.tbar = TitleBar()
        self.tbar.run_clicked.connect(self.run_analysis)
        root.addWidget(self.tbar)

        # ── §2 Splitter (sidebar | chart) ────────────────────────────────────
        spl = QSplitter(Qt.Orientation.Horizontal)
        spl.setHandleWidth(2)
        spl.setStyleSheet(f"QSplitter::handle {{ background: {C_BORDER}; }}")
        self.sidebar = Sidebar()
        self.chart   = ChartWidget()
        spl.addWidget(self.sidebar)
        spl.addWidget(self.chart)
        spl.setSizes([320, 960])
        spl.setStretchFactor(0, 0)
        spl.setStretchFactor(1, 1)
        root.addWidget(spl, 1)

        # ── §6 Status bar ─────────────────────────────────────────────────────
        sb = QStatusBar()
        sb.setFixedHeight(48)
        self._msg = QLabel(
            "  Ready — configure inputs and click  ▶ Run Analysis.")
        self._msg.setStyleSheet(f"color: {C_SLATE}; font-size: 11px;")
        sb.addWidget(self._msg, 1)
        self.setStatusBar(sb)

        # ── Connect sensitivity section ────────────────────────────────────────
        self.sidebar.sens.do_run.connect(self._run_sens)
        self.sidebar.sens.do_clear.connect(self.chart.clear_sens)

    # ── Status helper ─────────────────────────────────────────────────────────

    def _status(self, text: str, color: str = C_SLATE) -> None:
        self._msg.setStyleSheet(
            f"color: {color}; font-size: 11px; padding: 0 12px;")
        self._msg.setText(text)

    # ── Run Analysis ──────────────────────────────────────────────────────────

    def run_analysis(self) -> None:
        params = self.sidebar.all_values()
        self.tbar.run_btn.setEnabled(False)
        self._status("⏳  Computing…", C_BLUE)

        worker = AnalysisWorker(params)
        worker.signals.status.connect(lambda t: self._status(t, C_BLUE))
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)
        self._pool.start(worker)

    @pyqtSlot(object)
    def _on_result(self, data: dict) -> None:
        self.tbar.run_btn.setEnabled(True)
        self.chart.plot(data)
        sol = data["sol"]

        if sol["success"]:
            q, p = sol["operating_rate"], sol["operating_pwf"]
            pts_count = len(sol.get("all_points", []))
            pts_str = f" ({pts_count} intersections found)" if pts_count > 1 else ""
            self._status(
                f"✅  Operating Point:  q* = {q:.1f} STB/day"
                f"  |  Pwf* = {p:.1f} psia"
                f"  |  Method: Nodal Analysis{pts_str}",
                C_INK,
            )
        else:
            self._status(
                f"⚠️  No Operating Point — {sol['message']}", "#B71C1C")

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self.tbar.run_btn.setEnabled(True)
        last_line = msg.strip().splitlines()[-1]
        self._status(
            f"❌  Error: {last_line}  — check inputs and retry.", "#B71C1C")
        print(msg, file=sys.stderr)  # full traceback to console

    # ── Sensitivity ───────────────────────────────────────────────────────────

    @pyqtSlot(str, float, float, int)
    def _run_sens(self, param: str, lo: float, hi: float, steps: int) -> None:
        params = self.sidebar.all_values()
        values = np.linspace(lo, hi, steps).tolist()
        self._status(
            f"⏳  Sensitivity sweep: {param}  ({steps} curves)…", C_BLUE)
        self.sidebar.sens.run_btn.setEnabled(False)

        worker = SensWorker(params, values, param)
        worker.signals.status.connect(lambda t: self._status(t, C_BLUE))
        worker.signals.result.connect(self._on_sens)
        worker.signals.error.connect(self._on_error)
        self._pool.start(worker)

    @pyqtSlot(object)
    def _on_sens(self, results: list) -> None:
        self.sidebar.sens.run_btn.setEnabled(True)
        self.chart.add_sens(results)
        self._status(
            f"✅  Sensitivity complete — {len(results)} curves overlaid.", C_INK)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Nodal Analysis — IPM")
    app.setApplicationVersion("1.0")
    app.setStyle("Fusion")     # crisp cross-platform look; QSS overrides colours
    app.setStyleSheet(QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
