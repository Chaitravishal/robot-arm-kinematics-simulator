"""
app.py
SCARA Robot Arm Assembly-Line Simulator.
Subject: Robot Kinematics and Control (Mini Project 2).
Tech Stack: Python, Streamlit, Plotly, NumPy, pandas.

Neat, calm, light "morning in the factory" theme.
Planar 2R SCARA-style arm with parallel gripper operating on an industrial conveyor line.
"""

import time
from typing import Dict, List, Optional, Tuple
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & LIGHT FACTORY THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SCARA Robot Arm Assembly-Line Simulator",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom light morning factory CSS
st.markdown("""
<style>
    /* Light morning factory styling */
    .stApp {
        background-color: #fff8ea;
        color: #1e293b;
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #d97706;
        margin-bottom: 0.1rem;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 0.8rem;
    }
    .status-card {
        background-color: #ffffff;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-step {
        font-size: 1.1rem;
        font-weight: 700;
        color: #b45309;
        margin-bottom: 4px;
    }
    .status-telemetry {
        font-size: 0.92rem;
        color: #334155;
    }
    .caption-note {
        font-size: 0.88rem;
        color: #64748b;
        margin-top: 6px;
        font-style: italic;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #fef3c7;
        border-radius: 6px 6px 0 0;
        color: #92400e;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        border-top: 3px solid #d97706 !important;
        color: #b45309 !important;
    }
    .verdict-box {
        background-color: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 10px 14px;
        border-radius: 4px;
        font-weight: 600;
        color: #92400e;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MATHEMATICAL FOUNDATIONS (KINEMATICS & TRAJECTORY GENERATION)
# -----------------------------------------------------------------------------

def wrap_angle(angle: float) -> float:
    """Wraps an angle in radians strictly to the interval (-pi, pi]."""
    w = (angle + np.pi) % (2.0 * np.pi) - np.pi
    if w == -np.pi:
        w = np.pi
    return float(w)


def forward_kinematics(theta1: float, theta2: float, l1: float, l2: float) -> Dict:
    """
    Computes 2D planar forward kinematics and 3x3 homogeneous transforms.
    x = L1*cos(t1) + L2*cos(t1+t2)
    y = L1*sin(t1) + L2*sin(t1+t2)
    """
    x0, y0 = 0.0, 0.0
    x1 = float(l1 * np.cos(theta1))
    y1 = float(l1 * np.sin(theta1))

    phi = theta1 + theta2
    x2 = float(x1 + l2 * np.cos(phi))
    y2 = float(y1 + l2 * np.sin(phi))

    reach = float(np.sqrt(x2**2 + y2**2))

    # Planar homogeneous matrices: T = Rz(theta) * Tx(L)
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)

    T01 = np.array([
        [c1, -s1, l1 * c1],
        [s1,  c1, l1 * s1],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    T12 = np.array([
        [c2, -s2, l2 * c2],
        [s2,  c2, l2 * s2],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    T02 = T01 @ T12

    def to_latex(mat: np.ndarray) -> str:
        rows = [" & ".join([f"{v:.3f}" for v in row]) for row in mat]
        return r"\begin{bmatrix} " + r" \\ ".join(rows) + r" \end{bmatrix}"

    return {
        "base": (x0, y0),
        "elbow": (x1, y1),
        "ee": (x2, y2),
        "phi": float(phi),
        "phi_deg": float(np.degrees(phi)),
        "reach": reach,
        "T01": T01,
        "T12": T12,
        "T02": T02,
        "T01_latex": to_latex(T01),
        "T12_latex": to_latex(T12),
        "T02_latex": to_latex(T02),
    }


def solve_inverse_kinematics(
    x: float,
    y: float,
    l1: float,
    l2: float,
    t1_limits: Tuple[float, float],
    t2_limits: Tuple[float, float],
    preferred_elbow: str = "right"
) -> Dict:
    """
    Analytic 2-link Inverse Kinematics with elbow choice and reachability check.
    c2 = (x^2 + y^2 - L1^2 - L2^2) / (2*L1*L2)
    t2 = +- acos(c2)
    t1 = atan2(y, x) - atan2(L2*sin(t2), L1 + L2*cos(t2))
    """
    r_sq = x**2 + y**2
    r = float(np.sqrt(r_sq))
    r_max = l1 + l2
    r_min = abs(l1 - l2)

    # 1. Annulus reachability check
    if r > r_max + 1e-6 or r < r_min - 1e-6:
        err = f"Target distance ({r:.3f}m) outside reachable range [{r_min:.2f}m, {r_max:.2f}m]."
        return {
            "reachable": False,
            "annulus_reachable": False,
            "distance": r,
            "r_min": r_min,
            "r_max": r_max,
            "c2": None,
            "elbow_right": None,
            "elbow_left": None,
            "selected": None,
            "selected_deg": None,
            "error_message": err,
        }

    c2 = (r_sq - l1**2 - l2**2) / (2.0 * l1 * l2)
    c2 = float(np.clip(c2, -1.0, 1.0))

    # Two elbow solutions
    t2_right = float(np.arccos(c2))  # theta2 >= 0
    t2_left = -t2_right              # theta2 <= 0

    beta_r = np.arctan2(l2 * np.sin(t2_right), l1 + l2 * np.cos(t2_right))
    t1_right = wrap_angle(np.arctan2(y, x) - beta_r)

    beta_l = np.arctan2(l2 * np.sin(t2_left), l1 + l2 * np.cos(t2_left))
    t1_left = wrap_angle(np.arctan2(y, x) - beta_l)

    t1_min, t1_max = t1_limits
    t2_min, t2_max = t2_limits

    def check_lim(t1, t2):
        if not (t1_min - 1e-4 <= t1 <= t1_max + 1e-4):
            return False, f"Joint 1 ({np.degrees(t1):.1f}°) exceeds limits [{np.degrees(t1_min):.0f}°, {np.degrees(t1_max):.0f}°]"
        if not (t2_min - 1e-4 <= t2 <= t2_max + 1e-4):
            return False, f"Joint 2 ({np.degrees(t2):.1f}°) exceeds limits [{np.degrees(t2_min):.0f}°, {np.degrees(t2_max):.0f}°]"
        return True, "Valid"

    r_ok, r_msg = check_lim(t1_right, t2_right)
    l_ok, l_msg = check_lim(t1_left, t2_left)

    sol_right = {
        "theta1": t1_right, "theta2": t2_right,
        "theta1_deg": float(np.degrees(t1_right)), "theta2_deg": float(np.degrees(t2_right)),
        "valid": r_ok, "message": r_msg
    }
    sol_left = {
        "theta1": t1_left, "theta2": t2_left,
        "theta1_deg": float(np.degrees(t1_left)), "theta2_deg": float(np.degrees(t2_left)),
        "valid": l_ok, "message": l_msg
    }

    chosen = sol_right if preferred_elbow == "right" else sol_left
    other = sol_left if preferred_elbow == "right" else sol_right

    if chosen["valid"]:
        selected = (chosen["theta1"], chosen["theta2"])
        selected_deg = (chosen["theta1_deg"], chosen["theta2_deg"])
        is_reachable = True
        err = ""
    elif other["valid"]:
        selected = (other["theta1"], other["theta2"])
        selected_deg = (other["theta1_deg"], other["theta2_deg"])
        is_reachable = True
        err = f"Preferred elbow violated limits ({chosen['message']}). Used alternate configuration."
    else:
        selected = None
        selected_deg = None
        is_reachable = False
        err = f"Both configurations violate joint limits: Right ({sol_right['message']}), Left ({sol_left['message']})."

    return {
        "reachable": is_reachable,
        "annulus_reachable": True,
        "distance": r,
        "r_min": r_min,
        "r_max": r_max,
        "c2": c2,
        "elbow_right": sol_right,
        "elbow_left": sol_left,
        "selected": selected,
        "selected_deg": selected_deg,
        "error_message": err,
    }


def quintic_scaling(u: float) -> Tuple[float, float, float]:
    """
    Quintic polynomial time scaling: s(u) = 10*u^3 - 15*u^4 + 6*u^5.
    Zero velocity and zero acceleration at both u=0 and u=1.
    """
    u = float(np.clip(u, 0.0, 1.0))
    s = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
    ds = 30.0 * (u**2) - 60.0 * (u**3) + 30.0 * (u**4)
    d2s = 60.0 * u - 180.0 * (u**2) + 120.0 * (u**3)
    return s, ds, d2s


def plan_ptp_trajectory(
    q_start: Tuple[float, float],
    q_end: Tuple[float, float],
    l1: float,
    l2: float,
    num_steps: int = 30,
    transfer_time: float = 2.0
) -> Dict:
    """Plans Point-to-Point (Joint-Space) trajectory."""
    t1_s, t2_s = q_start
    t1_e, t2_e = q_end

    us = np.linspace(0.0, 1.0, num_steps)
    times = us * transfer_time
    dt = transfer_time / max(1, num_steps - 1)

    theta1_list, theta2_list = [], []
    dtheta1_list, dtheta2_list = [], []
    ee_x_list, ee_y_list = [], []
    arm_states = []

    for u in us:
        s, ds, _ = quintic_scaling(u)
        t1 = t1_s + s * (t1_e - t1_s)
        t2 = t2_s + s * (t2_e - t2_s)
        w1 = (ds / transfer_time) * (t1_e - t1_s)
        w2 = (ds / transfer_time) * (t2_e - t2_s)

        theta1_list.append(t1)
        theta2_list.append(t2)
        dtheta1_list.append(w1)
        dtheta2_list.append(w2)

        fk = forward_kinematics(t1, t2, l1, l2)
        ee_x_list.append(fk["ee"][0])
        ee_y_list.append(fk["ee"][1])
        arm_states.append(fk)

    vx = np.gradient(ee_x_list, dt)
    vy = np.gradient(ee_y_list, dt)
    tool_speeds = np.sqrt(vx**2 + vy**2).tolist()

    dx = np.diff(ee_x_list)
    dy = np.diff(ee_y_list)
    path_len = float(np.sum(np.sqrt(dx**2 + dy**2)))

    j1_travel = float(np.sum(np.abs(np.diff(theta1_list))))
    j2_travel = float(np.sum(np.abs(np.diff(theta2_list))))

    return {
        "name": "Point-to-Point (Joint-Space)",
        "feasible": True,
        "reason": "",
        "times": times,
        "theta1": theta1_list,
        "theta2": theta2_list,
        "dtheta1": dtheta1_list,
        "dtheta2": dtheta2_list,
        "ee_x": ee_x_list,
        "ee_y": ee_y_list,
        "tool_speed": tool_speeds,
        "path_length": path_len,
        "j1_travel": j1_travel,
        "j2_travel": j2_travel,
        "total_joint_travel": j1_travel + j2_travel,
        "peak_j1_speed_deg": float(np.max(np.abs(np.degrees(dtheta1_list)))),
        "peak_j2_speed_deg": float(np.max(np.abs(np.degrees(dtheta2_list)))),
        "peak_tool_speed": float(np.max(tool_speeds)),
        "arm_states": arm_states,
    }


def plan_continuous_trajectory(
    p_start: Tuple[float, float],
    p_end: Tuple[float, float],
    l1: float,
    l2: float,
    t1_limits: Tuple[float, float],
    t2_limits: Tuple[float, float],
    preferred_elbow: str = "right",
    num_steps: int = 30,
    transfer_time: float = 2.0
) -> Dict:
    """Plans Continuous Cartesian straight-line path with waypoint IK verification."""
    xs, ys = p_start
    xe, ye = p_end

    us = np.linspace(0.0, 1.0, num_steps)
    times = us * transfer_time
    dt = transfer_time / max(1, num_steps - 1)
    dist_straight = float(np.sqrt((xe - xs)**2 + (ye - ys)**2))

    ee_x_list, ee_y_list = [], []
    theta1_list, theta2_list = [], []
    arm_states = []

    feasible = True
    infeasible_reason = ""

    for idx, u in enumerate(us):
        s, _, _ = quintic_scaling(u)
        xk = xs + s * (xe - xs)
        yk = ys + s * (ye - ys)
        ee_x_list.append(xk)
        ee_y_list.append(yk)

        ik = solve_inverse_kinematics(
            xk, yk, l1, l2, t1_limits, t2_limits, preferred_elbow=preferred_elbow
        )

        branch = ik["elbow_right"] if preferred_elbow == "right" else ik["elbow_left"]

        if not ik["annulus_reachable"]:
            feasible = False
            if not infeasible_reason:
                infeasible_reason = f"Waypoint {idx+1} at ({xk:.2f}, {yk:.2f})m leaves workspace (r={ik['distance']:.2f}m)."
            t1 = theta1_list[-1] if theta1_list else 0.0
            t2 = theta2_list[-1] if theta2_list else 0.0
        elif branch is None or not branch["valid"]:
            feasible = False
            if not infeasible_reason:
                msg = branch["message"] if branch else "No solution"
                infeasible_reason = f"Waypoint {idx+1} at ({xk:.2f}, {yk:.2f})m violates joint limits ({msg})."
            if ik["selected"]:
                t1, t2 = ik["selected"]
            else:
                t1 = theta1_list[-1] if theta1_list else 0.0
                t2 = theta2_list[-1] if theta2_list else 0.0
        else:
            t1 = branch["theta1"]
            t2 = branch["theta2"]

        if theta2_list and abs(t2 - theta2_list[-1]) > np.radians(60):
            feasible = False
            if not infeasible_reason:
                infeasible_reason = f"Waypoint {idx+1} causes an elbow branch jump."

        theta1_list.append(t1)
        theta2_list.append(t2)
        arm_states.append(forward_kinematics(t1, t2, l1, l2))

    dtheta1_list = np.gradient(theta1_list, dt).tolist()
    dtheta2_list = np.gradient(theta2_list, dt).tolist()
    tool_speeds = [(quintic_scaling(u)[1] / transfer_time) * dist_straight for u in us]

    dx = np.diff(ee_x_list)
    dy = np.diff(ee_y_list)
    path_len = float(np.sum(np.sqrt(dx**2 + dy**2)))

    j1_travel = float(np.sum(np.abs(np.diff(theta1_list))))
    j2_travel = float(np.sum(np.abs(np.diff(theta2_list))))

    return {
        "name": "Continuous-Path (Cartesian)",
        "feasible": feasible,
        "reason": infeasible_reason,
        "times": times,
        "theta1": theta1_list,
        "theta2": theta2_list,
        "dtheta1": dtheta1_list,
        "dtheta2": dtheta2_list,
        "ee_x": ee_x_list,
        "ee_y": ee_y_list,
        "tool_speed": tool_speeds,
        "path_length": path_len,
        "j1_travel": j1_travel,
        "j2_travel": j2_travel,
        "total_joint_travel": j1_travel + j2_travel,
        "peak_j1_speed_deg": float(np.max(np.abs(np.degrees(dtheta1_list)))),
        "peak_j2_speed_deg": float(np.max(np.abs(np.degrees(dtheta2_list)))),
        "peak_tool_speed": float(np.max(tool_speeds)),
        "arm_states": arm_states,
    }


def compute_comparison_metrics(
    ptp: Dict,
    cart: Dict,
    p_start: Tuple[float, float],
    p_end: Tuple[float, float]
) -> pd.DataFrame:
    """Computes comparison metrics between PTP and Continuous Path."""
    xs, ys = p_start
    xe, ye = p_end
    dist_straight = float(np.sqrt((xe - xs)**2 + (ye - ys)**2))

    eff_ptp = (dist_straight / max(1e-5, ptp["path_length"])) * 100.0
    eff_cart = (dist_straight / max(1e-5, cart["path_length"])) * 100.0

    def get_max_dev(x_pts, y_pts):
        line_len = max(1e-5, dist_straight)
        devs = [
            abs((ye - ys) * x - (xe - xs) * y + xe * ys - ye * xs) / line_len
            for x, y in zip(x_pts, y_pts)
        ]
        return float(np.max(devs))

    max_dev_ptp = get_max_dev(ptp["ee_x"], ptp["ee_y"])
    max_dev_cart = get_max_dev(cart["ee_x"], cart["ee_y"])

    data = [
        {"Metric": "Cartesian Path Length", "Point-to-Point (Joint-Space)": f"{ptp['path_length']:.3f} m", "Continuous-Path (Cartesian)": f"{cart['path_length']:.3f} m"},
        {"Metric": "Straight-Line Distance", "Point-to-Point (Joint-Space)": f"{dist_straight:.3f} m", "Continuous-Path (Cartesian)": f"{dist_straight:.3f} m"},
        {"Metric": "Path Efficiency (%)", "Point-to-Point (Joint-Space)": f"{eff_ptp:.1f}%", "Continuous-Path (Cartesian)": f"{eff_cart:.1f}%"},
        {"Metric": "Max Deviation from Straight Line", "Point-to-Point (Joint-Space)": f"{max_dev_ptp:.3f} m", "Continuous-Path (Cartesian)": f"{max_dev_cart:.3f} m"},
        {"Metric": "Joint 1 Travel (θ1)", "Point-to-Point (Joint-Space)": f"{np.degrees(ptp['j1_travel']):.1f}°", "Continuous-Path (Cartesian)": f"{np.degrees(cart['j1_travel']):.1f}°"},
        {"Metric": "Joint 2 Travel (θ2)", "Point-to-Point (Joint-Space)": f"{np.degrees(ptp['j2_travel']):.1f}°", "Continuous-Path (Cartesian)": f"{np.degrees(cart['j2_travel']):.1f}°"},
        {"Metric": "Total Joint Motion", "Point-to-Point (Joint-Space)": f"{np.degrees(ptp['total_joint_travel']):.1f}°", "Continuous-Path (Cartesian)": f"{np.degrees(cart['total_joint_travel']):.1f}°"},
        {"Metric": "Peak Joint Speed (θ1)", "Point-to-Point (Joint-Space)": f"{ptp['peak_j1_speed_deg']:.1f} °/s", "Continuous-Path (Cartesian)": f"{cart['peak_j1_speed_deg']:.1f} °/s"},
        {"Metric": "Peak Joint Speed (θ2)", "Point-to-Point (Joint-Space)": f"{ptp['peak_j2_speed_deg']:.1f} °/s", "Continuous-Path (Cartesian)": f"{cart['peak_j2_speed_deg']:.1f} °/s"},
        {"Metric": "Peak Tool Speed", "Point-to-Point (Joint-Space)": f"{ptp['peak_tool_speed']:.2f} m/s", "Continuous-Path (Cartesian)": f"{cart['peak_tool_speed']:.2f} m/s"},
        {"Metric": "Path Feasibility", "Point-to-Point (Joint-Space)": "FEASIBLE" if ptp["feasible"] else f"INFEASIBLE: {ptp['reason']}", "Continuous-Path (Cartesian)": "FEASIBLE" if cart["feasible"] else f"INFEASIBLE: {cart['reason']}"},
    ]
    return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# 3. 2D TOP-VIEW WORKCELL RENDERER (PLOTLY)
# -----------------------------------------------------------------------------

def draw_2d_top_view(
    arm_state: Dict,
    l1: float,
    l2: float,
    pick_pos: Tuple[float, float],
    place_pos: Tuple[float, float],
    gripper_closed: bool,
    box_held: bool,
    box_at_pick: bool,
    box_delivered_offset: float = 0.0,
    traj_ptp: Optional[Dict] = None,
    traj_cart: Optional[Dict] = None,
    show_reach_circles: bool = True,
    chosen_strategy: str = "Both"
) -> go.Figure:
    """
    Renders a clean, calm 2D TOP-VIEW factory floor plan of the industrial workcell.
    Cream factory floor background, realistic dark belts with side rails, rollers, and legs.
    Labels positioned above belts to eliminate box overlap.
    Axes, gridlines, and ticks hidden for an authentic floor plan view.
    """
    fig = go.Figure()

    # Conveyor dimensions
    belt_w = 0.60
    belt_half = belt_w / 2.0
    y_conv = pick_pos[1]  # Conveyors share the same Y line

    # 1. INPUT CONVEYOR (LEFT)
    in_x0, in_x1 = -3.8, -1.2

    # 1a. Legs for input conveyor (small support brackets at ends)
    for lx in [in_x0 + 0.12, in_x1 - 0.12]:
        # Top leg
        fig.add_shape(
            type="rect",
            x0=lx - 0.06, x1=lx + 0.06,
            y0=y_conv + belt_half, y1=y_conv + belt_half + 0.08,
            fillcolor="#1e293b", line=dict(color="#0f172a", width=1),
            layer="below"
        )
        # Bottom leg
        fig.add_shape(
            type="rect",
            x0=lx - 0.06, x1=lx + 0.06,
            y0=y_conv - belt_half - 0.08, y1=y_conv - belt_half,
            fillcolor="#1e293b", line=dict(color="#0f172a", width=1),
            layer="below"
        )

    # 1b. Dark belt surface
    fig.add_shape(
        type="rect",
        x0=in_x0, x1=in_x1,
        y0=y_conv - belt_half, y1=y_conv + belt_half,
        fillcolor="#334155", line=dict(color="#1e293b", width=1.5),
        layer="below"
    )

    # 1c. Rollers across input belt (thin lines)
    for rx in np.arange(in_x0 + 0.15, in_x1, 0.20):
        fig.add_shape(
            type="line",
            x0=rx, x1=rx,
            y0=y_conv - belt_half, y1=y_conv + belt_half,
            line=dict(color="#475569", width=1.5),
            layer="below"
        )

    # 1d. Side rails for input conveyor (top and bottom)
    fig.add_shape(
        type="line",
        x0=in_x0 - 0.04, x1=in_x1 + 0.04,
        y0=y_conv + belt_half, y1=y_conv + belt_half,
        line=dict(color="#94a3b8", width=4),
        layer="below"
    )
    fig.add_shape(
        type="line",
        x0=in_x0 - 0.04, x1=in_x1 + 0.04,
        y0=y_conv - belt_half, y1=y_conv - belt_half,
        line=dict(color="#94a3b8", width=4),
        layer="below"
    )

    # 1e. Label moved cleanly ABOVE the belt (never overlaps boxes)
    fig.add_annotation(
        x=(in_x0 + in_x1) / 2.0, y=y_conv + belt_half + 0.16,
        text="INPUT CONVEYOR (INFLOW ➔)", showarrow=False,
        font=dict(size=11, color="#334155", family="sans-serif")
    )

    # 2. OUTPUT CONVEYOR (RIGHT)
    out_x0, out_x1 = 1.2, 3.8

    # 2a. Legs for output conveyor (small support brackets at ends)
    for lx in [out_x0 + 0.12, out_x1 - 0.12]:
        # Top leg
        fig.add_shape(
            type="rect",
            x0=lx - 0.06, x1=lx + 0.06,
            y0=y_conv + belt_half, y1=y_conv + belt_half + 0.08,
            fillcolor="#1e293b", line=dict(color="#0f172a", width=1),
            layer="below"
        )
        # Bottom leg
        fig.add_shape(
            type="rect",
            x0=lx - 0.06, x1=lx + 0.06,
            y0=y_conv - belt_half - 0.08, y1=y_conv - belt_half,
            fillcolor="#1e293b", line=dict(color="#0f172a", width=1),
            layer="below"
        )

    # 2b. Dark belt surface
    fig.add_shape(
        type="rect",
        x0=out_x0, x1=out_x1,
        y0=y_conv - belt_half, y1=y_conv + belt_half,
        fillcolor="#334155", line=dict(color="#1e293b", width=1.5),
        layer="below"
    )

    # 2c. Rollers across output belt (thin lines)
    for rx in np.arange(out_x0 + 0.15, out_x1, 0.20):
        fig.add_shape(
            type="line",
            x0=rx, x1=rx,
            y0=y_conv - belt_half, y1=y_conv + belt_half,
            line=dict(color="#475569", width=1.5),
            layer="below"
        )

    # 2d. Side rails for output conveyor (top and bottom)
    fig.add_shape(
        type="line",
        x0=out_x0 - 0.04, x1=out_x1 + 0.04,
        y0=y_conv + belt_half, y1=y_conv + belt_half,
        line=dict(color="#94a3b8", width=4),
        layer="below"
    )
    fig.add_shape(
        type="line",
        x0=out_x0 - 0.04, x1=out_x1 + 0.04,
        y0=y_conv - belt_half, y1=y_conv - belt_half,
        line=dict(color="#94a3b8", width=4),
        layer="below"
    )

    # 2e. Label moved cleanly ABOVE the belt (never overlaps boxes)
    fig.add_annotation(
        x=(out_x0 + out_x1) / 2.0, y=y_conv + belt_half + 0.16,
        text="OUTPUT CONVEYOR (OUTFLOW ➔)", showarrow=False,
        font=dict(size=11, color="#334155", family="sans-serif")
    )

    # 3. MARKED PICK & PLACE STATIONS (DASHED OUTLINES)
    sw = 0.32
    fig.add_shape(
        type="rect",
        x0=pick_pos[0] - sw/2, x1=pick_pos[0] + sw/2,
        y0=pick_pos[1] - sw/2, y1=pick_pos[1] + sw/2,
        fillcolor="rgba(56, 189, 248, 0.15)",
        line=dict(color="#38bdf8", width=2.5, dash="dash")
    )
    fig.add_annotation(
        x=pick_pos[0], y=y_conv - belt_half - 0.16,
        text="PICK STATION", showarrow=False,
        font=dict(size=11, color="#0284c7", family="sans-serif")
    )

    fig.add_shape(
        type="rect",
        x0=place_pos[0] - sw/2, x1=place_pos[0] + sw/2,
        y0=place_pos[1] - sw/2, y1=place_pos[1] + sw/2,
        fillcolor="rgba(251, 191, 36, 0.15)",
        line=dict(color="#fbbf24", width=2.5, dash="dash")
    )
    fig.add_annotation(
        x=place_pos[0], y=y_conv - belt_half - 0.16,
        text="PLACE STATION", showarrow=False,
        font=dict(size=11, color="#b45309", family="sans-serif")
    )

    # 4. LIGHT REACH CIRCLES
    if show_reach_circles:
        r_max = l1 + l2
        r_min = abs(l1 - l2)
        th = np.linspace(0, 2*np.pi, 120)
        fig.add_trace(go.Scatter(
            x=r_max * np.cos(th), y=r_max * np.sin(th),
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.45)", width=1.5, dash="dash"),
            name="Max Reach (L1+L2)", hoverinfo="name", showlegend=False
        ))
        if r_min > 0.05:
            fig.add_trace(go.Scatter(
                x=r_min * np.cos(th), y=r_min * np.sin(th),
                mode="lines",
                line=dict(color="rgba(148, 163, 184, 0.30)", width=1.5, dash="dot"),
                name="Inner Limit (|L1-L2|)", hoverinfo="name", showlegend=False
            ))

    # 5. PLANNED TRANSFER PATHS (THICKNESS DETERMINED BY CHOSEN STRATEGY)
    if chosen_strategy == "Point-to-point":
        ptp_width = 5.5
        ptp_color = "rgba(239, 68, 68, 0.95)"
        cart_width = 1.8
        cart_color = "rgba(217, 119, 6, 0.35)"
    elif chosen_strategy == "Continuous path":
        ptp_width = 1.8
        ptp_color = "rgba(239, 68, 68, 0.35)"
        cart_width = 5.5
        cart_color = "rgba(217, 119, 6, 0.95)"
    else:  # "Both"
        ptp_width = 4.0
        ptp_color = "rgba(239, 68, 68, 0.85)"
        cart_width = 4.0
        cart_color = "rgba(217, 119, 6, 0.85)"

    if traj_ptp is not None and "ee_x" in traj_ptp:
        fig.add_trace(go.Scatter(
            x=traj_ptp["ee_x"], y=traj_ptp["ee_y"],
            mode="lines",
            line=dict(color=ptp_color, width=ptp_width, dash="dot"),
            name="Point-to-point path (joint arc)",
            hoverinfo="name",
            showlegend=True
        ))
    if traj_cart is not None and "ee_x" in traj_cart:
        fig.add_trace(go.Scatter(
            x=traj_cart["ee_x"], y=traj_cart["ee_y"],
            mode="lines",
            line=dict(color=cart_color, width=cart_width),
            name="Continuous path (straight line)",
            hoverinfo="name",
            showlegend=True
        ))

    # 6. QUEUED AND DELIVERED BOXES (CARDBOARD BROWN)
    box_size = 0.22
    # Input belt queue
    for qx in [-3.4, -2.8]:
        fig.add_shape(
            type="rect",
            x0=qx - box_size/2, x1=qx + box_size/2,
            y0=y_conv - box_size/2, y1=y_conv + box_size/2,
            fillcolor="#b45309", line=dict(color="#78350f", width=1.5)
        )

    # Box at PICK station (if waiting)
    if box_at_pick and not box_held:
        fig.add_shape(
            type="rect",
            x0=pick_pos[0] - box_size/2, x1=pick_pos[0] + box_size/2,
            y0=pick_pos[1] - box_size/2, y1=pick_pos[1] + box_size/2,
            fillcolor="#d97706", line=dict(color="#92400e", width=2)
        )

    # Delivered box sliding away on output belt
    if box_delivered_offset > 0.05:
        dx = place_pos[0] + box_delivered_offset
        if dx < 3.7:
            fig.add_shape(
                type="rect",
                x0=dx - box_size/2, x1=dx + box_size/2,
                y0=y_conv - box_size/2, y1=y_conv + box_size/2,
                fillcolor="#b45309", line=dict(color="#78350f", width=1.5)
            )

    # Output belt finished queue
    for fx in [3.1, 3.5]:
        fig.add_shape(
            type="rect",
            x0=fx - box_size/2, x1=fx + box_size/2,
            y0=y_conv - box_size/2, y1=y_conv + box_size/2,
            fillcolor="#b45309", line=dict(color="#78350f", width=1.5)
        )

    # 7. ROBOT ARM: BASE, LINK 1, ELBOW, LINK 2
    bx, by = arm_state["base"]
    ex, ey = arm_state["elbow"]
    eex, eey = arm_state["ee"]
    phi = arm_state["phi"]

    # Base Pedestal (Center)
    fig.add_shape(
        type="circle",
        x0=bx - 0.22, x1=bx + 0.22,
        y0=by - 0.22, y1=by + 0.22,
        fillcolor="#334155", line=dict(color="#1e293b", width=2)
    )
    fig.add_annotation(x=bx, y=by - 0.32, text="ROBOT BASE (0,0)", showarrow=False, font=dict(size=10, color="#475569"))

    # Link 1 (Blue)
    fig.add_trace(go.Scatter(
        x=[bx, ex], y=[by, ey],
        mode="lines",
        line=dict(color="#2563eb", width=8),
        name=f"Link 1 ({l1:.1f}m)", hoverinfo="name"
    ))

    # Joint 2 Hub (Elbow)
    fig.add_shape(
        type="circle",
        x0=ex - 0.10, x1=ex + 0.10,
        y0=ey - 0.10, y1=ey + 0.10,
        fillcolor="#f59e0b", line=dict(color="#d97706", width=2)
    )

    # Link 2 (Orange)
    fig.add_trace(go.Scatter(
        x=[ex, eex], y=[ey, eey],
        mode="lines",
        line=dict(color="#ea580c", width=7),
        name=f"Link 2 ({l2:.1f}m)", hoverinfo="name"
    ))

    # 8. PARALLEL GRIPPER (TWO JAWS THAT OPEN AND CLOSE)
    # Unit vector along tool orientation: u = (cos phi, sin phi)
    # Unit vector perpendicular to tool: v = (-sin phi, cos phi)
    ux, uy = np.cos(phi), np.sin(phi)
    vx, vy = -np.sin(phi), np.cos(phi)

    jaw_half = 0.08 if gripper_closed else 0.12  # Clamped vs Open spacing
    finger_len = 0.14
    palm_len = 0.06

    # Palm crossbar
    p1x, p1y = eex + jaw_half * vx, eey + jaw_half * vy
    p2x, p2y = eex - jaw_half * vx, eey - jaw_half * vy
    fig.add_trace(go.Scatter(
        x=[p1x, p2x], y=[p1y, p2y],
        mode="lines", line=dict(color="#1e293b", width=5),
        showlegend=False, hoverinfo="skip"
    ))

    # Jaw 1 (Finger 1)
    j1_end_x = p1x + finger_len * ux
    j1_end_y = p1y + finger_len * uy
    fig.add_trace(go.Scatter(
        x=[p1x, j1_end_x], y=[p1y, j1_end_y],
        mode="lines", line=dict(color="#0f172a", width=4),
        name="Gripper Jaws", showlegend=False, hoverinfo="skip"
    ))

    # Jaw 2 (Finger 2)
    j2_end_x = p2x + finger_len * ux
    j2_end_y = p2y + finger_len * uy
    fig.add_trace(go.Scatter(
        x=[p2x, j2_end_x], y=[p2y, j2_end_y],
        mode="lines", line=dict(color="#0f172a", width=4),
        showlegend=False, hoverinfo="skip"
    ))

    # 9. BOX CARRIED BY GRIPPER
    if box_held:
        box_cx = eex + (finger_len / 2.0) * ux
        box_cy = eey + (finger_len / 2.0) * uy
        # Draw rotated box inside gripper
        b_corners = [
            (-box_size/2, -box_size/2),
            (box_size/2, -box_size/2),
            (box_size/2, box_size/2),
            (-box_size/2, box_size/2),
            (-box_size/2, -box_size/2),
        ]
        b_x = [box_cx + c[0]*ux - c[1]*vx for c in b_corners]
        b_y = [box_cy + c[0]*uy + c[1]*ux for c in b_corners]
        fig.add_trace(go.Scatter(
            x=b_x, y=b_y, fill="toself",
            fillcolor="#d97706", line=dict(color="#92400e", width=2),
            name="Payload Box (Gripped)", hoverinfo="name",
            showlegend=False
        ))

    # Plot Layout (Clean 2D Plan View with cream factory floor background, no axes/gridlines)
    fig.update_layout(
        plot_bgcolor="#fff8ea",
        paper_bgcolor="#fff8ea",
        xaxis=dict(
            range=[-4.2, 4.2],
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            ticks="",
            title=""
        ),
        yaxis=dict(
            range=[-0.8, 3.4],
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            ticks="",
            title="",
            scaleanchor="x", scaleratio=1
        ),
        margin=dict(l=20, r=20, t=10, b=65),
        height=490,
        legend=dict(
            orientation="h",
            y=-0.08,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=11, color="#1e293b"),
            bgcolor="rgba(255, 248, 234, 0.85)",
            bordercolor="#cbd5e1",
            borderwidth=1
        )
    )

    return fig


# -----------------------------------------------------------------------------
# 4. SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ SCARA Arm Parameters")
l1 = st.sidebar.slider("Link 1 Length $L_1$ (m)", 1.0, 3.0, 2.0, 0.1)
l2 = st.sidebar.slider("Link 2 Length $L_2$ (m)", 0.5, 2.5, 1.5, 0.1)

st.sidebar.markdown("### 🔒 Joint Limits")
t1_lim_deg = st.sidebar.slider("Joint 1 Limit $\\pm\\theta_1$ (deg)", 90.0, 180.0, 170.0, 5.0)
t2_lim_deg = st.sidebar.slider("Joint 2 Limit $\\pm\\theta_2$ (deg)", 90.0, 180.0, 150.0, 5.0)

elbow_config = st.sidebar.selectbox(
    "Elbow Branch Configuration",
    options=["right", "left"],
    format_func=lambda x: "Elbow-Right (θ2 ≥ 0)" if x == "right" else "Elbow-Left (θ2 ≤ 0)"
)

st.sidebar.markdown("### ⏱️ Trajectory Settings")
num_waypoints = st.sidebar.slider("Waypoints", 15, 60, 30, 5)
transfer_time = st.sidebar.slider("Transfer Time (seconds)", 1.0, 5.0, 2.0, 0.2)

st.sidebar.markdown("### 📍 Conveyor Stations")
pick_x = st.sidebar.slider("Pick Target $X$ (m)", -3.5, -0.5, -2.2, 0.05)
pick_y = st.sidebar.slider("Pick Target $Y$ (m)", 0.5, 3.0, 2.0, 0.05)
place_x = st.sidebar.slider("Place Target $X$ (m)", 0.5, 3.5, 2.2, 0.05)
place_y = st.sidebar.slider("Place Target $Y$ (m)", 0.5, 3.0, 2.0, 0.05)

t1_limits = (-np.radians(t1_lim_deg), np.radians(t1_lim_deg))
t2_limits = (-np.radians(t2_lim_deg), np.radians(t2_lim_deg))
pick_pos = (pick_x, pick_y)
place_pos = (place_x, place_y)


# -----------------------------------------------------------------------------
# 5. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "manual_t1" not in st.session_state:
    st.session_state.manual_t1 = 0.0
if "manual_t2" not in st.session_state:
    st.session_state.manual_t2 = 0.0
if "manual_holding" not in st.session_state:
    st.session_state.manual_holding = False
if "manual_delivered_count" not in st.session_state:
    st.session_state.manual_delivered_count = 0
if "manual_message" not in st.session_state:
    st.session_state.manual_message = "Enter joint angles and click 'Move arm' to begin."

if "auto_step" not in st.session_state:
    st.session_state.auto_step = 0
if "auto_playing" not in st.session_state:
    st.session_state.auto_playing = False


# -----------------------------------------------------------------------------
# 6. HEADER & TAB STRUCTURE
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🏭 SCARA Robot Arm Assembly-Line Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Robot Kinematics and Control — Mini Project 2 | Planar 2R SCARA-Style Manipulator with Parallel Gripper</div>', unsafe_allow_html=True)

tab_assembly, tab_kinematics, tab_documentation = st.tabs([
    "1. Assembly line",
    "2. Kinematics and trajectory analysis",
    "3. Documentation"
])


# =============================================================================
# TAB 1: ASSEMBLY LINE (THE SIMULATION)
# =============================================================================
with tab_assembly:
    st.markdown("#### Mode Selector")
    sim_mode = st.radio(
        "Select Operation Mode:",
        ["A) MANUAL MODE", "B) AUTOMATIC MODE"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # Precompute static trajectory paths for visualization overlay
    ik_p = solve_inverse_kinematics(pick_x, pick_y, l1, l2, t1_limits, t2_limits, preferred_elbow=elbow_config)
    ik_pl = solve_inverse_kinematics(place_x, place_y, l1, l2, t1_limits, t2_limits, preferred_elbow=elbow_config)

    q_pick = ik_p["selected"] if ik_p["reachable"] else (0.0, 0.0)
    q_place = ik_pl["selected"] if ik_pl["reachable"] else (0.0, 0.0)

    static_ptp = plan_ptp_trajectory(q_pick, q_place, l1, l2, num_steps=num_waypoints, transfer_time=transfer_time)
    static_cart = plan_continuous_trajectory(pick_pos, place_pos, l1, l2, t1_limits, t2_limits, preferred_elbow=elbow_config, num_steps=num_waypoints, transfer_time=transfer_time)

    # -------------------------------------------------------------------------
    # MODE A: MANUAL MODE
    # -------------------------------------------------------------------------
    if sim_mode == "A) MANUAL MODE":
        if "man_joint1" not in st.session_state:
            st.session_state.man_joint1 = float(st.session_state.manual_t1)
        if "man_joint2" not in st.session_state:
            st.session_state.man_joint2 = float(st.session_state.manual_t2)

        def reset_manual_arm():
            st.session_state.man_joint1 = 0.0
            st.session_state.man_joint2 = 0.0
            st.session_state.manual_t1 = 0.0
            st.session_state.manual_t2 = 0.0
            st.session_state.manual_holding = False
            st.session_state.manual_message = "Reset completed. Arm is at 0°, 0°."

        col_inp, col_info = st.columns([1.3, 1.7])

        with col_inp:
            st.markdown("##### Manual Joint Controls")
            inp_c1, inp_c2 = st.columns(2)
            with inp_c1:
                t1_input = st.number_input("Joint 1 (θ1) deg", min_value=-t1_lim_deg, max_value=t1_lim_deg, step=5.0, key="man_joint1")
            with inp_c2:
                t2_input = st.number_input("Joint 2 (θ2) deg", min_value=-t2_lim_deg, max_value=t2_lim_deg, step=5.0, key="man_joint2")

            btn_col1, btn_col2 = st.columns(2)
            move_clicked = btn_col1.button("Move arm", width="stretch")
            btn_col2.button("Reset", width="stretch", on_click=reset_manual_arm)
            show_hint = st.checkbox("Show hint", value=False)

        if move_clicked:
            st.session_state.manual_t1 = float(t1_input)
            st.session_state.manual_t2 = float(t2_input)

            # Evaluate FK
            fk_eval = forward_kinematics(np.radians(t1_input), np.radians(t2_input), l1, l2)
            ee_cur = fk_eval["ee"]

            if not st.session_state.manual_holding:
                # Check distance to PICK
                d_pick = np.sqrt((ee_cur[0] - pick_x)**2 + (ee_cur[1] - pick_y)**2)
                if d_pick <= 0.05:
                    st.session_state.manual_holding = True
                    st.session_state.manual_message = f"🎯 Success! Gripper landed at PICK ({d_pick:.3f}m ≤ 0.05m). Jaws closed and box picked! Now move to PLACE."
                else:
                    st.session_state.manual_message = f"Gripper is {d_pick:.3f}m from PICK station (Target: ≤ 0.05m)."
            else:
                # Check distance to PLACE
                d_place = np.sqrt((ee_cur[0] - place_x)**2 + (ee_cur[1] - place_y)**2)
                if d_place <= 0.05:
                    st.session_state.manual_holding = False
                    st.session_state.manual_delivered_count += 1
                    st.session_state.manual_message = f"🎉 Great job! Gripper reached PLACE ({d_place:.3f}m ≤ 0.05m). Box released! A new box has arrived at PICK."
                else:
                    st.session_state.manual_message = f"Holding box. Gripper is {d_place:.3f}m from PLACE station (Target: ≤ 0.05m)."

        # Hint logic
        with col_info:
            st.markdown("##### Task Status & Guidance")
            st.info(st.session_state.manual_message)
            if show_hint:
                target_name = "PLACE" if st.session_state.manual_holding else "PICK"
                tgt_coord = place_pos if st.session_state.manual_holding else pick_pos
                ik_hint = solve_inverse_kinematics(tgt_coord[0], tgt_coord[1], l1, l2, t1_limits, t2_limits, preferred_elbow=elbow_config)
                if ik_hint["reachable"]:
                    h_t1, h_t2 = ik_hint["selected_deg"]
                    st.success(f"💡 **Hint to reach {target_name}:** Set Joint 1 = `{h_t1:.1f}°`, Joint 2 = `{h_t2:.1f}°`")
                else:
                    st.error(f"Target {target_name} is outside allowable range: {ik_hint['error_message']}")

        # Current arm configuration for display
        cur_arm = forward_kinematics(np.radians(st.session_state.manual_t1), np.radians(st.session_state.manual_t2), l1, l2)
        gripper_state_str = "Closed (Grasping Box)" if st.session_state.manual_holding else "Open"

        # Status title banner
        st.markdown(f"""
        <div class="status-card">
            <div class="status-step">Manual Control Step: {"Holding Box → Transfer to PLACE" if st.session_state.manual_holding else "Empty Gripper → Move to PICK"}</div>
            <div class="status-telemetry">
                <b>Joint 1 (θ1):</b> {st.session_state.manual_t1:.1f}° &nbsp;|&nbsp;
                <b>Joint 2 (θ2):</b> {st.session_state.manual_t2:.1f}° &nbsp;|&nbsp;
                <b>Gripper Position:</b> ({cur_arm['ee'][0]:.2f}, {cur_arm['ee'][1]:.2f}) m &nbsp;|&nbsp;
                <b>Gripper State:</b> {gripper_state_str} &nbsp;|&nbsp;
                <b>Delivered Count:</b> {st.session_state.manual_delivered_count}
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_man = draw_2d_top_view(
            arm_state=cur_arm,
            l1=l1, l2=l2,
            pick_pos=pick_pos, place_pos=place_pos,
            gripper_closed=st.session_state.manual_holding,
            box_held=st.session_state.manual_holding,
            box_at_pick=not st.session_state.manual_holding,
            traj_ptp=static_ptp,
            traj_cart=static_cart,
            chosen_strategy="Both"
        )
        st.plotly_chart(fig_man, width="stretch", key="manual_mode_workcell_chart")
        st.markdown('<div class="caption-note">How to read the picture: Robot base is at (0,0). Blue line is Link 1, orange line is Link 2. Two black jaws form the gripper. Boxes arrive on the left input belt, stop at the dashed PICK station, and are transferred to the dashed PLACE station on the right output belt.</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE B: AUTOMATIC MODE
    # -------------------------------------------------------------------------
    else:
        st.markdown("##### 🚀 Automatic Assembly Trajectory Planning")
        col_strat, col_act = st.columns([2.2, 1.3])
        with col_strat:
            auto_traj_type = st.radio(
                "Trajectory Planning Strategy:",
                options=["Point-to-point", "Continuous path", "Both"],
                horizontal=True,
                index=0,
                key="auto_traj_selector",
                help="Select between Point-to-point (joint-space arc), Continuous path (Cartesian straight line), or Both (consecutive cycles)."
            )

        # Reachability verification for Pick and Place
        if not ik_p["reachable"] or not ik_pl["reachable"]:
            unreach_reason = ik_p["error_message"] if not ik_p["reachable"] else ik_pl["error_message"]
            st.error(f"❌ Cannot run automatic cycle: {unreach_reason}")
        else:
            q_p = ik_p["selected"]
            q_pl = ik_pl["selected"]
            home_q = (0.0, 0.0)

            # Generate cycle animation frames
            def build_single_cycle(mode_kind: str, cycle_label: str) -> List[Dict]:
                frames = []
                # Phase 1: Box arrives at pick, arm at home (4 frames)
                for _ in range(4):
                    fk = forward_kinematics(home_q[0], home_q[1], l1, l2)
                    frames.append({
                        "step_name": f"{cycle_label}Step 1: New box arrives on input conveyor and stops at PICK",
                        "arm": fk, "gripper_closed": False, "box_held": False,
                        "box_at_pick": True, "box_slide": 0.0,
                        "active_mode": "Point-to-point" if mode_kind == "ptp" else "Continuous path"
                    })
                # Phase 2: Arm moves Home -> PICK (8 frames)
                for u in np.linspace(0, 1, 8):
                    s, _, _ = quintic_scaling(u)
                    t1 = home_q[0] + s * (q_p[0] - home_q[0])
                    t2 = home_q[1] + s * (q_p[1] - home_q[1])
                    frames.append({
                        "step_name": f"{cycle_label}Step 2: Arm moves to PICK station (open gripper)",
                        "arm": forward_kinematics(t1, t2, l1, l2), "gripper_closed": False,
                        "box_held": False, "box_at_pick": True, "box_slide": 0.0,
                        "active_mode": "Point-to-point" if mode_kind == "ptp" else "Continuous path"
                    })
                # Phase 3: Gripper closes around box (4 frames)
                fk_p = forward_kinematics(q_p[0], q_p[1], l1, l2)
                for _ in range(4):
                    frames.append({
                        "step_name": f"{cycle_label}Step 3: Gripper closes and clamps onto the box",
                        "arm": fk_p, "gripper_closed": True, "box_held": True,
                        "box_at_pick": False, "box_slide": 0.0,
                        "active_mode": "Point-to-point" if mode_kind == "ptp" else "Continuous path"
                    })
                # Phase 4: Transfer PICK -> PLACE
                if mode_kind == "continuous":
                    t_plan = plan_continuous_trajectory(pick_pos, place_pos, l1, l2, t1_limits, t2_limits, preferred_elbow=elbow_config, num_steps=num_waypoints, transfer_time=transfer_time)
                    if not t_plan["feasible"]:
                        st.sidebar.warning(f"⚠️ Continuous path infeasible: {t_plan['reason']}. Falling back to Point-to-Point.")
                        t_plan = plan_ptp_trajectory(q_p, q_pl, l1, l2, num_steps=num_waypoints, transfer_time=transfer_time)
                else:
                    t_plan = plan_ptp_trajectory(q_p, q_pl, l1, l2, num_steps=num_waypoints, transfer_time=transfer_time)

                for k in range(len(t_plan["theta1"])):
                    t1_k = t_plan["theta1"][k]
                    t2_k = t_plan["theta2"][k]
                    frames.append({
                        "step_name": f"{cycle_label}Step 4: Transferring box from PICK to PLACE ({t_plan['name']})",
                        "arm": forward_kinematics(t1_k, t2_k, l1, l2), "gripper_closed": True,
                        "box_held": True, "box_at_pick": False, "box_slide": 0.0,
                        "active_mode": "Point-to-point" if mode_kind == "ptp" else "Continuous path"
                    })
                # Phase 5: Gripper opens at PLACE (4 frames)
                fk_pl = forward_kinematics(q_pl[0], q_pl[1], l1, l2)
                for _ in range(4):
                    frames.append({
                        "step_name": f"{cycle_label}Step 5: Gripper opens, releasing box onto output conveyor",
                        "arm": fk_pl, "gripper_closed": False, "box_held": False,
                        "box_at_pick": False, "box_slide": 0.05,
                        "active_mode": "Point-to-point" if mode_kind == "ptp" else "Continuous path"
                    })
                # Phase 6: Box rides away on output conveyor (6 frames)
                for dist in np.linspace(0.1, 0.9, 6):
                    frames.append({
                        "step_name": f"{cycle_label}Step 6: Box rides away downstream on output conveyor",
                        "arm": fk_pl, "gripper_closed": False, "box_held": False,
                        "box_at_pick": False, "box_slide": float(dist),
                        "active_mode": "Point-to-point" if mode_kind == "ptp" else "Continuous path"
                    })
                # Phase 7: Arm returns from PLACE to Home (8 frames)
                for u in np.linspace(0, 1, 8):
                    s, _, _ = quintic_scaling(u)
                    t1 = q_pl[0] + s * (home_q[0] - q_pl[0])
                    t2 = q_pl[1] + s * (home_q[1] - q_pl[1])
                    frames.append({
                        "step_name": f"{cycle_label}Step 7: Arm returns home to rest position",
                        "arm": forward_kinematics(t1, t2, l1, l2), "gripper_closed": False,
                        "box_held": False, "box_at_pick": False, "box_slide": 1.0,
                        "active_mode": "Point-to-point" if mode_kind == "ptp" else "Continuous path"
                    })
                return frames

            # Assemble full animation frame sequence based on selection
            if auto_traj_type == "Point-to-point":
                all_anim_frames = build_single_cycle("ptp", "")
            elif auto_traj_type == "Continuous path":
                all_anim_frames = build_single_cycle("continuous", "")
            else:  # "Both"
                f1 = build_single_cycle("ptp", "[Cycle 1: PTP Arc] ")
                f2 = build_single_cycle("continuous", "[Cycle 2: Continuous Line] ")
                all_anim_frames = f1 + f2

            total_anim = len(all_anim_frames)

            with col_act:
                btn_p1, btn_p2, btn_p3 = st.columns(3)
                play_btn = btn_p1.button("▶ Play", width="stretch")
                pause_btn = btn_p2.button("⏸ Pause", width="stretch")
                reset_anim = btn_p3.button("↺ Reset", width="stretch")

            if reset_anim:
                st.session_state.auto_step = 0
                st.session_state.auto_playing = False

            if pause_btn:
                st.session_state.auto_playing = False

            timeline_idx = st.slider("Simulation Timeline Step", 0, total_anim - 1, int(st.session_state.auto_step))
            if not play_btn:
                st.session_state.auto_step = timeline_idx

            anim_spot = st.empty()
            chart_spot = st.empty()

            if play_btn:
                st.session_state.auto_playing = True
                for f_i in range(st.session_state.auto_step, total_anim):
                    f_data = all_anim_frames[f_i]
                    st.session_state.auto_step = f_i

                    grip_str = "Closed (Holding Box)" if f_data["box_held"] else ("Closed" if f_data["gripper_closed"] else "Open")
                    ee_pt = f_data["arm"]["ee"]
                    t1_val = np.degrees(np.arctan2(f_data["arm"]["elbow"][1], f_data["arm"]["elbow"][0]))
                    t2_val = np.degrees(f_data["arm"]["phi"]) - t1_val

                    anim_spot.markdown(f"""
                    <div class="status-card">
                        <div class="status-step">{f_data['step_name']}</div>
                        <div class="status-telemetry">
                            <b>Joint 1 (θ1):</b> {t1_val:.1f}° &nbsp;|&nbsp;
                            <b>Joint 2 (θ2):</b> {t2_val:.1f}° &nbsp;|&nbsp;
                            <b>Gripper Position:</b> ({ee_pt[0]:.2f}, {ee_pt[1]:.2f}) m &nbsp;|&nbsp;
                            <b>Gripper State:</b> {grip_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    fig_f = draw_2d_top_view(
                        arm_state=f_data["arm"],
                        l1=l1, l2=l2,
                        pick_pos=pick_pos, place_pos=place_pos,
                        gripper_closed=f_data["gripper_closed"],
                        box_held=f_data["box_held"],
                        box_at_pick=f_data["box_at_pick"],
                        box_delivered_offset=f_data["box_slide"],
                        traj_ptp=static_ptp, traj_cart=static_cart,
                        chosen_strategy=f_data.get("active_mode", auto_traj_type)
                    )
                    chart_spot.plotly_chart(fig_f, width="stretch", key=f"auto_workcell_frame_{f_i}")
                    time.sleep(0.04)
                st.session_state.auto_playing = False
                st.success("✅ Automatic Assembly Task Cycle Complete!")
            else:
                active_frame = all_anim_frames[st.session_state.auto_step]
                grip_str = "Closed (Holding Box)" if active_frame["box_held"] else ("Closed" if active_frame["gripper_closed"] else "Open")
                ee_pt = active_frame["arm"]["ee"]
                t1_val = np.degrees(np.arctan2(active_frame["arm"]["elbow"][1], active_frame["arm"]["elbow"][0]))
                t2_val = np.degrees(active_frame["arm"]["phi"]) - t1_val

                anim_spot.markdown(f"""
                <div class="status-card">
                    <div class="status-step">{active_frame['step_name']}</div>
                    <div class="status-telemetry">
                        <b>Joint 1 (θ1):</b> {t1_val:.1f}° &nbsp;|&nbsp;
                        <b>Joint 2 (θ2):</b> {t2_val:.1f}° &nbsp;|&nbsp;
                        <b>Gripper Position:</b> ({ee_pt[0]:.2f}, {ee_pt[1]:.2f}) m &nbsp;|&nbsp;
                        <b>Gripper State:</b> {grip_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig_stat = draw_2d_top_view(
                    arm_state=active_frame["arm"],
                    l1=l1, l2=l2,
                    pick_pos=pick_pos, place_pos=place_pos,
                    gripper_closed=active_frame["gripper_closed"],
                    box_held=active_frame["box_held"],
                    box_at_pick=active_frame["box_at_pick"],
                    box_delivered_offset=active_frame["box_slide"],
                    traj_ptp=static_ptp, traj_cart=static_cart,
                    chosen_strategy=active_frame.get("active_mode", auto_traj_type)
                )
                chart_spot.plotly_chart(fig_stat, width="stretch", key="auto_workcell_static")

            st.markdown('<div class="caption-note">How to read the picture: Robot base is at (0,0). Blue line is Link 1, orange line is Link 2. Two black jaws form the gripper. Boxes arrive on the left input belt, stop at the dashed PICK station, and are transferred to the dashed PLACE station on the right output belt.</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2: KINEMATICS AND TRAJECTORY ANALYSIS
# =============================================================================
with tab_kinematics:
    st.markdown("### 🧮 Kinematics & Trajectory Benchmark Analysis")

    k_tab1, k_tab2, k_tab3 = st.tabs([
        "Forward Kinematics & Matrices",
        "Inverse Kinematics & Reachability",
        "PTP vs Continuous Benchmark"
    ])

    # 1. FORWARD KINEMATICS & MATRICES
    with k_tab1:
        st.markdown("#### Forward Kinematics (FK) Interactive Calculator")
        fk_c1, fk_c2 = st.columns(2)
        with fk_c1:
            test_t1 = st.slider("Joint 1 (θ1) deg", -t1_lim_deg, t1_lim_deg, 30.0, 1.0, key="fk_tab_t1")
            test_t2 = st.slider("Joint 2 (θ2) deg", -t2_lim_deg, t2_lim_deg, 45.0, 1.0, key="fk_tab_t2")

            fk_res = forward_kinematics(np.radians(test_t1), np.radians(test_t2), l1, l2)
            st.write(f"- **Elbow Position $(X_1, Y_1)$:** `({fk_res['elbow'][0]:.3f}, {fk_res['elbow'][1]:.3f}) m`")
            st.write(f"- **Tool Position $(X_2, Y_2)$:** `({fk_res['ee'][0]:.3f}, {fk_res['ee'][1]:.3f}) m`")
            st.write(f"- **Total Reach Distance ($r$):** `{fk_res['reach']:.3f} m`")
            st.write(f"- **Total Tool Orientation ($\\phi$):** `{fk_res['phi_deg']:.1f}°`")

        with fk_c2:
            st.markdown("##### Planar Homogeneous Transformation Matrices ($3 \\times 3$)")
            st.markdown(r"**Base to Link 1 ($T_{01} = R_z(\theta_1) \cdot T_x(L_1)$):**")
            st.latex(f"T_{{01}} = {fk_res['T01_latex']}")
            st.markdown(r"**Link 1 to Link 2 ($T_{12} = R_z(\theta_2) \cdot T_x(L_2)$):**")
            st.latex(f"T_{{12}} = {fk_res['T12_latex']}")
            st.markdown(r"**Composite Base to Tool ($T_{02} = T_{01} \cdot T_{12}$):**")
            st.latex(f"T_{{02}} = {fk_res['T02_latex']}")

    # 2. INVERSE KINEMATICS & REACHABILITY
    with k_tab2:
        st.markdown("#### Inverse Kinematics (IK) & Target Reachability Verification")
        ik_c1, ik_c2 = st.columns([1, 1.2])

        with ik_c1:
            ik_eval_x = st.slider("Target Coordinate $X$ (m)", -4.0, 4.0, 2.0, 0.05, key="ik_verif_x")
            ik_eval_y = st.slider("Target Coordinate $Y$ (m)", -4.0, 4.0, 1.5, 0.05, key="ik_verif_y")
            ik_chk = solve_inverse_kinematics(ik_eval_x, ik_eval_y, l1, l2, t1_limits, t2_limits, preferred_elbow=elbow_config)

        with ik_c2:
            st.write(f"- **Target Distance ($r$):** `{ik_chk['distance']:.3f} m`")
            st.write(f"- **Workspace Reach Limits:** `[{ik_chk['r_min']:.2f} m, {ik_chk['r_max']:.2f} m]`")

            if not ik_chk["annulus_reachable"]:
                st.error(f"❌ Target Unreachable: {ik_chk['error_message']}")
            else:
                st.success("✅ Target distance is within the annular workspace.")
                st.write(f"- **Cosine Law Value $c_2 = \\cos(\\theta_2)$:** `{ik_chk['c2']:.4f}`")

                r_br = ik_chk["elbow_right"]
                l_br = ik_chk["elbow_left"]
                sol_c1, sol_c2 = st.columns(2)
                with sol_c1:
                    st.write("**Elbow-Right (θ2 ≥ 0):**")
                    st.write(f"θ1 = `{r_br['theta1_deg']:.1f}°`, θ2 = `{r_br['theta2_deg']:.1f}°`")
                    st.write(f"Status: `{'Valid' if r_br['valid'] else r_br['message']}`")
                with sol_c2:
                    st.write("**Elbow-Left (θ2 ≤ 0):**")
                    st.write(f"θ1 = `{l_br['theta1_deg']:.1f}°`, θ2 = `{l_br['theta2_deg']:.1f}°`")
                    st.write(f"Status: `{'Valid' if l_br['valid'] else l_br['message']}`")

    # 3. PTP VS CONTINUOUS BENCHMARK
    with k_tab3:
        st.markdown("#### Performance Metric Benchmark (PTP vs. Continuous Path)")
        cmp_table = compute_comparison_metrics(static_ptp, static_cart, pick_pos, place_pos)
        st.dataframe(cmp_table, width="stretch", hide_index=True)

        # Engineering verdict callout
        diff_len = ((static_ptp["path_length"] - static_cart["path_length"]) / max(1e-5, static_ptp["path_length"])) * 100.0
        diff_j2 = np.degrees(static_cart["j2_travel"] - static_ptp["j2_travel"])
        if static_cart["feasible"]:
            st.markdown(f"""
            <div class="verdict-box">
                💡 <b>Engineering Verdict:</b> Continuous path is {diff_len:.1f}% shorter in Cartesian travel distance, but requires {diff_j2:.1f}° more elbow rotation.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"⚠️ Continuous path is infeasible ({static_cart['reason']}). Point-to-Point transfer is the only viable solution.")

        # Joint Angles vs Time Plot
        st.markdown("##### Joint Angles vs. Time Profile ($\theta_1(t), \theta_2(t)$)")
        fig_j_time = go.Figure()
        t_steps = static_ptp["times"]
        fig_j_time.add_trace(go.Scatter(x=t_steps, y=np.degrees(static_ptp["theta1"]), mode="lines", line=dict(color="#2563eb", width=2.5), name="PTP θ1 (deg)"))
        fig_j_time.add_trace(go.Scatter(x=t_steps, y=np.degrees(static_ptp["theta2"]), mode="lines", line=dict(color="#3b82f6", width=2.5, dash="dot"), name="PTP θ2 (deg)"))
        fig_j_time.add_trace(go.Scatter(x=t_steps, y=np.degrees(static_cart["theta1"]), mode="lines", line=dict(color="#ea580c", width=2.5), name="Continuous θ1 (deg)"))
        fig_j_time.add_trace(go.Scatter(x=t_steps, y=np.degrees(static_cart["theta2"]), mode="lines", line=dict(color="#f97316", width=2.5, dash="dot"), name="Continuous θ2 (deg)"))

        fig_j_time.update_layout(
            plot_bgcolor="#fff8ea", paper_bgcolor="#fff8ea",
            xaxis=dict(title="Time (seconds)", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="Joint Angle (degrees)", showgrid=True, gridcolor="#f1f5f9"),
            margin=dict(l=20, r=20, t=10, b=20),
            height=340,
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_j_time, width="stretch", key="joint_angles_time_profile_chart")


# =============================================================================
# TAB 3: DOCUMENTATION
# =============================================================================
with tab_documentation:
    st.markdown("### 📚 Technical Documentation & Mathematical Formulations")
    st.markdown("#### Subject: Robot Kinematics and Control (Mini Project 2)")

    st.markdown(r"""
#### 1. System Parameters & Configuration
- **Manipulator Classification:** Planar 2R SCARA-style arm with two revolute joints operating in the horizontal $X$-$Y$ plane, equipped with a parallel two-jaw mechanical gripper.
- **Link 1 Length ($L_1$):** Default $2.0\text{ m}$ (Shoulder to Elbow boom).
- **Link 2 Length ($L_2$):** Default $1.5\text{ m}$ (Elbow to Tool wrist).
- **Joint Limits:** Joint 1 $\pm 170^\circ$, Joint 2 $\pm 150^\circ$.
- **Pick Station:** Located on the input conveyor belt at default $(-2.2, 2.0)\text{ m}$.
- **Place Station:** Located on the output conveyor belt at default $(2.2, 2.0)\text{ m}$.

---

#### 2. Engineering Assumptions
1. **Planar Motion:** The arm links move strictly in a two-dimensional horizontal plane.
2. **Rigid Link Geometry:** Elastic deflections, torsional bending, and joint mechanical backlash are considered zero.
3. **Quasi-Static Gripper:** The parallel gripper jaws actuate symmetrically to hold and release rigid cardboard boxes without changing the kinematic chain length.
4. **Frictionless Conveyor Slip:** Workpieces slide smoothly along the conveyor surface when released.

---

#### 3. Standard Denavit-Hartenberg (DH) Parameter Table
For the planar 2-link revolute kinematic chain:

| Link $i$ | Joint Type | Joint Angle ($\theta_i$) | Link Offset ($d_i$) | Link Length ($a_i$) | Link Twist ($\alpha_i$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Revolute (Base/Shoulder) | $\theta_1^*$ | $0$ | $L_1$ | $0^\circ$ |
| **2** | Revolute (Elbow) | $\theta_2^*$ | $0$ | $L_2$ | $0^\circ$ |

---

#### 4. Forward Kinematics (FK) Formulation
Given joint angles $\theta_1$ and $\theta_2$:
- **Elbow Position (Joint 1):**
  $$x_1 = L_1 \cos(\theta_1), \quad y_1 = L_1 \sin(\theta_1)$$
- **End-Effector Position:**
  $$x_2 = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2)$$
  $$y_2 = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2)$$
- **Tool Orientation Angle:**
  $$\phi = \theta_1 + \theta_2$$

---

#### 5. Planar Homogeneous Transformation Matrices ($3 \times 3$)
Each link transformation is expressed as a pure rotation about the $Z$-axis followed by translation along $X$:
$$T = R_z(\theta) \cdot T_x(L)$$

$$T_{01} = \begin{bmatrix} \cos\theta_1 & -\sin\theta_1 & L_1 \cos\theta_1 \\ \sin\theta_1 & \cos\theta_1 & L_1 \sin\theta_1 \\ 0 & 0 & 1 \end{bmatrix}, \quad T_{12} = \begin{bmatrix} \cos\theta_2 & -\sin\theta_2 & L_2 \cos\theta_2 \\ \sin\theta_2 & \cos\theta_2 & L_2 \sin\theta_2 \\ 0 & 0 & 1 \end{bmatrix}$$

$$T_{02} = T_{01} \cdot T_{12} = \begin{bmatrix} \cos(\theta_1+\theta_2) & -\sin(\theta_1+\theta_2) & x_2 \\ \sin(\theta_1+\theta_2) & \cos(\theta_1+\theta_2) & y_2 \\ 0 & 0 & 1 \end{bmatrix}$$

---

#### 6. Analytic Inverse Kinematics (IK)
Given Cartesian target coordinates $(x, y)$:
1. **Target Distance Squared:** $r^2 = x^2 + y^2$
2. **Law of Cosines on Link Triangle:**
   $$c_2 = \cos(\theta_2) = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 L_1 L_2}$$
3. **Reachability Check:**
   Target is physically reachable if and only if $|c_2| \le 1$, which equates to the annular constraint:
   $$|L_1 - L_2| \le \sqrt{x^2 + y^2} \le L_1 + L_2$$
4. **Elbow Solutions:**
   $$\theta_2 = \pm \arccos(c_2)$$
   - Positive branch ($+$): **Elbow-Right** ($\theta_2 \ge 0$)
   - Negative branch ($-$): **Elbow-Left** ($\theta_2 \le 0$)
5. **Joint 1 Angle:**
   $$\theta_1 = \text{atan2}(y, x) - \text{atan2}(L_2 \sin\theta_2, L_1 + L_2 \cos\theta_2)$$
6. Angles are wrapped strictly to $(-\pi, \pi]$ and verified against $[\theta_{min}, \theta_{max}]$.

---

#### 7. Trajectory Planning with Quintic Time Scaling
Trajectory interpolation uses normalized quintic polynomial time scaling $s(u)$ where $u = t / T \in [0, 1]$:
$$s(u) = 10u^3 - 15u^4 + 6u^5$$

**Derivatives:**
$$\dot{s}(u) = \frac{1}{T} \left(30u^2 - 60u^3 + 30u^4\right)$$
$$\ddot{s}(u) = \frac{1}{T^2} \left(60u - 180u^2 + 120u^3\right)$$

**Boundary Conditions Satisfied:**
- $s(0) = 0, \quad s(1) = 1$
- $\dot{s}(0) = 0, \quad \dot{s}(1) = 0$ (Smooth zero velocity start/stop)
- $\ddot{s}(0) = 0, \quad \ddot{s}(1) = 0$ (Zero initial/final acceleration)

---

#### 8. Performance Measures
- **Cartesian Path Length ($S$):** $S = \sum_{k=1}^{N-1} \sqrt{(x_{k+1}-x_k)^2 + (y_{k+1}-y_k)^2}$
- **Straight-Line Distance ($D$):** $D = \sqrt{(x_{place}-x_{pick})^2 + (y_{place}-y_{pick})^2}$
- **Path Efficiency ($\eta$):** $\eta = \frac{D}{S} \times 100\%$
- **Maximum Deviation:** $\max_k \left( \frac{|(y_e - y_s) x_k - (x_e - x_s) y_k + x_e y_s - y_e x_s|}{D} \right)$
- **Joint Travel:** $\Delta \theta_i = \sum |\theta_{i, k+1} - \theta_{i, k}|$
    """)
