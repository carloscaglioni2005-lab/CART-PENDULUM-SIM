import io
import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import FancyBboxPatch

from physics_sim import (
    SimulationParams,
    constant_force_value,
    reference_angle_for_constant_acceleration,
    simulate_system,
)

st.set_page_config(
    page_title="Simulatore Carrello-Pendolo",
    layout="wide",
)

SCENARIO_PRESETS = {
    "Forzamento armonico": {
        "description": "Una spinta sinusoidale continua mette in evidenza il comportamento accoppiato tra carrello e pendolo.",
        "cart_mass": 1.5,
        "cart_damping": 0.18,
        "pendulum_damping": 0.04,
        "pendulum_mass": 0.35,
        "rod_length": 0.9,
        "force_expression": "1.8 * sin(1.6 * t)",
        "torque_expression": "0",
        "gravity": 9.81,
        "x0": 0.0,
        "v0": 0.0,
        "theta0_deg": 0.0,
        "omega0_deg": 0.0,
        "duration": 12.0,
        "time_step": 0.01,
    },
    "Oscillazione libera": {
        "description": "Il carrello non e forzato: il pendolo parte inclinato e rilascia la sua energia al sistema.",
        "cart_mass": 1.4,
        "cart_damping": 0.10,
        "pendulum_damping": 0.03,
        "pendulum_mass": 0.30,
        "rod_length": 1.0,
        "force_expression": "0",
        "torque_expression": "0",
        "gravity": 9.81,
        "x0": 0.0,
        "v0": 0.0,
        "theta0_deg": 18.0,
        "omega0_deg": 0.0,
        "duration": 14.0,
        "time_step": 0.01,
    },
    "Impulso breve": {
        "description": "Una forza rettangolare breve spinge il carrello e poi lo lascia oscillare liberamente.",
        "cart_mass": 1.7,
        "cart_damping": 0.14,
        "pendulum_damping": 0.05,
        "pendulum_mass": 0.35,
        "rod_length": 0.9,
        "force_expression": "2.8 * Heaviside(t - 0.5) - 2.8 * Heaviside(t - 1.0)",
        "torque_expression": "0",
        "gravity": 9.81,
        "x0": 0.0,
        "v0": 0.0,
        "theta0_deg": 6.0,
        "omega0_deg": 0.0,
        "duration": 12.0,
        "time_step": 0.01,
    },
    "Accelerazione costante": {
        "description": "Una forza costante accelera il sistema nel transitorio; con smorzamento il carrello tende poi a una velocita quasi costante e il pendolo si raddrizza.",
        "cart_mass": 1.8,
        "cart_damping": 0.20,
        "pendulum_damping": 0.05,
        "pendulum_mass": 0.40,
        "rod_length": 0.95,
        "force_expression": "1.3",
        "torque_expression": "0",
        "gravity": 9.81,
        "x0": 0.0,
        "v0": 0.0,
        "theta0_deg": 0.0,
        "omega0_deg": 0.0,
        "duration": 10.0,
        "time_step": 0.01,
    },
    "Swing-up energico": {
        "description": "Una forzante intensa porta il sistema in un regime non lineare con oscillazioni molto ampie.",
        "cart_mass": 1.2,
        "cart_damping": 0.08,
        "pendulum_damping": 0.01,
        "pendulum_mass": 0.25,
        "rod_length": 0.85,
        "force_expression": "6.0 * sin(2.3 * t)",
        "torque_expression": "0",
        "gravity": 9.81,
        "x0": 0.0,
        "v0": 0.0,
        "theta0_deg": 10.0,
        "omega0_deg": 0.0,
        "duration": 14.0,
        "time_step": 0.008,
    },
    "Smorzamento forte": {
        "description": "Attriti piu elevati fanno decadere velocemente l'energia e riportano il sistema verso l'equilibrio.",
        "cart_mass": 1.5,
        "cart_damping": 0.55,
        "pendulum_damping": 0.18,
        "pendulum_mass": 0.35,
        "rod_length": 0.9,
        "force_expression": "0",
        "torque_expression": "0",
        "gravity": 9.81,
        "x0": 0.0,
        "v0": 0.0,
        "theta0_deg": 22.0,
        "omega0_deg": 0.0,
        "duration": 12.0,
        "time_step": 0.01,
    },
}
DEFAULT_PRESET_NAME = "Forzamento armonico"

DISPLAY_CART_WIDTH = 0.72
DISPLAY_CART_HEIGHT = 0.34
DISPLAY_PENDULUM_LENGTH = 0.92
DISPLAY_X_HALF_SPAN = 3.2
DISPLAY_Y_MIN = -1.95
DISPLAY_Y_MAX = 1.35

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(251, 191, 36, 0.18), transparent 22%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.22), transparent 26%),
            linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
    }
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.2rem;
        max-width: 1500px;
    }
    .hero-card,
    .control-card,
    .plot-card,
    .insight-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.10);
        border-radius: 24px;
        backdrop-filter: blur(12px);
    }
    .hero-card {
        padding: 1.2rem 1.4rem 0.8rem 1.4rem;
        margin-bottom: 1rem;
    }
    .control-card {
        padding: 1.1rem 1rem 0.4rem 1rem;
        margin-bottom: 1rem;
    }
    .plot-card {
        padding: 0.8rem 1rem 0.4rem 1rem;
        margin-top: 1rem;
    }
    .eyebrow {
        color: #0f766e;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        color: #0f172a;
        font-size: 2.35rem;
        line-height: 1.05;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .hero-copy {
        color: #334155;
        font-size: 1rem;
        max-width: 56rem;
        margin-bottom: 0.7rem;
    }
    .metric-row,
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin-top: 0.4rem;
    }
    .metric-pill,
    .insight-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(241,245,249,0.96));
        border: 1px solid rgba(203, 213, 225, 0.9);
        border-radius: 18px;
        padding: 0.85rem 0.95rem;
    }
    .metric-label,
    .insight-label {
        color: #475569;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.15;
        margin-top: 0.15rem;
    }
    .insight-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 800;
        margin-top: 0.1rem;
        margin-bottom: 0.35rem;
    }
    .insight-copy {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .panel-title {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .panel-copy {
        color: #475569;
        font-size: 0.92rem;
        margin-bottom: 0.8rem;
    }
    .preset-note {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.45;
        margin-top: 0.35rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _apply_preset(preset_name: str) -> None:
    preset = SCENARIO_PRESETS[preset_name]
    for key, value in preset.items():
        if key != "description":
            st.session_state[key] = value


def _ensure_default_state() -> None:
    if st.session_state.get("_preset_initialized"):
        return

    st.session_state["selected_preset"] = DEFAULT_PRESET_NAME
    _apply_preset(DEFAULT_PRESET_NAME)
    st.session_state["_preset_initialized"] = True


def _style_axes(axes) -> None:
    for ax in axes:
        ax.set_facecolor("#fcfdff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.grid(True, alpha=0.22)


def _build_overview_figure(result, reference_angle_deg: float | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=140)
    fig.patch.set_facecolor("#ffffff")
    time = result.time

    cart_velocity_axis = axes[0].twinx()
    cart_position_line = axes[0].plot(time, result.cart_position, color="#0f766e", linewidth=2.3, label="x [m]")[0]
    cart_velocity_line = cart_velocity_axis.plot(
        time,
        result.cart_velocity,
        color="#0ea5e9",
        linewidth=1.8,
        label="v [m/s]",
    )[0]
    axes[0].set_title("Carrello", color="#0f172a", fontweight="bold")
    axes[0].set_xlabel("Tempo [s]")
    axes[0].set_ylabel("x [m]", color="#0f766e")
    cart_velocity_axis.set_ylabel("v [m/s]", color="#0ea5e9")
    axes[0].tick_params(axis="y", colors="#0f766e")
    cart_velocity_axis.tick_params(axis="y", colors="#0ea5e9")
    axes[0].legend([cart_position_line, cart_velocity_line], ["x [m]", "v [m/s]"], frameon=False, loc="upper right")

    omega_axis = axes[1].twinx()
    theta_line = axes[1].plot(time, result.theta_deg, color="#0f766e", linewidth=2.4, label="theta [deg]")[0]
    omega_line = omega_axis.plot(time, result.omega_deg, color="#7c3aed", linewidth=1.9, label="omega [deg/s]")[0]
    reference_line = axes[1].axhline(0.0, color="#94a3b8", linewidth=1.0, alpha=0.75, label="theta = 0")
    legend_lines = [theta_line, omega_line, reference_line]
    legend_labels = ["theta [deg]", "omega [deg/s]", "theta = 0"]
    if reference_angle_deg is not None:
        accel_line = axes[1].axhline(
            reference_angle_deg,
            color="#ef4444",
            linewidth=1.6,
            linestyle="--",
            alpha=0.9,
            label="rif. accel. [deg]",
        )
        legend_lines.append(accel_line)
        legend_labels.append("rif. accel. [deg]")
    axes[1].set_title("Pendolo", color="#0f172a", fontweight="bold")
    axes[1].set_xlabel("Tempo [s]")
    axes[1].set_ylabel("theta [deg]", color="#0f766e")
    omega_axis.set_ylabel("omega [deg/s]", color="#7c3aed")
    axes[1].tick_params(axis="y", colors="#0f766e")
    omega_axis.tick_params(axis="y", colors="#7c3aed")
    axes[1].legend(legend_lines, legend_labels, frameon=False, loc="upper right")

    axes[2].plot(time, result.force, color="#d97706", linewidth=2.2)
    axes[2].axhline(0.0, color="#94a3b8", linewidth=1.0, alpha=0.75)
    axes[2].set_title("Forza esterna", color="#0f172a", fontweight="bold")
    axes[2].set_xlabel("Tempo [s]")
    axes[2].set_ylabel("F [N]")

    _style_axes(axes)
    for secondary_axis in (cart_velocity_axis, omega_axis):
        secondary_axis.set_facecolor("none")
        secondary_axis.spines["top"].set_visible(False)
        secondary_axis.spines["left"].set_visible(False)
        secondary_axis.spines["right"].set_color("#cbd5e1")
        secondary_axis.grid(False)
    fig.tight_layout()
    return fig


def _build_energy_figure(result) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), dpi=140)
    fig.patch.set_facecolor("#ffffff")
    drift = result.total_energy - result.total_energy[0]

    axes[0].plot(result.time, result.kinetic_energy, color="#0ea5e9", linewidth=2.1)
    axes[0].plot(result.time, result.potential_energy, color="#f97316", linewidth=2.1)
    axes[0].plot(result.time, result.total_energy, color="#0f172a", linewidth=2.3)
    axes[0].set_title("Bilancio energetico", color="#0f172a", fontweight="bold")
    axes[0].set_xlabel("Tempo [s]")
    axes[0].set_ylabel("Energia [J]")
    axes[0].legend(["Cinetica", "Potenziale", "Totale"], frameon=False)

    axes[1].plot(result.time, drift, color="#dc2626", linewidth=2.2)
    axes[1].fill_between(result.time, 0.0, drift, color="#fecaca", alpha=0.45)
    axes[1].axhline(0.0, color="#94a3b8", linewidth=1.0, alpha=0.75)
    axes[1].set_title("Deriva dell'energia totale", color="#0f172a", fontweight="bold")
    axes[1].set_xlabel("Tempo [s]")
    axes[1].set_ylabel("Delta E [J]")

    _style_axes(axes)
    fig.tight_layout()
    return fig


def _build_phase_figure(result, reference_angle_deg: float | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), dpi=140)
    fig.patch.set_facecolor("#ffffff")

    axes[0].plot(result.cart_position, result.cart_velocity, color="#0f766e", linewidth=2.0)
    axes[0].scatter(result.cart_position[0], result.cart_velocity[0], color="#2563eb", s=55, zorder=5)
    axes[0].scatter(result.cart_position[-1], result.cart_velocity[-1], color="#f97316", s=55, zorder=5)
    axes[0].set_title("Ritratto di fase del carrello", color="#0f172a", fontweight="bold")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("v [m/s]")

    axes[1].plot(result.theta_deg, result.omega_deg, color="#7c3aed", linewidth=2.0)
    axes[1].scatter(result.theta_deg[0], result.omega_deg[0], color="#2563eb", s=55, zorder=5)
    axes[1].scatter(result.theta_deg[-1], result.omega_deg[-1], color="#f97316", s=55, zorder=5)
    if reference_angle_deg is not None:
        axes[1].axvline(reference_angle_deg, color="#ef4444", linewidth=1.6, linestyle="--", alpha=0.9)
    axes[1].set_title("Ritratto di fase del pendolo", color="#0f172a", fontweight="bold")
    axes[1].set_xlabel("theta [deg]")
    axes[1].set_ylabel("omega [deg/s]")

    _style_axes(axes)
    fig.tight_layout()
    return fig


def _scene_geometry(cart_width: float, cart_height: float) -> dict[str, float]:
    wheel_radius = cart_height * 0.20
    wheel_y = wheel_radius
    body_bottom = wheel_y + wheel_radius + 0.035
    body_top = body_bottom + cart_height
    pivot_y = body_bottom - 0.05
    wheel_offset = cart_width * 0.32
    arrow_y = body_top + 0.36
    return {
        "wheel_radius": wheel_radius,
        "wheel_y": wheel_y,
        "body_bottom": body_bottom,
        "body_top": body_top,
        "pivot_y": pivot_y,
        "wheel_offset": wheel_offset,
        "arrow_y": arrow_y,
    }


def _camera_base_half_span() -> float:
    return DISPLAY_X_HALF_SPAN


def _camera_window(result) -> tuple[float, float]:
    camera_center = float(0.5 * (np.min(result.cart_position) + np.max(result.cart_position)))
    base_half_span = _camera_base_half_span()
    return (camera_center - base_half_span, camera_center + base_half_span)


def _wrap_position(x: float, xlim: tuple[float, float]) -> float:
    xmin, xmax = xlim
    width = xmax - xmin
    return xmin + ((x - xmin) % width)


def _wrapped_system_positions(x: float, xlim: tuple[float, float]) -> list[float]:
    xmin, xmax = xlim
    width = xmax - xmin
    base_x = _wrap_position(x, xlim)
    candidates = [base_x - width, base_x, base_x + width]
    margin = width * 0.15
    return [candidate for candidate in candidates if xmin - margin <= candidate <= xmax + margin]


def _draw_system(ax, draw_x: float, theta: float, force: float, length: float, cart_width: float, cart_height: float) -> None:
    geometry = _scene_geometry(cart_width, cart_height)
    pivot_y = geometry["pivot_y"]
    bob_x = draw_x + length * np.sin(theta)
    bob_y = pivot_y - length * np.cos(theta)

    wheel_radius = geometry["wheel_radius"]
    wheel_y = geometry["wheel_y"]
    wheel_offset = geometry["wheel_offset"]
    for wheel_x in (draw_x - wheel_offset, draw_x + wheel_offset):
        ax.add_patch(plt.Circle((wheel_x, wheel_y), wheel_radius, facecolor="#0f172a", edgecolor="#0f172a", zorder=5))
        ax.add_patch(plt.Circle((wheel_x, wheel_y), wheel_radius * 0.46, facecolor="#94a3b8", edgecolor="none", zorder=6))

    ax.add_patch(
        FancyBboxPatch(
            (draw_x - cart_width / 2, geometry["body_bottom"]),
            cart_width,
            cart_height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#0ea5e9",
            edgecolor="#0f172a",
            linewidth=2.2,
            zorder=4,
        )
    )
    ax.plot([draw_x, draw_x], [geometry["body_bottom"], geometry["pivot_y"]], color="#0f172a", linewidth=3.2, zorder=7)
    ax.plot([draw_x, bob_x], [pivot_y, bob_y], color="#0f172a", linewidth=3.6, zorder=7)
    ax.scatter([draw_x], [pivot_y], s=72, color="#0f172a", zorder=8)
    ax.scatter([bob_x], [bob_y], s=520, color="#f97316", edgecolors="#9a3412", linewidths=2, zorder=8)
    ax.scatter([bob_x], [bob_y], s=120, color="#fdba74", edgecolors="none", zorder=9, alpha=0.8)

    if abs(force) > 1e-6:
        arrow_scale = min(1.7, 0.34 + 0.17 * abs(force))
        arrow_length = np.sign(force) * arrow_scale
        ax.arrow(
            draw_x,
            geometry["arrow_y"],
            arrow_length,
            0.0,
            width=0.026,
            head_width=0.16,
            head_length=0.16,
            color="#7c3aed",
            length_includes_head=True,
            zorder=10,
        )


def _build_scene_figure(
    result,
    index: int,
    cart_width: float,
    cart_height: float,
    reference_angle_rad: float | None = None,
) -> plt.Figure:
    x = result.cart_position[index]
    theta = result.theta[index]
    force = result.force[index]

    geometry = _scene_geometry(cart_width, cart_height)

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fbff")

    xlim = _camera_window(result)
    ylim = (DISPLAY_Y_MIN, DISPLAY_Y_MAX)

    ax.axhspan(0.0, ylim[1], color="#eff6ff", alpha=0.55)
    ax.axhspan(ylim[0], 0.0, color="#fef3c7", alpha=0.32)
    ax.axhline(0.0, color="#334155", linewidth=3.2, zorder=1)

    sleeper_positions = np.linspace(xlim[0], xlim[1], 14)
    for pos in sleeper_positions:
        ax.plot([pos - 0.22, pos + 0.22], [-0.08, -0.08], color="#64748b", linewidth=1.2, alpha=0.55, zorder=1)

    wrapped_positions = _wrapped_system_positions(float(x), xlim)
    for draw_x in wrapped_positions:
        if reference_angle_rad is not None:
            pivot_y = geometry["pivot_y"]
            ref_bob_x = draw_x + DISPLAY_PENDULUM_LENGTH * np.sin(reference_angle_rad)
            ref_bob_y = pivot_y - DISPLAY_PENDULUM_LENGTH * np.cos(reference_angle_rad)
            ax.plot([draw_x, ref_bob_x], [pivot_y, ref_bob_y], color="#ef4444", linewidth=2.0, linestyle="--", alpha=0.45, zorder=3)
            ax.scatter([ref_bob_x], [ref_bob_y], s=120, color="#fecaca", edgecolors="#ef4444", linewidths=1.2, alpha=0.45, zorder=3)
        _draw_system(ax, draw_x, float(theta), float(force), DISPLAY_PENDULUM_LENGTH, cart_width, cart_height)

    if wrapped_positions and abs(force) > 1e-6:
        primary_x = min(wrapped_positions, key=lambda candidate: abs(candidate - (xlim[0] + xlim[1]) * 0.5))
        arrow_scale = min(1.7, 0.34 + 0.17 * abs(force))
        arrow_length = np.sign(force) * arrow_scale
        ax.text(
            primary_x + arrow_length * 0.52,
            geometry["arrow_y"] + 0.21,
            f"F = {force:.2f} N",
            ha="center",
            va="bottom",
            color="#5b21b6",
            fontsize=11,
            fontweight="bold",
            zorder=11,
        )

    ax.text(
        0.02,
        0.96,
        f"t = {result.time[index]:.2f} s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.02,
        0.90,
        f"x = {x:.3f} m   |   theta = {result.theta_deg[index]:.2f} deg   |   v = {result.cart_velocity[index]:.3f} m/s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#334155",
    )
    if reference_angle_rad is not None:
        ax.text(
            0.02,
            0.84,
            f"theta_ref = {np.rad2deg(reference_angle_rad):.2f} deg",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color="#b91c1c",
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig


def _scene_limits(result, cart_height: float) -> tuple[tuple[float, float], tuple[float, float]]:
    geometry = _scene_geometry(DISPLAY_CART_WIDTH, cart_height)
    xlim = _camera_window(result)
    ylim = (DISPLAY_Y_MIN, max(DISPLAY_Y_MAX, geometry["body_top"] + 0.45))
    return xlim, ylim


def _build_scene_animation_html(
    result,
    cart_width: float,
    cart_height: float,
    time_step: float,
    max_frames: int = 420,
) -> str:
    total_frames = len(result.time)
    frame_step = max(1, int(np.ceil(total_frames / max_frames)))
    frame_indices = np.arange(0, total_frames, frame_step, dtype=int)
    if frame_indices[-1] != total_frames - 1:
        frame_indices = np.append(frame_indices, total_frames - 1)

    geometry = _scene_geometry(cart_width, cart_height)
    xlim, ylim = _scene_limits(result, cart_height)
    payload = {
        "time": np.round(result.time[frame_indices], 5).tolist(),
        "x": np.round(result.cart_position[frame_indices], 6).tolist(),
        "theta": np.round(result.theta[frame_indices], 6).tolist(),
        "theta_deg": np.round(result.theta_deg[frame_indices], 4).tolist(),
        "velocity": np.round(result.cart_velocity[frame_indices], 6).tolist(),
        "force": np.round(result.force[frame_indices], 6).tolist(),
        "length": float(DISPLAY_PENDULUM_LENGTH),
        "cart_width": float(cart_width),
        "cart_height": float(cart_height),
        "wheel_radius": float(geometry["wheel_radius"]),
        "wheel_y": float(geometry["wheel_y"]),
        "body_bottom": float(geometry["body_bottom"]),
        "body_top": float(geometry["body_top"]),
        "pivot_y": float(geometry["pivot_y"]),
        "wheel_offset": float(geometry["wheel_offset"]),
        "arrow_y": float(geometry["arrow_y"]),
        "xlim": [float(xlim[0]), float(xlim[1])],
        "scene_width": float(xlim[1] - xlim[0]),
        "ylim": [float(ylim[0]), float(ylim[1])],
        "interval_ms": max(20, int(time_step * frame_step * 1000)),
    }
    payload_json = json.dumps(payload)

    return f"""
    <div id="sim-root" style="background:rgba(255,255,255,0.9);border:1px solid rgba(203,213,225,0.8);border-radius:24px;padding:16px 16px 12px 16px;box-shadow:0 18px 42px rgba(15,23,42,0.08);">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;">
        <div>
          <div style="color:#0f766e;font-size:12px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;">Animazione</div>
          <div style="color:#0f172a;font-size:28px;font-weight:800;line-height:1.05;">Simulazione carrello-pendolo</div>
          <div style="color:#475569;font-size:14px;margin-top:6px;">Carrello su guida orizzontale con pendolo collegato sotto il centro mediante un'asta rigida inestensibile.</div>
        </div>
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
          <button id="start-btn" style="border:none;background:#0f766e;color:white;padding:10px 16px;border-radius:999px;font-weight:700;cursor:pointer;">Start</button>
          <button id="stop-btn" style="border:none;background:#e2e8f0;color:#0f172a;padding:10px 16px;border-radius:999px;font-weight:700;cursor:pointer;">Stop</button>
          <button id="reset-btn" style="border:none;background:#dbeafe;color:#1d4ed8;padding:10px 16px;border-radius:999px;font-weight:700;cursor:pointer;">Reset</button>
        </div>
      </div>
      <canvas id="sim-canvas" style="width:100%;height:520px;display:block;margin-top:14px;border-radius:18px;background:linear-gradient(180deg,#f8fbff 0%,#fffdf5 100%);"></canvas>
      <div style="display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;margin-top:12px;">
        <input id="time-slider" type="range" min="0" max="1" value="0" step="1" style="width:100%;" />
        <div id="time-label" style="min-width:130px;text-align:right;color:#334155;font-weight:700;">t = 0.00 s</div>
      </div>
      <div id="state-label" style="margin-top:8px;color:#475569;font-size:14px;">x = 0.000 m | theta = 0.00 deg | v = 0.000 m/s | F = 0.00 N</div>
    </div>
    <script>
    (() => {{
      const data = {payload_json};
      const canvas = document.getElementById("sim-canvas");
      const ctx = canvas.getContext("2d");
      const startBtn = document.getElementById("start-btn");
      const stopBtn = document.getElementById("stop-btn");
      const resetBtn = document.getElementById("reset-btn");
      const slider = document.getElementById("time-slider");
      const timeLabel = document.getElementById("time-label");
      const stateLabel = document.getElementById("state-label");

      let frame = 0;
      let timer = null;
      slider.max = Math.max(0, data.time.length - 1).toString();

      function resizeCanvas() {{
        const ratio = window.devicePixelRatio || 1;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        canvas.width = Math.floor(width * ratio);
        canvas.height = Math.floor(height * ratio);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        drawFrame(frame);
      }}

      function wrapX(x) {{
        const width = data.scene_width;
        const xmin = data.xlim[0];
        return xmin + (((x - xmin) % width) + width) % width;
      }}

      function systemCopies(x) {{
        const base = wrapX(x);
        const width = data.scene_width;
        const margin = width * 0.15;
        return [base - width, base, base + width].filter((candidate) => candidate >= data.xlim[0] - margin && candidate <= data.xlim[1] + margin);
      }}

      function worldToCanvas(x, y) {{
        const [xmin, xmax] = data.xlim;
        const padX = 34;
        const padY = 26;
        const usableW = canvas.clientWidth - 2 * padX;
        const usableH = canvas.clientHeight - 2 * padY;
        const px = padX + (x - xmin) / (xmax - xmin) * usableW;
        const py = padY + (1 - (y - data.ylim[0]) / (data.ylim[1] - data.ylim[0])) * usableH;
        return [px, py];
      }}

      function worldLengthX(value) {{
        const usableW = canvas.clientWidth - 68;
        return value / data.scene_width * usableW;
      }}

      function worldLengthY(value) {{
        const usableH = canvas.clientHeight - 52;
        return value / (data.ylim[1] - data.ylim[0]) * usableH;
      }}

      function drawArrow(x1, y1, x2, y2) {{
        const headLength = 12;
        const angle = Math.atan2(y2 - y1, x2 - x1);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - headLength * Math.cos(angle - Math.PI / 6), y2 - headLength * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(x2 - headLength * Math.cos(angle + Math.PI / 6), y2 - headLength * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
      }}

      function drawFrame(index) {{
        frame = index;
        slider.value = String(index);
        ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

        const x = data.x[index];
        const theta = data.theta[index];
        const force = data.force[index];

        const [groundX1, groundY] = worldToCanvas(data.xlim[0], 0);
        const [groundX2] = worldToCanvas(data.xlim[1], 0);
        ctx.fillStyle = "#eef6ff";
        ctx.fillRect(0, 0, canvas.clientWidth, groundY);
        ctx.fillStyle = "#fef3c7";
        ctx.fillRect(0, groundY, canvas.clientWidth, canvas.clientHeight - groundY);

        ctx.strokeStyle = "#334155";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(groundX1, groundY);
        ctx.lineTo(groundX2, groundY);
        ctx.stroke();

        ctx.strokeStyle = "rgba(100,116,139,0.55)";
        ctx.lineWidth = 1.2;
        const sleepers = 14;
        for (let i = 0; i < sleepers; i += 1) {{
          const wx = data.xlim[0] + (i / (sleepers - 1)) * data.scene_width;
          const [sx] = worldToCanvas(wx, 0);
          ctx.beginPath();
          ctx.moveTo(sx - 12, groundY + 8);
          ctx.lineTo(sx + 12, groundY + 8);
          ctx.stroke();
        }}

        const wheelRadius = Math.max(10, worldLengthY(data.wheel_radius));
        function drawWheel(cx, cy) {{
          ctx.fillStyle = "#0f172a";
          ctx.beginPath();
          ctx.arc(cx, cy, wheelRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#94a3b8";
          ctx.beginPath();
          ctx.arc(cx, cy, wheelRadius * 0.45, 0, Math.PI * 2);
          ctx.fill();
        }}

        const systemCopiesX = systemCopies(x);
        const cartWidthPx = worldLengthX(data.cart_width);
        const cartHeightPx = worldLengthY(data.cart_height);
        const viewCenter = (data.xlim[0] + data.xlim[1]) / 2;
        const primaryX = systemCopiesX.length > 0
          ? systemCopiesX.reduce((best, candidate) => Math.abs(candidate - viewCenter) < Math.abs(best - viewCenter) ? candidate : best, systemCopiesX[0])
          : wrapX(x);

        systemCopiesX.forEach((drawX) => {{
          const bobX = drawX + data.length * Math.sin(theta);
          const bobY = data.pivot_y - data.length * Math.cos(theta);
          const [leftWheelX, leftWheelY] = worldToCanvas(drawX - data.wheel_offset, data.wheel_y);
          const [rightWheelX, rightWheelY] = worldToCanvas(drawX + data.wheel_offset, data.wheel_y);
          const [cartLeft, cartTop] = worldToCanvas(drawX - data.cart_width / 2, data.body_top);

          ctx.fillStyle = "#0ea5e9";
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 2;
          const radius = 16;
          ctx.beginPath();
          ctx.moveTo(cartLeft + radius, cartTop);
          ctx.lineTo(cartLeft + cartWidthPx - radius, cartTop);
          ctx.quadraticCurveTo(cartLeft + cartWidthPx, cartTop, cartLeft + cartWidthPx, cartTop + radius);
          ctx.lineTo(cartLeft + cartWidthPx, cartTop + cartHeightPx - radius);
          ctx.quadraticCurveTo(cartLeft + cartWidthPx, cartTop + cartHeightPx, cartLeft + cartWidthPx - radius, cartTop + cartHeightPx);
          ctx.lineTo(cartLeft + radius, cartTop + cartHeightPx);
          ctx.quadraticCurveTo(cartLeft, cartTop + cartHeightPx, cartLeft, cartTop + cartHeightPx - radius);
          ctx.lineTo(cartLeft, cartTop + radius);
          ctx.quadraticCurveTo(cartLeft, cartTop, cartLeft + radius, cartTop);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          drawWheel(leftWheelX, leftWheelY);
          drawWheel(rightWheelX, rightWheelY);

          const [jointTopX, jointTopY] = worldToCanvas(drawX, data.body_bottom);
          const [pivotX, pivotY] = worldToCanvas(drawX, data.pivot_y);
          const [bobCanvasX, bobCanvasY] = worldToCanvas(bobX, bobY);
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 3.4;
          ctx.beginPath();
          ctx.moveTo(jointTopX, jointTopY);
          ctx.lineTo(pivotX, pivotY);
          ctx.stroke();
          ctx.lineWidth = 3.6;
          ctx.beginPath();
          ctx.moveTo(pivotX, pivotY);
          ctx.lineTo(bobCanvasX, bobCanvasY);
          ctx.stroke();

          ctx.fillStyle = "#0f172a";
          ctx.beginPath();
          ctx.arc(pivotX, pivotY, 6, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = "rgba(251,146,60,0.45)";
          ctx.beginPath();
          ctx.arc(bobCanvasX, bobCanvasY, 18, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#f97316";
          ctx.strokeStyle = "#9a3412";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(bobCanvasX, bobCanvasY, 13, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();

          if (Math.abs(force) > 1e-6) {{
            const arrowScale = Math.min(1.7, 0.34 + 0.17 * Math.abs(force));
            const [arrowStartX, arrowStartY] = worldToCanvas(drawX, data.arrow_y);
            const [arrowEndX] = worldToCanvas(drawX + Math.sign(force) * arrowScale, data.arrow_y);
            ctx.strokeStyle = "#7c3aed";
            ctx.fillStyle = "#7c3aed";
            ctx.lineWidth = 3;
            drawArrow(arrowStartX, arrowStartY, arrowEndX, arrowStartY);
          }}
        }});

        if (Math.abs(force) > 1e-6) {{
          const arrowScale = Math.min(1.7, 0.34 + 0.17 * Math.abs(force));
          const [arrowStartX, arrowStartY] = worldToCanvas(primaryX, data.arrow_y);
          const [arrowEndX] = worldToCanvas(primaryX + Math.sign(force) * arrowScale, data.arrow_y);
          ctx.fillStyle = "#5b21b6";
          ctx.font = "700 14px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(`F = ${{force.toFixed(2)}} N`, (arrowStartX + arrowEndX) / 2, arrowStartY - 12);
        }}

        timeLabel.textContent = `t = ${{data.time[index].toFixed(2)}} s`;
        stateLabel.textContent = `x = ${{x.toFixed(3)}} m | theta = ${{data.theta_deg[index].toFixed(2)}} deg | v = ${{data.velocity[index].toFixed(3)}} m/s | F = ${{force.toFixed(2)}} N`;
      }}

      function stopAnimation() {{
        if (timer !== null) {{
          clearInterval(timer);
          timer = null;
        }}
      }}

      function startAnimation() {{
        stopAnimation();
        timer = window.setInterval(() => {{
          frame = (frame + 1) % data.time.length;
          drawFrame(frame);
        }}, data.interval_ms);
      }}

      startBtn.addEventListener("click", startAnimation);
      stopBtn.addEventListener("click", stopAnimation);
      resetBtn.addEventListener("click", () => {{
        stopAnimation();
        frame = 0;
        drawFrame(frame);
      }});
      slider.addEventListener("input", (event) => {{
        stopAnimation();
        frame = Number(event.target.value);
        drawFrame(frame);
      }});
      window.addEventListener("resize", resizeCanvas);

      resizeCanvas();
      drawFrame(0);
    }})();
    </script>
    """


def _metric_card(label: str, value: str) -> str:
    return (
        "<div class='metric-pill'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        "</div>"
    )


def _insight_card(title: str, body: str) -> str:
    return (
        "<div class='insight-card'>"
        "<div class='insight-label'>Insight automatico</div>"
        f"<div class='insight-title'>{title}</div>"
        f"<div class='insight-copy'>{body}</div>"
        "</div>"
    )


def _simulation_insights(result, params: SimulationParams) -> list[tuple[str, str]]:
    peak_theta = float(np.max(np.abs(result.theta_deg)))
    travel = float(np.ptp(result.cart_position))
    final_theta = float(result.theta_deg[-1])
    final_omega = float(result.omega_deg[-1])
    reference_energy = max(1e-9, float(np.max(np.abs(result.total_energy))))
    relative_drift = abs(float(result.total_energy[-1] - result.total_energy[0])) / reference_energy * 100.0

    if peak_theta < 15.0:
        regime_text = "Il pendolo resta in un regime quasi lineare: le oscillazioni sono contenute e facili da interpretare."
    elif peak_theta < 60.0:
        regime_text = "La risposta e non lineare ma ancora leggibile: carrello e pendolo si scambiano energia in modo evidente."
    else:
        regime_text = "La simulazione entra in un regime ampio e decisamente non lineare, utile per osservare effetti fuori dal piccolo angolo."

    reference_force = constant_force_value(params.force_expression)
    ref_angle = reference_angle_for_constant_acceleration(params)
    if reference_force is None:
        forcing_text = "La forza cambia nel tempo: non esiste un unico equilibrio statico e contano molto frequenza, fase e smorzamento."
    elif abs(params.cart_damping) > 1e-12:
        forcing_text = (
            "Con smorzamento sul carrello, una forza costante non impone un angolo finale inclinato: "
            "dopo il transitorio il carrello tende a una velocita di regime e il pendolo verso theta = 0 deg."
        )
    elif ref_angle is None:
        forcing_text = "La forza e costante, ma la presenza di una coppia esterna rende meno utile un riferimento semplice sull'angolo."
    else:
        forcing_text = (
            f"Nel caso ideale di accelerazione costante, il riferimento inerziale del pendolo vale circa "
            f"{np.rad2deg(ref_angle):.2f} deg."
        )

    if relative_drift < 0.5:
        energy_text = f"La deriva energetica e molto contenuta ({relative_drift:.2f}%): l'integrazione numerica e coerente."
    elif relative_drift < 2.0:
        energy_text = f"La deriva energetica resta accettabile ({relative_drift:.2f}%), ma conviene controllare il passo temporale."
    else:
        energy_text = f"La deriva energetica arriva a {relative_drift:.2f}%: puoi ridurre dt per una simulazione piu fedele."

    if ref_angle is not None and params.pendulum_damping == 0.0:
        final_text = (
            f"Senza smorzamento del pendolo non c'e convergenza: la traiettoria resta attorno a theta_ref = "
            f"{np.rad2deg(ref_angle):.2f} deg invece di fermarsi su quell'angolo."
        )
    elif abs(final_theta) < 4.0 and abs(final_omega) < 8.0:
        final_text = "Lo stato finale e vicino all'equilibrio dinamico: il sistema sta rientrando o ci e gia quasi arrivato."
    else:
        final_text = (
            f"Alla fine della finestra simulata il pendolo e ancora attivo "
            f"(theta = {final_theta:.2f} deg, omega = {final_omega:.2f} deg/s)."
        )

    if travel < params.rod_length:
        cart_text = f"Il carrello copre un'escursione moderata di {travel:.2f} m, inferiore alla lunghezza dell'asta."
    elif travel < 4.0 * params.rod_length:
        cart_text = f"Il carrello compie uno spostamento importante di {travel:.2f} m, abbastanza da modificare molto il vincolo percepito dal pendolo."
    else:
        cart_text = f"Il carrello percorre {travel:.2f} m: e una manovra ampia, con forte trasferimento di energia lungo la guida."

    return [
        ("Regime dinamico", regime_text),
        ("Forzamento ed equilibrio", forcing_text),
        ("Energia numerica", energy_text),
        ("Stato finale", final_text),
        ("Escursione del carrello", cart_text),
    ]


def _simulation_csv(result) -> str:
    matrix = np.column_stack(
        [
            result.time,
            result.cart_position,
            result.cart_velocity,
            result.theta,
            result.theta_deg,
            result.omega,
            result.omega_deg,
            result.force,
            result.kinetic_energy,
            result.potential_energy,
            result.total_energy,
        ]
    )
    buffer = io.StringIO()
    np.savetxt(
        buffer,
        matrix,
        delimiter=",",
        header="time_s,cart_position_m,cart_velocity_m_s,theta_rad,theta_deg,omega_rad_s,omega_deg_s,force_N,kinetic_energy_J,potential_energy_J,total_energy_J",
        comments="",
    )
    return buffer.getvalue()


_ensure_default_state()

col_main, col_controls = st.columns([1.9, 1], gap="large")

with col_controls:
    st.markdown(
        """
        <div class="control-card">
            <div class="panel-title">Parametri del sistema</div>
            <div class="panel-copy">
                Scegli uno scenario gia pronto oppure modifica a mano masse, attriti e forzanti.
                La simulazione viene ricalcolata a ogni variazione.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.selectbox(
        "Scenario guidato",
        options=list(SCENARIO_PRESETS.keys()),
        key="selected_preset",
    )
    st.markdown(
        f"<div class='preset-note'>{SCENARIO_PRESETS[st.session_state['selected_preset']]['description']}</div>",
        unsafe_allow_html=True,
    )
    if st.button("Carica preset nello spazio di lavoro", use_container_width=True):
        _apply_preset(st.session_state["selected_preset"])
        st.rerun()

    st.number_input("Massa carrello [kg]", min_value=0.1, step=0.1, key="cart_mass")
    st.number_input("Attrito viscoso carrello [N s/m]", min_value=0.0, step=0.01, key="cart_damping")
    st.number_input(
        "Attrito viscoso pendolo [N m s/rad]",
        min_value=0.0,
        step=0.01,
        key="pendulum_damping",
    )
    st.number_input("Massa pendolo [kg]", min_value=0.05, step=0.05, key="pendulum_mass")
    st.number_input("Lunghezza trave rigida [m]", min_value=0.1, step=0.05, key="rod_length")
    st.text_input("Forza esterna F(t) [N]", key="force_expression")
    st.text_input("Coppia esterna tau(t) [N m]", key="torque_expression")
    st.caption("Sono disponibili `t`, `sin`, `cos`, `tan`, `exp`, `sqrt`, `pi`, `Heaviside`, `Abs`.")
    st.caption("Con forzanti intense il pendolo puo entrare in un regime non lineare molto ampio.")

    with st.expander("Impostazioni avanzate"):
        st.number_input("Gravita [m/s^2]", min_value=0.1, step=0.1, key="gravity")
        st.number_input("Posizione iniziale carrello x0 [m]", step=0.1, key="x0")
        st.number_input("Velocita iniziale carrello v0 [m/s]", step=0.1, key="v0")
        st.number_input("Angolo iniziale pendolo theta0 [deg]", step=1.0, key="theta0_deg")
        st.number_input("Velocita angolare iniziale [deg/s]", step=1.0, key="omega0_deg")
        st.number_input("Durata simulazione [s]", min_value=1.0, step=1.0, key="duration")
        st.number_input(
            "Passo integrazione dt [s]",
            min_value=0.001,
            step=0.001,
            format="%.3f",
            key="time_step",
        )

params = SimulationParams(
    cart_mass=float(st.session_state["cart_mass"]),
    pendulum_mass=float(st.session_state["pendulum_mass"]),
    rod_length=float(st.session_state["rod_length"]),
    gravity=float(st.session_state["gravity"]),
    cart_damping=float(st.session_state["cart_damping"]),
    pendulum_damping=float(st.session_state["pendulum_damping"]),
    x0=float(st.session_state["x0"]),
    v0=float(st.session_state["v0"]),
    theta0_rad=float(np.deg2rad(st.session_state["theta0_deg"])),
    omega0_rad=float(np.deg2rad(st.session_state["omega0_deg"])),
    force_expression=st.session_state["force_expression"],
    torque_expression=st.session_state["torque_expression"],
    duration=float(st.session_state["duration"]),
    time_step=float(st.session_state["time_step"]),
)

try:
    result = simulate_system(params)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Errore durante la simulazione: {exc}")
    st.stop()

peak_x = float(np.max(np.abs(result.cart_position)))
peak_theta = float(np.max(np.abs(result.theta_deg)))
peak_force = float(np.max(np.abs(result.force)))
peak_speed = float(np.max(np.abs(result.cart_velocity)))
energy_drift = float(result.total_energy[-1] - result.total_energy[0])
reference_energy = max(1e-9, float(np.max(np.abs(result.total_energy))))
energy_drift_pct = abs(energy_drift) / reference_energy * 100.0
reference_angle_rad = reference_angle_for_constant_acceleration(params)
reference_angle_deg = None if reference_angle_rad is None else float(np.rad2deg(reference_angle_rad))

slider_step = max(round(float(result.time[-1]) / 220.0, 3), params.time_step)
default_view_time = min(
    float(st.session_state.get("view_time", min(2.0, float(result.time[-1])))),
    float(result.time[-1]),
)

cart_width = DISPLAY_CART_WIDTH
cart_height = DISPLAY_CART_HEIGHT
insights = _simulation_insights(result, params)
csv_data = _simulation_csv(result)

with col_main:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Sistema Dinamico</div>
            <div class="hero-title">Carrello con pendolo sospeso</div>
            <div class="hero-copy">
                Esplora la risposta del sistema con scenari pronti, animazione continua, grafici energetici e ritratti di fase.
                Puoi partire da un preset e poi rifinire ogni parametro per studiare stabilita, trasferimento di energia e stati finali.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Modello del moto usato", expanded=False):
        st.caption("Nella notazione dell'immagine allegata: x2 = x_dot, x3 = theta, x4 = theta_dot.")
        st.latex(r"(M + m)\,\ddot{x} + m l \cos(\theta)\,\ddot{\theta} = F - C_m\,\dot{x} + m l \sin(\theta)\,\dot{\theta}^{2}")
        st.latex(r"m l \cos(\theta)\,\ddot{x} + m l^{2}\,\ddot{\theta} = \tau - m g l \sin(\theta) - C_p\,\dot{\theta}")
        st.caption(
            "La resa grafica usa una scala fissa: i valori inseriti dall'utente modificano il moto fisico, non la dimensione visiva di carrello, asta e scena."
        )

    st.html(
        _build_scene_animation_html(
            result,
            cart_width=cart_width,
            cart_height=cart_height,
            time_step=params.time_step,
        ),
        unsafe_allow_javascript=True,
    )

    st.markdown(
        "<div class='metric-row'>"
        + _metric_card("Spostamento massimo", f"{peak_x:.3f} m")
        + _metric_card("Angolo massimo", f"{peak_theta:.2f} deg")
        + _metric_card("Velocita max", f"{peak_speed:.3f} m/s")
        + _metric_card("Forza massima", f"{peak_force:.2f} N")
        + _metric_card("Theta rif. accel.", "--" if reference_angle_deg is None else f"{reference_angle_deg:.2f} deg")
        + _metric_card("Deriva energia", f"{energy_drift_pct:.2f}%")
        + "</div>",
        unsafe_allow_html=True,
    )

    tab_scene, tab_signals, tab_energy, tab_phase = st.tabs(
        ["Vista istantanea", "Andamenti", "Energia", "Traiettorie ed export"]
    )

    with tab_scene:
        view_time = st.slider(
            "Tempo di visualizzazione manuale [s]",
            min_value=0.0,
            max_value=float(result.time[-1]),
            value=float(default_view_time),
            step=float(slider_step),
            key="view_time",
        )
        frame_index = int(np.argmin(np.abs(result.time - view_time)))
        st.pyplot(
            _build_scene_figure(
                result,
                frame_index,
                cart_width=cart_width,
                cart_height=cart_height,
                reference_angle_rad=reference_angle_rad,
            ),
            width="stretch",
        )

    with tab_signals:
        st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
        st.subheader("Andamenti temporali principali")
        st.caption("Le coppie x/v e theta/omega usano scale verticali separate, cosi il confronto visivo resta fedele alle unita fisiche.")
        st.pyplot(_build_overview_figure(result, reference_angle_deg=reference_angle_deg), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_energy:
        st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
        st.subheader("Energia e lettura della simulazione")
        st.pyplot(_build_energy_figure(result), width="stretch")
        st.markdown(
            "<div class='insight-grid'>" + "".join(_insight_card(title, body) for title, body in insights) + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_phase:
        st.markdown("<div class='plot-card'>", unsafe_allow_html=True)
        st.subheader("Traiettorie nello spazio delle fasi")
        st.pyplot(_build_phase_figure(result, reference_angle_deg=reference_angle_deg), width="stretch")
        st.download_button(
            "Scarica risultati in CSV",
            data=csv_data,
            file_name="simulazione_carrello_pendolo.csv",
            mime="text/csv",
            use_container_width=False,
        )
        st.caption(f"Delta energia finale: {energy_drift:.4f} J. I punti blu e arancione indicano stato iniziale e finale.")
        st.markdown("</div>", unsafe_allow_html=True)
