from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass
class SimulationParams:
    cart_mass: float
    pendulum_mass: float
    rod_length: float
    gravity: float
    cart_damping: float
    pendulum_damping: float
    x0: float
    v0: float
    theta0_rad: float
    omega0_rad: float
    force_expression: str
    torque_expression: str
    duration: float
    time_step: float


@dataclass
class SimulationResult:
    time: np.ndarray
    state: np.ndarray
    force: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    length: float

    @property
    def cart_position(self) -> np.ndarray:
        return self.state[:, 0]

    @property
    def cart_velocity(self) -> np.ndarray:
        return self.state[:, 1]

    @property
    def theta(self) -> np.ndarray:
        return self.state[:, 2]

    @property
    def omega(self) -> np.ndarray:
        return self.state[:, 3]

    @property
    def theta_deg(self) -> np.ndarray:
        return np.rad2deg(self.theta)

    @property
    def omega_deg(self) -> np.ndarray:
        return np.rad2deg(self.omega)


def constant_force_value(expression: str) -> float | None:
    expr = (expression or "").strip() or "0"
    t = sp.symbols("t", real=True)
    local_dict = {
        "t": t,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "Heaviside": sp.Heaviside,
        "Abs": sp.Abs,
    }

    try:
        parsed = sp.sympify(expr, locals=local_dict)
    except Exception:
        return None

    if parsed.free_symbols:
        return None

    try:
        return float(parsed.evalf())
    except Exception:
        return None


def equilibrium_angle_for_constant_force(params: SimulationParams) -> float | None:
    force_value = constant_force_value(params.force_expression)
    if force_value is None:
        return None

    denominator = (params.cart_mass + params.pendulum_mass) * params.gravity
    if denominator <= 0.0:
        return None

    return float(-np.arctan(force_value / denominator))


def _validate_params(params: SimulationParams) -> None:
    if params.cart_mass <= 0.0:
        raise ValueError("La massa del carrello deve essere positiva.")
    if params.pendulum_mass <= 0.0:
        raise ValueError("La massa del pendolo deve essere positiva.")
    if params.rod_length <= 0.0:
        raise ValueError("La lunghezza dell'asta deve essere positiva.")
    if params.gravity <= 0.0:
        raise ValueError("La gravita deve essere positiva.")
    if params.duration <= 0.0:
        raise ValueError("La durata della simulazione deve essere positiva.")
    if params.time_step <= 0.0:
        raise ValueError("Il passo di integrazione deve essere positivo.")
    if params.time_step >= params.duration:
        raise ValueError("Il passo di integrazione deve essere piu piccolo della durata totale.")


def _build_force_function(expression: str):
    expr = expression.strip() or "0"
    t = sp.symbols("t", real=True)
    local_dict = {
        "t": t,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "Heaviside": sp.Heaviside,
        "Abs": sp.Abs,
    }

    try:
        parsed = sp.sympify(expr, locals=local_dict)
        raw_fn = sp.lambdify(t, parsed, modules=["numpy"])
    except Exception as exc:
        raise ValueError(f"Espressione della forza non valida: {exc}") from exc

    def force_fn(time_value):
        values = np.asarray(raw_fn(time_value), dtype=float)
        return values

    try:
        sample = force_fn(np.array([0.0, 0.5, 1.0]))
        if sample.ndim == 0:
            float(sample)
    except Exception as exc:
        raise ValueError(f"Non riesco a valutare F(t): {exc}") from exc

    return force_fn


def _build_torque_function(expression: str):
    return _build_force_function(expression)


def _dynamics(time_value: float, state: np.ndarray, params: SimulationParams, force_fn, torque_fn) -> np.ndarray:
    x, v, theta, omega = state
    m_cart = params.cart_mass
    m_pend = params.pendulum_mass
    length = params.rod_length
    gravity = params.gravity
    cart_damping = params.cart_damping
    pendulum_damping = params.pendulum_damping

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    applied_force = float(np.asarray(force_fn(time_value)).reshape(-1)[0])
    applied_torque = float(np.asarray(torque_fn(time_value)).reshape(-1)[0])

    a11 = m_cart + m_pend
    a12 = m_pend * length * cos_theta
    a21 = cos_theta
    a22 = length
    b1 = applied_force - cart_damping * v + m_pend * length * sin_theta * omega**2
    b2 = (
        applied_torque / (m_pend * length)
        - gravity * sin_theta
        - (pendulum_damping / (m_pend * length)) * omega
    )
    determinant = a11 * a22 - a12 * a21

    x_ddot = (b1 * a22 - a12 * b2) / determinant
    theta_ddot = (a11 * b2 - a21 * b1) / determinant

    return np.array([v, x_ddot, omega, theta_ddot], dtype=float)


def _rk4_step(time_value: float, state: np.ndarray, step: float, params: SimulationParams, force_fn, torque_fn) -> np.ndarray:
    k1 = _dynamics(time_value, state, params, force_fn, torque_fn)
    k2 = _dynamics(time_value + 0.5 * step, state + 0.5 * step * k1, params, force_fn, torque_fn)
    k3 = _dynamics(time_value + 0.5 * step, state + 0.5 * step * k2, params, force_fn, torque_fn)
    k4 = _dynamics(time_value + step, state + step * k3, params, force_fn, torque_fn)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _compute_energy(state: np.ndarray, params: SimulationParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = state[:, 1]
    theta = state[:, 2]
    omega = state[:, 3]
    m_cart = params.cart_mass
    m_pend = params.pendulum_mass
    length = params.rod_length
    gravity = params.gravity

    kinetic = (
        0.5 * (m_cart + m_pend) * v**2
        + m_pend * length * np.cos(theta) * v * omega
        + 0.5 * m_pend * (length**2) * omega**2
    )
    potential = m_pend * gravity * length * (1.0 - np.cos(theta))
    total = kinetic + potential
    return kinetic, potential, total


def simulate_system(params: SimulationParams) -> SimulationResult:
    _validate_params(params)
    force_fn = _build_force_function(params.force_expression)
    torque_fn = _build_torque_function(params.torque_expression)

    steps = int(np.ceil(params.duration / params.time_step))
    time = np.linspace(0.0, params.duration, steps + 1)
    state = np.empty((steps + 1, 4), dtype=float)
    state[0] = np.array([params.x0, params.v0, params.theta0_rad, params.omega0_rad], dtype=float)

    for index in range(steps):
        current_time = time[index]
        step = time[index + 1] - time[index]
        state[index + 1] = _rk4_step(current_time, state[index], step, params, force_fn, torque_fn)

    force = np.asarray(force_fn(time), dtype=float)
    if force.ndim == 0:
        force = np.full_like(time, float(force))

    kinetic_energy, potential_energy, total_energy = _compute_energy(state, params)

    return SimulationResult(
        time=time,
        state=state,
        force=force,
        kinetic_energy=kinetic_energy,
        potential_energy=potential_energy,
        total_energy=total_energy,
        length=params.rod_length,
    )
