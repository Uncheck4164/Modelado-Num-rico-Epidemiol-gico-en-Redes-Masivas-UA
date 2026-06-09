"""
seir_core.py
============
Núcleo numérico del proyecto: modelo epidemiológico SEIR e integrador
Runge-Kutta de cuarto orden (RK4).

Este módulo NO sabe nada de paralelismo. Contiene únicamente *funciones puras*
(sin estado global, sin efectos secundarios) que son reutilizadas idénticamente
por la fase secuencial (baseline), la fase MPI (dominio acoplado) y la fase
Monte Carlo (colas de trabajo). Mantener el matemática aislada de la
infraestructura de cómputo es lo que permite validar la corrección numérica una
sola vez y confiar en ella en los tres paradigmas.

Modelo SEIR aplicado a propagación de malware en una red corporativa de N hosts:

    dS/dt = -beta * S * I / N
    dE/dt =  beta * S * I / N - sigma * E
    dI/dt =  sigma * E        - gamma * I
    dP/dt =  gamma * I

donde:
    S = Susceptibles      (hosts sanos y vulnerables)
    E = Expuestos         (infectados en periodo de latencia, aún no contagian)
    I = Infectados        (activos, propagan el gusano)
    P = Parcheados/Inmunes (host remediados, ya no participan)

    beta  = tasa de propagación del malware
    sigma = tasa de latencia (E -> I)
    gamma = velocidad de despliegue de parches (I -> P)

Invariante físico: la suma de las cuatro derivadas es 0, por lo que la población
total N = S + E + I + P debe conservarse en cada iteración. La fase 1 verifica
explícitamente esta conservación como prueba de corrección del integrador.
"""

from __future__ import annotations

import numpy as np


def seir_derivatives(
    state: np.ndarray,
    beta: float,
    sigma: float,
    gamma: float,
    coupling_infected: float = 0.0,
    beta_cross: float = 0.0,
) -> np.ndarray:
    """Calcula el vector de derivadas dY/dt = f(t, Y) del sistema SEIR.

    Es una función pura: dado el mismo estado y los mismos parámetros, siempre
    devuelve el mismo resultado. No depende explícitamente de `t` porque el
    sistema es autónomo (los parámetros no varían en el tiempo).

    Parameters
    ----------
    state : np.ndarray, shape (4,)
        Vector de estado [S, E, I, P] de la VLAN local.
    beta : float
        Tasa de propagación intra-VLAN del malware.
    sigma : float
        Tasa de latencia de la vulnerabilidad (transición E -> I).
    gamma : float
        Velocidad de distribución de parches (transición I -> P).
    coupling_infected : float, optional
        Suma de infectados (I) de las VLAN vecinas. En la fase MPI este valor
        llega por la red mediante Sendrecv; en la fase secuencial vale 0.
    beta_cross : float, optional
        Tasa de propagación inter-VLAN (contagio que cruza fronteras de red).
        Si es 0 las VLAN están desacopladas.

    Returns
    -------
    np.ndarray, shape (4,)
        Vector de derivadas [dS/dt, dE/dt, dI/dt, dP/dt].
    """
    S, E, I, P = state
    N = S + E + I + P
    if N <= 0:
        # VLAN vacía: el sistema permanece estático, evita división por cero.
        return np.zeros(4, dtype=float)

    # Fuerza de infección: componente interna + componente que llega desde las
    # VLAN vecinas (término de acoplamiento del dominio distribuido).
    infection_force = beta * (S * I / N) + beta_cross * (S * coupling_infected / N)

    dS = -infection_force
    dE = infection_force - sigma * E
    dI = sigma * E - gamma * I
    dP = gamma * I
    return np.array([dS, dE, dI, dP], dtype=float)


def rk4_step(
    state: np.ndarray,
    h: float,
    derivative_fn,
) -> np.ndarray:
    """Avanza un único paso temporal con Runge-Kutta de cuarto orden.

    Calcula el promedio ponderado de cuatro pendientes intermedias (k1..k4)
    para pasar de Y_n a Y_{n+1} = Y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4).

    El integrador está completamente desacoplado del modelo: recibe
    `derivative_fn`, un *callable* que evalúa f(estado). Esto permite que la
    fase MPI inyecte una función que, internamente, dispara la comunicación de
    red para refrescar la zona de halo antes de evaluar cada pendiente, sin
    modificar una sola línea de este integrador.

    Parameters
    ----------
    state : np.ndarray, shape (4,)
        Estado actual Y_n = [S, E, I, P].
    h : float
        Tamaño del paso temporal.
    derivative_fn : callable
        Función f(estado) -> np.ndarray que devuelve dY/dt en ese estado.

    Returns
    -------
    np.ndarray, shape (4,)
        Estado avanzado Y_{n+1}.
    """
    k1 = derivative_fn(state)
    k2 = derivative_fn(state + 0.5 * h * k1)
    k3 = derivative_fn(state + 0.5 * h * k2)
    k4 = derivative_fn(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_seir(
    y0: np.ndarray,
    beta: float,
    sigma: float,
    gamma: float,
    steps: int,
    h: float,
) -> np.ndarray:
    """Integra el modelo SEIR aislado (sin acoplamiento) sobre un horizonte.

    Es el solver de referencia usado por la fase 1 (baseline) y por cada
    simulación Monte Carlo individual de la fase 3.

    Parameters
    ----------
    y0 : np.ndarray, shape (4,)
        Condición inicial [S0, E0, I0, P0].
    beta, sigma, gamma : float
        Parámetros del modelo SEIR.
    steps : int
        Número de pasos temporales a integrar.
    h : float
        Tamaño del paso temporal.

    Returns
    -------
    np.ndarray, shape (steps + 1, 4)
        Trayectoria completa; la fila i es el estado en el instante i*h.
    """
    trajectory = np.empty((steps + 1, 4), dtype=float)
    trajectory[0] = y0

    # La derivada se evalúa siempre sobre el mismo modelo aislado; encapsularla
    # en una closure mantiene `rk4_step` agnóstico del modelo concreto.
    def f(state):
        return seir_derivatives(state, beta, sigma, gamma)

    state = y0.astype(float)
    for i in range(steps):
        state = rk4_step(state, h, f)
        trajectory[i + 1] = state
    return trajectory


def peak_infection(trajectory: np.ndarray, h: float) -> tuple[float, float]:
    """Devuelve el pico de infección de una trayectoria SEIR.

    Métrica resumen usada por el análisis Monte Carlo: de cada escenario solo
    nos interesa cuán grave fue el brote y cuándo ocurrió.

    Parameters
    ----------
    trajectory : np.ndarray, shape (steps + 1, 4)
        Trayectoria devuelta por `integrate_seir`.
    h : float
        Tamaño del paso temporal, para convertir el índice en tiempo.

    Returns
    -------
    tuple[float, float]
        (instante del pico, número máximo de infectados simultáneos).
    """
    infected = trajectory[:, 2]
    idx = int(np.argmax(infected))
    return idx * h, float(infected[idx])
