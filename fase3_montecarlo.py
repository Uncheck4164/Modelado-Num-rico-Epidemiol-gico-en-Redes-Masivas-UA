"""
fase3_montecarlo.py
===================
FASE 3 — Fuzzing y Monte Carlo vía Colas de Trabajo.

En lugar de una sola simulación acoplada, aquí se evalúan 50.000 "mutaciones"
independientes del malware: cada una varía aleatoriamente sus parámetros
(beta, sigma, gamma) dentro de rangos plausibles. Como cada simulación es
totalmente independiente del resto (problema "embarrassingly parallel"), el
trabajo se reparte mediante una COLA: el planificador (scheduler) de ipyparallel
inyecta tareas a los trabajadores (engines) de forma asíncrona.

Ventajas frente a MPI para este problema:
  * No hay comunicación entre tareas -> casi sin overhead -> Speedup ~lineal.
  * Tolerancia a fallos: si un engine cae, el scheduler reasigna su tarea.

Ejecución:
    # 1) Levantar el clúster local de trabajadores (ej. 8 engines):
    ipcluster start -n 8 &
    # 2) Lanzar el fuzzing:
    python fase3_montecarlo.py
"""

from __future__ import annotations

import time

import numpy as np


def simular_mutacion(semilla: int) -> dict:
    """Ejecuta UNA simulación SEIR con parámetros mutados aleatoriamente.

    Esta función se serializa y se envía a un engine remoto, por lo que importa
    sus dependencias internamente (cada engine es un intérprete Python aislado
    que no comparte el espacio de nombres del proceso cliente).

    Parameters
    ----------
    semilla : int
        Semilla del generador aleatorio; garantiza que cada tarea de la cola
        produzca una mutación distinta y reproducible.

    Returns
    -------
    dict
        Parámetros muestreados y métricas del brote resultante.
    """
    import numpy as np
    from seir_core import integrate_seir, peak_infection

    rng = np.random.default_rng(semilla)

    # Fuzzing: inyección de variaciones aleatorias en los parámetros iniciales.
    beta = rng.uniform(0.15, 0.80)    # agresividad del contagio
    sigma = rng.uniform(0.10, 0.40)   # rapidez de la latencia
    gamma = rng.uniform(0.05, 0.25)   # velocidad de parcheo

    poblacion = 100_000
    y0 = np.array([poblacion - 10, 0.0, 10.0, 0.0], dtype=float)
    trayectoria = integrate_seir(y0, beta, sigma, gamma, steps=1800, h=0.1)
    dia_pico, max_infectados = peak_infection(trayectoria, h=0.1)

    return {
        "beta": beta, "sigma": sigma, "gamma": gamma,
        "dia_pico": dia_pico, "max_infectados": max_infectados,
        # R0 efectivo aproximado: clave para clasificar la severidad del brote.
        "r0_aprox": beta / gamma,
    }


def ejecutar_fuzzing(n_mutaciones: int = 50_000) -> list[dict]:
    """Reparte las mutaciones en la cola de ipyparallel y recolecta resultados.

    Parameters
    ----------
    n_mutaciones : int
        Número de escenarios aleatorios a simular.

    Returns
    -------
    list[dict]
        Una entrada por mutación con sus parámetros y métricas.
    """
    import ipyparallel as ipp

    # El cliente se conecta al clúster levantado con `ipcluster start`.
    cliente = ipp.Client()
    vista = cliente.load_balanced_view()   # balanceo de carga sobre la cola

    # Aseguramos que cada engine pueda importar el núcleo numérico.
    cliente[:].execute("import sys; sys.path.insert(0, '.')")

    semillas = list(range(n_mutaciones))

    inicio = time.perf_counter()
    # map_async encola las tareas y devuelve de inmediato (no bloquea). El
    # scheduler las inyecta a los engines a medida que quedan libres, saturando
    # los trabajadores al 100% sin que el cliente gestione la asignación.
    futuro = vista.map_async(simular_mutacion, semillas)
    resultados = futuro.get()              # bloquea solo aquí, al recolectar
    duracion = time.perf_counter() - inicio

    print("=" * 60)
    print(f"FASE 3 — MONTE CARLO / FUZZING  ({len(cliente.ids)} engines)")
    print("=" * 60)
    print(f"Mutaciones simuladas      : {n_mutaciones:,}")
    print(f"Tiempo total              : {duracion:.2f} s")
    print(f"Throughput                : {n_mutaciones / duracion:,.0f} sim/s")

    picos = np.array([r["max_infectados"] for r in resultados])
    print(f"Pico de infección medio   : {picos.mean():,.0f} hosts")
    print(f"Peor escenario (p99)      : {np.percentile(picos, 99):,.0f} hosts")
    return resultados


if __name__ == "__main__":
    ejecutar_fuzzing()
