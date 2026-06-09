"""
fase1_baseline.py
=================
FASE 1 — Baseline Secuencial.

Resuelve el modelo SEIR completo en Python puro sobre un solo núcleo. Su único
objetivo es validar la *corrección matemática* del solver RK4 ANTES de introducir
la complejidad de la red. Sirve de referencia (T_1, tiempo en 1 núcleo) para los
cálculos de Speedup y Eficiencia de las fases siguientes.

Uso:
    python fase1_baseline.py
"""

from __future__ import annotations

import time

import numpy as np

from seir_core import integrate_seir, peak_infection


def run_baseline(
    poblacion: int = 100_000,
    infectados_iniciales: int = 10,
    beta: float = 0.45,
    sigma: float = 0.25,
    gamma: float = 0.12,
    horizonte_dias: float = 180.0,
    paso_h: float = 0.1,
) -> dict:
    """Ejecuta una simulación SEIR secuencial y verifica la conservación de N.

    Parameters
    ----------
    poblacion : int
        Número total de hosts en la red (N).
    infectados_iniciales : int
        Hosts infectados en t=0 ("paciente cero" del gusano).
    beta, sigma, gamma : float
        Parámetros del modelo SEIR.
    horizonte_dias : float
        Duración simulada del brote.
    paso_h : float
        Tamaño del paso temporal del integrador RK4.

    Returns
    -------
    dict
        Métricas de la corrida: tiempo de cómputo, error de conservación de
        población y características del pico de infección.
    """
    steps = int(horizonte_dias / paso_h)
    y0 = np.array(
        [poblacion - infectados_iniciales, 0.0, infectados_iniciales, 0.0],
        dtype=float,
    )

    inicio = time.perf_counter()
    trayectoria = integrate_seir(y0, beta, sigma, gamma, steps, paso_h)
    duracion = time.perf_counter() - inicio

    # ---- Verificación de corrección: la población total debe conservarse ----
    # Como suma(dY/dt) = 0, cualquier desviación significativa delataría un error
    # en la formulación de las ecuaciones o en el integrador.
    poblacion_por_paso = trayectoria.sum(axis=1)
    error_max = float(np.max(np.abs(poblacion_por_paso - poblacion)))

    t_pico, max_infectados = peak_infection(trayectoria, paso_h)

    return {
        "tiempo_segundos": duracion,
        "error_conservacion": error_max,
        "dia_pico": t_pico,
        "max_infectados": max_infectados,
        "trayectoria": trayectoria,
    }


def main() -> None:
    """Punto de entrada: corre el baseline e imprime un reporte de validación."""
    resultado = run_baseline()

    print("=" * 60)
    print("FASE 1 — BASELINE SECUENCIAL (SEIR + RK4)")
    print("=" * 60)
    print(f"Tiempo de cómputo (T_1)   : {resultado['tiempo_segundos']:.4f} s")
    print(f"Error de conservación de N: {resultado['error_conservacion']:.3e}")
    print(f"Día del pico de infección : {resultado['dia_pico']:.1f}")
    print(f"Infectados en el pico     : {resultado['max_infectados']:.0f}")

    if resultado["error_conservacion"] < 1e-6:
        print("\n[OK] El integrador conserva la población: modelo validado.")
    else:
        print("\n[ADVERTENCIA] Deriva numérica detectada; revisar ecuaciones.")


if __name__ == "__main__":
    main()
