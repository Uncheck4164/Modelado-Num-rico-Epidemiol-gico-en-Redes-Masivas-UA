"""
benchmark.py
============
Análisis de escalabilidad: Speedup, Eficiencia Paralela y Ley de Amdahl.

Toma los tiempos de ejecución medidos (T_1 en la fase 1; T_n para cada conteo de
núcleos en las fases 2 y 3) y produce:
  * Speedup        S(n) = T_1 / T_n
  * Eficiencia     E(n) = S(n) / n
  * Ajuste de la fracción secuencial de la Ley de Amdahl.

Genera dos gráficas (Speedup y Eficiencia) que comparan el dominio acoplado MPI
contra las colas Monte Carlo, evidenciando por qué MPI se aplana (overhead de
red creciente) mientras que las colas escalan casi linealmente.

Uso:
    python benchmark.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: guarda PNG directamente
import matplotlib.pyplot as plt


def speedup(t1: float, tn: np.ndarray) -> np.ndarray:
    """Speedup S(n) = T_1 / T_n."""
    return t1 / tn


def eficiencia(s: np.ndarray, nucleos: np.ndarray) -> np.ndarray:
    """Eficiencia paralela E(n) = S(n) / n."""
    return s / nucleos


def fraccion_paralelizable_amdahl(s: np.ndarray, nucleos: np.ndarray) -> float:
    """Estima la fracción paralelizable p de la Ley de Amdahl por mínimos cuadrados.

    Ley de Amdahl:  S(n) = 1 / ((1 - p) + p / n)
    Reordenando:    1/S = (1 - p) + p*(1/n)  -> regresión lineal de 1/S contra 1/n.
    La PENDIENTE es la fracción paralelizable p; el INTERCEPTO es la fracción
    secuencial (1 - p), que es la que impone el techo de escalabilidad: ninguna
    cantidad de núcleos acelera la parte que debe correr en serie.

    Returns
    -------
    float
        Fracción paralelizable p estimada (1.0 = perfectamente paralelo).
    """
    x = 1.0 / nucleos
    y = 1.0 / s
    pendiente, intercepto = np.polyfit(x, y, 1)
    p = float(pendiente)
    return max(0.0, min(1.0, p))


def graficar(nucleos, datos: dict, salida: str = "../resultados/escalabilidad.png") -> None:
    """Dibuja Speedup y Eficiencia para cada paradigma en una figura de 2 paneles.

    Parameters
    ----------
    nucleos : np.ndarray
        Número de núcleos probados.
    datos : dict
        {nombre_paradigma: tiempos_array}. Cada array es T_n alineado con
        `nucleos`; el primer elemento (n=1) es la referencia T_1.
    salida : str
        Ruta del PNG de salida.
    """
    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(13, 5))
    colores = {"MPI acoplado": "#F87171", "Colas Monte Carlo": "#2DD4BF"}

    # Referencia ideal (Speedup lineal y Eficiencia constante = 1).
    ax_s.plot(nucleos, nucleos, "--", color="#94A3B8", label="Ideal (lineal)")
    ax_e.axhline(1.0, ls="--", color="#94A3B8", label="Ideal (100%)")

    for nombre, tiempos in datos.items():
        s = speedup(tiempos[0], tiempos)
        e = eficiencia(s, nucleos)
        p = fraccion_paralelizable_amdahl(s, nucleos)
        col = colores.get(nombre, "#6366F1")
        ax_s.plot(nucleos, s, "o-", color=col, lw=2,
                  label=f"{nombre}  (paralelizable≈{p:.1%})")
        ax_e.plot(nucleos, e * 100, "o-", color=col, lw=2, label=nombre)

    ax_s.set_title("Speedup S(n) = T₁ / Tₙ", fontweight="bold")
    ax_s.set_xlabel("Número de núcleos (n)")
    ax_s.set_ylabel("Speedup")
    ax_s.legend()
    ax_s.grid(alpha=0.3)

    ax_e.set_title("Eficiencia Paralela E(n) = S(n) / n", fontweight="bold")
    ax_e.set_xlabel("Número de núcleos (n)")
    ax_e.set_ylabel("Eficiencia (%)")
    ax_e.legend()
    ax_e.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(salida, dpi=130)
    print(f"[OK] Gráfica guardada en {salida}")


def main() -> None:
    """Demuestra el análisis con tiempos de ejemplo.

    Reemplazar los arrays `t_*` por los tiempos reales medidos en el clúster.
    """
    nucleos = np.array([1, 2, 4, 8, 16, 32])

    # Tiempos de EJEMPLO (segundos). Sustituir por mediciones reales:
    #   - MPI: el overhead de halo crece con n -> el tiempo baja cada vez menos.
    #   - Colas: tareas independientes -> el tiempo cae casi proporcionalmente.
    t_mpi = np.array([100.0, 55.9, 34.0, 23.0, 17.5, 14.7])
    t_colas = np.array([100.0, 50.3, 25.4, 12.9, 6.7, 3.6])

    datos = {"MPI acoplado": t_mpi, "Colas Monte Carlo": t_colas}

    print("=" * 60)
    print("ANÁLISIS DE ESCALABILIDAD")
    print("=" * 60)
    for nombre, t in datos.items():
        s = speedup(t[0], t)
        e = eficiencia(s, nucleos)
        p = fraccion_paralelizable_amdahl(s, nucleos)
        print(f"\n{nombre}:")
        print(f"  Speedup @32 núcleos     : {s[-1]:.1f}x")
        print(f"  Eficiencia @32 núcleos  : {e[-1]:.1%}")
        print(f"  Fracción paralelizable p: {p:.2%}")
        print(f"  Fracción secuencial 1-p : {1 - p:.2%}")
        techo = 1.0 / (1.0 - p) if p < 1.0 else float("inf")
        print(f"  Techo de Speedup (n->∞) : {techo:.1f}x")

    graficar(nucleos, datos)


if __name__ == "__main__":
    main()
