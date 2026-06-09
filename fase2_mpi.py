"""
fase2_mpi.py
============
FASE 2 — Redes Acopladas y Overhead vía MPI.

La red global se subdivide en M VLAN contiguas; cada núcleo (rank) integra el
RK4 de UNA sola VLAN. Como el malware cruza fronteras entre VLAN, el cálculo de
las pendientes k2, k3 y k4 de cada núcleo necesita el número de infectados I de
sus VLAN vecinas. Ese valor de frontera (la "zona de halo") se intercambia por
la red con comm.Sendrecv() en cada subpaso del RK4.

Arquitectura SPMD (Single Program, Multiple Data): todos los procesos ejecutan
ESTE mismo archivo, pero cada uno opera sobre la VLAN correspondiente a su rank.

Ejecución (M = número de VLAN/núcleos):
    mpirun -n 4 python fase2_mpi.py

Requisitos: mpi4py y una implementación MPI (OpenMPI / MPICH).
"""

from __future__ import annotations

import time

import numpy as np
from mpi4py import MPI

from seir_core import rk4_step, seir_derivatives


def intercambiar_halo(comm, valor_local_I: float, izq: int, der: int) -> tuple[float, float]:
    """Intercambia el número de infectados (I) con las VLAN vecinas.

    Esta es la BARRERA DE SINCRONIZACIÓN crítica del programa. Usamos Sendrecv
    (envío y recepción simultáneos en una sola llamada) en lugar de un Send
    seguido de un Recv por una razón concreta de control de deadlocks: si todos
    los nodos hicieran Send primero, todos quedarían bloqueados esperando que
    alguien reciba y nadie recibiría -> interbloqueo (deadlock) y el clúster
    entero se congela. Sendrecv delega al runtime MPI el orden de las
    operaciones, eliminando ese riesgo.

    Parameters
    ----------
    comm : MPI.Comm
        Comunicador MPI.
    valor_local_I : float
        Infectados de la VLAN local que se enviarán a ambos vecinos.
    izq, der : int
        Ranks de los vecinos izquierdo y derecho (MPI.PROC_NULL en los bordes).

    Returns
    -------
    tuple[float, float]
        (I del vecino izquierdo, I del vecino derecho). En los bordes del
        dominio el vecino inexistente aporta 0.0.
    """
    send = np.array([valor_local_I], dtype="d")

    # Recibir del vecino izquierdo mientras enviamos al derecho.
    recv_izq = np.zeros(1, dtype="d")
    comm.Sendrecv(send, dest=der, sendtag=0, recvbuf=recv_izq, source=izq, recvtag=0)

    # Recibir del vecino derecho mientras enviamos al izquierdo.
    recv_der = np.zeros(1, dtype="d")
    comm.Sendrecv(send, dest=izq, sendtag=1, recvbuf=recv_der, source=der, recvtag=1)

    return float(recv_izq[0]), float(recv_der[0])


def paso_rk4_acoplado(comm, estado, h, params, izq, der):
    """Avanza un paso RK4 refrescando la zona de halo en cada subpendiente.

    Para ser fiel al método, la fuerza de infección inter-VLAN de cada pendiente
    (k1..k4) debe evaluarse con el I del vecino EN ESE subestado. Por eso la
    comunicación de red ocurre cuatro veces por paso temporal. Este patrón es
    deliberado: maximiza la fidelidad numérica a costa de overhead de red, y es
    justamente lo que hace que la curva de Speedup de MPI se aplane (Ley de
    Amdahl) cuando crece el número de nodos.

    Parameters
    ----------
    comm : MPI.Comm
        Comunicador MPI.
    estado : np.ndarray, shape (4,)
        Estado local [S, E, I, P] de la VLAN.
    h : float
        Paso temporal.
    params : tuple
        (beta, sigma, gamma, beta_cross).
    izq, der : int
        Ranks vecinos.

    Returns
    -------
    np.ndarray, shape (4,)
        Estado local avanzado.
    """
    beta, sigma, gamma, beta_cross = params

    def derivada_con_halo(s):
        # 's' es el subestado del RK4 (Y_n, Y_n + h/2 k1, ...). Antes de evaluar
        # la pendiente refrescamos el halo con el I de los vecinos en ese mismo
        # subestado, de modo que el término de acoplamiento sea consistente.
        i_izq, i_der = intercambiar_halo(comm, s[2], izq, der)
        return seir_derivatives(
            s, beta, sigma, gamma,
            coupling_infected=i_izq + i_der,
            beta_cross=beta_cross,
        )

    return rk4_step(estado, h, derivada_con_halo)


def main() -> None:
    """Integra el dominio acoplado en paralelo y consolida con Gather."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()      # identidad de este proceso
    size = comm.Get_size()      # número total de VLAN/núcleos

    # --- Parámetros globales (idénticos en todos los procesos: SPMD) ---
    poblacion_por_vlan = 100_000
    beta, sigma, gamma, beta_cross = 0.45, 0.25, 0.12, 0.05
    horizonte_dias, paso_h = 180.0, 0.1
    steps = int(horizonte_dias / paso_h)

    # Topología en anillo abierto: el rank 0 no tiene vecino izquierdo y el
    # último no tiene vecino derecho. MPI.PROC_NULL hace que Sendrecv ignore
    # esas direcciones sin bloquear.
    izq = rank - 1 if rank > 0 else MPI.PROC_NULL
    der = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    # Condición inicial: solo la VLAN 0 arranca infectada (foco del gusano).
    if rank == 0:
        estado = np.array([poblacion_por_vlan - 10, 0.0, 10.0, 0.0], dtype=float)
    else:
        estado = np.array([poblacion_por_vlan, 0.0, 0.0, 0.0], dtype=float)

    params = (beta, sigma, gamma, beta_cross)

    comm.Barrier()                       # alinear el cronómetro entre procesos
    inicio = MPI.Wtime()

    pico_local = estado[2]
    for _ in range(steps):
        estado = paso_rk4_acoplado(comm, estado, paso_h, params, izq, der)
        pico_local = max(pico_local, estado[2])

    comm.Barrier()
    duracion = MPI.Wtime() - inicio

    # --- Consolidación de resultados hacia el nodo maestro (rank 0) ---
    # Gather recolecta el estado final de cada VLAN; Reduce agrega el pico
    # global. Son operaciones colectivas: todos los procesos deben invocarlas.
    estados_finales = comm.gather(estado, root=0)
    pico_global = comm.reduce(pico_local, op=MPI.MAX, root=0)

    if rank == 0:
        print("=" * 60)
        print(f"FASE 2 — MPI DOMINIO ACOPLADO  ({size} VLAN / núcleos)")
        print("=" * 60)
        print(f"Tiempo de cómputo (T_{size}) : {duracion:.4f} s")
        print(f"Pico de infección global    : {pico_global:.0f} hosts")
        infectados_residuales = sum(e[2] for e in estados_finales)
        print(f"Infectados residuales (t=fin): {infectados_residuales:.1f}")
        print("\nGuardar este tiempo para la curva de Speedup (ver benchmark).")


if __name__ == "__main__":
    main()
