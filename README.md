# Modelado Numérico Epidemiológico en Redes Masivas

Simulación de la cinética de infección de un gusano informático mediante el
modelo **SEIR** resuelto con **Runge-Kutta de cuarto orden (RK4)**, contrastando
dos paradigmas de cómputo de alto rendimiento (HPC):

- **MPI (mpi4py)** — memoria distribuida, dominio acoplado por VLAN.
- **Colas de trabajo (ipyparallel)** — Monte Carlo / fuzzing asíncrono.

## Estructura

```
proyecto_seir/
├── src/
│   ├── seir_core.py        # Núcleo matemático: SEIR + RK4 (funciones puras)
│   ├── fase1_baseline.py   # Fase 1: baseline secuencial + validación
│   ├── fase2_mpi.py        # Fase 2: dominio acoplado MPI (zonas de halo)
│   ├── fase3_montecarlo.py # Fase 3: fuzzing Monte Carlo vía colas
│   └── benchmark.py        # Speedup, Eficiencia y Ley de Amdahl
└── resultados/
    └── escalabilidad.png   # Curvas de Speedup y Eficiencia
```

## Requisitos

```bash
pip install numpy matplotlib mpi4py ipyparallel
# mpi4py requiere una implementación MPI del sistema (OpenMPI o MPICH).
```

## Ejecución

**Fase 1 — Baseline secuencial** (valida la matemática y entrega T₁):

```bash
cd src
python fase1_baseline.py
```

**Fase 2 — MPI dominio acoplado** (n = número de VLAN/núcleos):

```bash
cd src
mpirun -n 4 python fase2_mpi.py
```

**Fase 3 — Monte Carlo vía colas** (50.000 mutaciones):

```bash
cd src
ipcluster start -n 8 &     # levanta 8 engines (trabajadores)
python fase3_montecarlo.py
```

**Análisis de escalabilidad** (sustituir los tiempos de ejemplo por los medidos):

```bash
cd src
python benchmark.py
```

## Modelo

```
dS/dt = -β·S·I/N
dE/dt =  β·S·I/N − σ·E
dI/dt =  σ·E − γ·I
dP/dt =  γ·I
```

Invariante: la población total `N = S + E + I + P` se conserva en cada paso
(la fase 1 lo verifica explícitamente como prueba de corrección).

## Notas de diseño

- El integrador RK4 está **desacoplado** del modelo: recibe la función de
  derivadas como argumento. Esto permite que la fase MPI inyecte una versión
  que dispara la comunicación de red (Sendrecv) sin tocar el integrador.
- La sincronización de fronteras (**zonas de halo**) usa `Sendrecv` para
  **evitar interbloqueos (deadlock)**.
- La fase 3 usa `map_async` sobre una vista con balanceo de carga: el scheduler
  satura los engines y reasigna tareas ante caídas (tolerancia a fallos).
