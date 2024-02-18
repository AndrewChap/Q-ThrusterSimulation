# Q-ThrusterSimulation
This simulation is similar to a particle-in-cell simulation and is parallelized for the GPU by cell (referred to in the code as "branches").
Particles are moved in the ParticleMoverGPU function.  
Particles are transferred efficiently between cells through the functions "FillGapsGPU" and "TransferParticlesGPU". 
GPU memory is managed so that memory coalescense is maximized, resulting in the minimum required calls to global GPU memory.
See [QThrusterMemoryOrganization.md](QThrusterMemoryOrganization.md) or QThrusterMemoryOrganization.pptx for more information (contains ppt animations.)

# Dependencies
This has been tested on Linux (Debian 11 with an Nvidia).
- CUDA 7.5 or greater
- CMake
- Julia (for plotting instead of Matlab)
- (optional) ninja (`apt install ninja-buildtool`) or make

# Install
- Clone this repo and cd into it.
- `mkdir build; cd build`
- make:
  - `cmake ..`
  - `make`
- ninja:
  - `cmake -G Ninja ..`
  - `ninja -v`

# Run
- Run Simulation: `./particle_branch_11`

## Run Output

- `ForceCurveGPU%{date}.txt` see Visualize Force Results.
- `GPU%{date}/` see Visualize VTK
  - `density%i.vtk`
  - `electron%i.vtk`
  - `positron%i.vtk`

## Analysis
- Visualize Force Results: `julia ForceCurveReader.jl`
  - May need to copy in the ForceCurveGPU output file and do manual error corrections yourself.
- Visualize VTK Sim:
  - ParaView https://www.paraview.org/download/
  - [cone_closed.vtk](src/cone_closed.vtk) and [cone_open.vtk](src/cone_open.vtk) are the resonator geometries
  - Other VTK files for the electron and positron densities will be generated during run.
- Benchmark with Nsight Compute (GUI).

## Compilation Modes
- CPUrun: gpu style code that runs on the CPU. 
  - performance:
- GPUrun: normal way to run. (true). 
  - performance:
- CPUorig: does not compile currently. Unoptomized CPU code during prototyping.
  - performance:
- AddPartGPU: spawn particles on CPU+move or spawn on GPU. currently CPU+move is faster due to low particles.
  (Need to have GPUrun enabled for this to work.)
  - performance:

## Code Outline
-- QThrusterMain.cu
-  - Defines macro constants (like CPUrun, toggling if particles are created on CPU or GPU, etc)
-  - Defines physical constants, like pi
-  - Create E&B Fields:
-    - Loads `input_deck_2_7.txt`, a COMSOL file. See H. White 'Anomalous Thrust Production from an RF Test Device Measured on a Low-Thrust Torsion Pendulum' for the COMSOL details on this experiment.
-    - Reads the electric field file's array's rows and col's, and other COMSOL consts internally from the file.
-    - Uses `EB_LOAD.cuh` to load the `complex/b_real.txt` `complex/a_imag.txt` etc field files to arrays.
-    - Then starts calculating lagrangian packet parameters like real particles per macroparticle, calculates Cone (E-M cavity shape) geometry and those parameters.
-  - Original CPU Calculation:
-    - Defined last in `QThrusterMain.cu`, unoptimized CPU calculations.
-    - Original derivation, doesn't currently work.
-  - CPU:
-    - Modern CPU kernel is then defined if not using GPU. See "Compilation Modes"
-    - Manually creates electrons/positrons as normal, but can see a little clearer than in the GPU, but with optimizations.
-  - GPU:
-    - Starts managing branches as outlined in the QThrusterMemoryOrganization.pptx, transfers particles to GPU, or can create them on GPU.
-    - ThrustCalculationGPU 1-3 are then called, along with the rest of te sim (ParticleMoverGPU, TransferParticlesGPU, ThrustCalculations, etc).
-    - GPU memory is then freed, and some statistics are shown.

# License
MIT
