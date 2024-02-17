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
- Visualize Force Results: `julia ForceCurveReader.jl`
  - May need to copy in the ForceCurveGPU output file and do manual error corrections yourself.
- Visualize Sim:
  - ParaView https://www.paraview.org/download/

# License
MIT
