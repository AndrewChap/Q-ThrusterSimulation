# Q-ThrusterSimulation
This simulation is similar to a particle-in-cell simulation and is parallelized for the GPU by cell (refered to in the code as "branches").  Particles are moved in the ParticleMoverGPU function.  Particles are transferred efficiently between cells through the functions "FillGapsGPU" and "TransferParticlesGPU".  GPU memory is managed so that memory coalescense is maximized, resulting in the minimum required calls to global GPU memory.  See QThrusterMemoryOrganization.pptx for more information (contains ppt animations.)
