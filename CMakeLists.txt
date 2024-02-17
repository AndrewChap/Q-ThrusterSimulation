cmake_minimum_required(VERSION 3.18) # Required for CUDA language support
project(particle_branch_11 LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(PROJECT_SOURCE_DIR "particle_branch_11")

include_directories(
    # Add CUDA toolkit include directories
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Define source and header files for the project
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/QThrusterMain.cu
    # Add any .cpp files if necessary
)

set(HEADER_FILES
    ${PROJECT_SOURCE_DIR}/AddParticlesGPU.cuh
    ${PROJECT_SOURCE_DIR}/BoxConeCollider.cuh
    ${PROJECT_SOURCE_DIR}/CalculateCellVolumesGPU.cuh
    ${PROJECT_SOURCE_DIR}/DataIO.cuh
    ${PROJECT_SOURCE_DIR}/FillGapsCPU.cuh
    ${PROJECT_SOURCE_DIR}/FillGapsGPU.cuh
    ${PROJECT_SOURCE_DIR}/gpuErrchk.cuh
    ${PROJECT_SOURCE_DIR}/Load_EB.cuh
    ${PROJECT_SOURCE_DIR}/OtherFunctions.cuh
    ${PROJECT_SOURCE_DIR}/ParticleMoverCPU.cuh
    ${PROJECT_SOURCE_DIR}/ParticleMoverGPU.cuh
    ${PROJECT_SOURCE_DIR}/ThrustCalculationGPU.cuh
    ${PROJECT_SOURCE_DIR}/ThrustCalculationGPU1.cuh
    ${PROJECT_SOURCE_DIR}/ThrustCalculationGPU2.cuh
    ${PROJECT_SOURCE_DIR}/TimerWrappers.cuh
    ${PROJECT_SOURCE_DIR}/TransferParticlesCPU.cuh
    ${PROJECT_SOURCE_DIR}/TransferParticlesGPU.cuh
    ${PROJECT_SOURCE_DIR}/WriteForce.cuh
    ${PROJECT_SOURCE_DIR}/WriteToSnowballGPU.cuh
    ${PROJECT_SOURCE_DIR}/WriteVTK.cuh
)

# Add executable target with source and header files
add_executable(particle_branch_11 ${SOURCE_FILES} ${HEADER_FILES})

#find_package(CUDA 7.5 REQUIRED)

# Specify include directories
include_directories(
    # Specify the directories where the compiler can find headers
    # Example: "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}"
)

# Find CUDA
find_package(CUDA 10.0 REQUIRED) # Adjust based on your CUDA version

# Link CUDA libraries
target_link_libraries(particle_branch_11
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Specify additional options for the CUDA compiler
set_target_properties(particle_branch_11 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# Copy the configuration text file to the build directory
configure_file(${PROJECT_SOURCE_DIR}/input_deck_2_7.txt ${CMAKE_CURRENT_BINARY_DIR}/input_deck_2_7.txt COPYONLY)
# Copy the COMSOL field files to the build directory
file(COPY ${PROJECT_SOURCE_DIR}/complex DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
