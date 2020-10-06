project(autoferry_mcmc)
cmake_minimum_required( VERSION 3.0.0 )

find_package(OpenMP)

include_directories(stats/include)
include_directories(gcem/include)
include_directories(mcmc/include)
include_directories(eigen)
find_library(MCMC mcmc)
add_executable( ${PROJECT_NAME} main.cpp )
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX ${MCMC})