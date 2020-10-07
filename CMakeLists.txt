project(autoferry_mcmc)
cmake_minimum_required( VERSION 3.0.0 )

find_package(OpenMP)
find_package(LAPACK)
set(CSV_CXX_STANDARD 20)
add_subdirectory(csv-parser)

find_library(MCMC libmcmc.so PATHS mcmc/)

include_directories(stats/include)
include_directories(gcem/include)
include_directories(mcmc/include)
include_directories(csv-parser/single_include)
include_directories(eigen)
add_executable( ${PROJECT_NAME} main.cpp )
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES} ${MCMC})