project(autoferry_mcmc)
cmake_minimum_required( VERSION 3.0.0 )

set(CSV_CXX_STANDARD 20)
add_subdirectory(csv-parser)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_library(MCMC libmcmc.so PATHS mcmc/)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
find_package(Python3 COMPONENTS Development NumPy)
set(BLA_VENDOR Intel10_64lp_seq)
find_package(BLAS REQUIRED)
find_package(OpenMP)
find_package(LAPACK)


include_directories(stats/include)
include_directories(gcem/include)
include_directories(mcmc/include)
include_directories(csv-parser/single_include)
include_directories(eigen)
include_directories(/usr/include/mkl)
add_executable( ${PROJECT_NAME} src/main.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_include_directories(${PROJECT_NAME} PRIVATE ${MATPLOTLIB_CPP_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${MCMC} Python3::Python Python3::NumPy)