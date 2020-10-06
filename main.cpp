#define STATS_ENABLE_EIGEN_WRAPPERS
#include "stats.hpp"
#include <iostream>
#include <Eigen/Dense>


int main(int argc, char** argv) {
    Eigen::Vector2<double> x = {2.0, 5.0};
    auto mu = Eigen::Vector2<double>::Zero();
    auto var = Eigen::Vector2<double>::Ones();
    std::cout << stats::dnorm(x, 5.0, 0.001, true) << std::endl;
    return 0;
}
