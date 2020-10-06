#define STATS_ENABLE_EIGEN_WRAPPERS
#include "stats.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <armadillo>


constexpr auto P = 4;
constexpr auto N_chains = 10;
using Chains = Eigen::Matrix<double, P, N_chains>;

const arma::mat eig2arma(Eigen::Vector2d eig) {
    const auto a = arma::mat(eig.data(), eig.rows(), eig.cols(), false, false);
    return a;
}

const Eigen::MatrixXd arma2eig(arma::mat arm) {
    return Eigen::Map<Eigen::MatrixXd>(arm.memptr(), arm.n_rows, arm.n_cols);
}


int main(int argc, char** argv) {
    Eigen::Vector2<double> x = {2.0, 5.0};
    auto mu = Eigen::Vector2<double>::Zero();
    auto var = Eigen::Vector2<double>::Ones();
    auto a = eig2arma(var);
    std::cout << var << '\n' << a << std::endl;
    return 0;
}
