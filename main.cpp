#include "mcmc.hpp"

#define STATS_ENABLE_ARMA_WRAPPERS

#include <cmath>
#include <iostream>

#include "csv.hpp"
#include "gcem.hpp"
#include "stats.hpp"
#include <omp.h>
#include "matplotlibcpp.h"

#include "Eigen/Dense"
#include <algorithm>

#include <armadillo>

constexpr auto P = 4;
constexpr auto N = 100;

constexpr double PI = 3.14;

constexpr double THETA = PI / 4;
constexpr double RW_SCALING = 1.0 / 50.0;


std::tuple<Eigen::Vector<double, N>, Eigen::Vector<int, N>> load_csv() {
    std::vector<double> theta_vec;
    std::vector<int> intention_vec;
    csv::CSVReader reader("data.csv");
    int i = 0;
    for (csv::CSVRow &row : reader) {
        theta_vec.push_back(row[0].get<double>());
        intention_vec.push_back(row[1].get<int>());
        if (++i >= 100) {
            break;
        }
    }
    Eigen::Vector<double, N> theta;
    Eigen::Vector<int, N> intention;

    for (int i = 0; i < theta_vec.size(); ++i) {
        theta(i) = theta_vec[i];
        intention(i) = intention_vec[i];
    }
    return std::tuple(theta, intention);
}

auto data = load_csv();

template<typename D>
void softmax(Eigen::MatrixBase<D>& p) {
    auto exp = p.array().exp();
    p = exp.array().rowwise() / exp.array().colwise().sum();
}

Eigen::Vector3d map_probs(double theta, double c, const Eigen::Vector3d& alpha) {
    const Eigen::Matrix3d m =
        theta <= 0.0
            ? Eigen::Matrix3d({{(std::exp(-c * std::abs(theta))), 1.0 / 2.0, 0.0},
                           {(1 - std::exp(c * theta)), 1.0 / 2.0, 0.0},
                           {(0.0), 0.0, 1.0}})
            : Eigen::Matrix3d({{(std::exp(-c * std::abs(theta))), 1.0 / 2.0, 0.0},
                           {(0.0), 0.0, 1.0},
                           {(1 - std::exp(-c * theta)), 1.0 / 2.0, 0.0}});
    return m * alpha;
}

double ldirchilet(const Eigen::Vector3d& x, const Eigen::Vector3d& p) {
    return ((p.array() - 1.0) * x.array().log() - Eigen::Vector3d({std::lgamma(p[0]), gcem::lgamma(p[1]), gcem::lgamma(p[2])}).array()).sum() +
           gcem::lgamma(p.sum());
}

double lgamma(double x, double shape, double scale) {
    return (shape - 1) * std::log(x) - x / scale - gcem::lgamma(shape) -
           shape * std::log(scale);
}

double log_likelihood(const arma::vec &arma_p, void *ll_data) {

    const auto p = Eigen::Map<const Eigen::VectorXd>(arma_p.memptr(), arma_p.n_rows);
    auto c = std::exp(p(0) * RW_SCALING);
    Eigen::Vector3d a = p.tail(3) * RW_SCALING; 
    softmax(a);

    auto &[theta, intention] = data;

    Eigen::Matrix<double, 3, N> probs;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        probs.col(i) = map_probs(theta(i), c, a);
    }
    Eigen::Matrix<double, 3, N> lprobs = probs.array().log();
    auto lp_c = lgamma(c*10, 7.5, 1.0);
    auto lp_a = ldirchilet(a, {20, 4, 2.5});
    auto ll_intention = 0.0;
    #pragma omp parallel for simd reduction(+:ll_intention)
    for (int i = 0; i < intention.size(); ++i) {
        ll_intention += lprobs(intention(i), i);
    }
    static int i = 0;
    auto ll = lp_c + lp_a;// + ll_intention;// / static_cast<double>(intention.size());// / intention.n_rows;
    if (++i % 10000 == 0) {
        std::cout << i << ": " << ll << std::endl;
        printf("* c:     %.4f - %.4f\n", c, lp_c);
        printf("* a1:    %.4f - %.4f\n", a(0), lp_a);
        printf("* a2:    %.4f - %.4f\n", a(1), lp_a);
        printf("* a3:    %.4f - %.4f\n", a(2), lp_a);
        printf("* ll_intention: %.4f\n", ll_intention);
    }
    return ll;
}

std::tuple<Eigen::VectorXd, Eigen::Matrix3Xd> run(mcmc::algo_settings_t settings) {
    arma::vec4 initial(arma::fill::randu);
    initial.row(0) *= 2;
    arma::mat arma_draws;
    std::cout << "Starting MCMC - Chain Number " << omp_get_thread_num() << std::endl;
    mcmc::rwmh(initial, arma_draws, log_likelihood, nullptr, settings);
    std::cout << "MCMC Done - Chain Number " << omp_get_thread_num() << std::endl;

    Eigen::MatrixXd draws = Eigen::Map<Eigen::MatrixXd>(arma_draws.memptr(), arma_draws.n_rows, arma_draws.n_cols);
    Eigen::VectorXd c = (draws.col(0)*RW_SCALING).array().exp();
    Eigen::Matrix3Xd a = draws.rightCols(3).transpose()*RW_SCALING;
    softmax(a);
    std::cout << "c: " << c.mean() << std::endl;
    std::cout << "a: " << a.rowwise().mean() << std::endl;
    std::cout << "Accept: " << settings.rwmh_accept_rate << std::endl;
    return {c, a};
}

namespace plt = matplotlibcpp;
int main(int argc, char** argv) {
    mcmc::algo_settings_t settings;
    settings.rwmh_n_draws = 3e5;
    auto [c, a] = run(settings);
    return 0;
}