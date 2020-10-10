#include "mcmc.hpp"

#include <cmath>
#include <iostream>

#include <armadillo>
#include "csv.hpp"
#include "gcem.hpp"
#include <omp.h>
#include "matplotlibcpp.h"

#define EIGEN_USE_BLAS
#include "Eigen/Dense"
#include "unsupported/Eigen/SpecialFunctions"
#include <algorithm>


constexpr auto P = 4;
constexpr auto N = 1000;

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
        if (++i >= N) {
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
void apply_softmax(Eigen::MatrixBase<D>& p) {
    auto exp = p.array().exp();
    p = exp.array().rowwise() / exp.array().colwise().sum();
}

double intention_probs(double theta, double c, const Eigen::Vector3d& alpha, int intention) {
    if (theta <= 0.0) {
        return (Eigen::Matrix<double, 3, 3, Eigen::RowMajor>({{(std::exp(-c * std::abs(theta))), 1.0 / 2.0, 0.0},
                           {(1 - std::exp(c * theta)), 1.0 / 2.0, 0.0},
                           {(0.0), 0.0, 1.0}}).row(intention) * alpha);
    } else {
        return (Eigen::Matrix<double, 3, 3, Eigen::RowMajor>({{(std::exp(-c * std::abs(theta))), 1.0 / 2.0, 0.0},
                           {(0.0), 0.0, 1.0},
                           {(1 - std::exp(-c * theta)), 1.0 / 2.0, 0.0}}).row(intention) * alpha);
    }
}

double ldirchilet(const Eigen::Vector3d& x, const Eigen::Vector3d& p) {
    auto v = (p.array() - 1.0) * x.array().log() - p.array().lgamma();
    return v.sum() + gcem::lgamma(p.sum());
}

double lgamma(double x, double shape, double scale) {
    return (shape - 1) * std::log(x) - x / scale - gcem::lgamma(shape) -
           shape * std::log(scale);
}

double log_prior(const Eigen::Vector4<double> p) {
    double lp_c = lgamma(p(0)*10, 7.5, 1.0);
    double lp_a = ldirchilet(p.tail(3), Eigen::Vector3d({20, 4, 2.5}));
    return lp_c + lp_a;
}

double log_likelihood(const Eigen::Vector4<double> p) {
    auto &[theta, intention] = data;

    auto ll_intention = 0.0;
    #pragma omp parallel for simd reduction(+:ll_intention)
    for (int i = 0; i < intention.size(); ++i) {
        auto c = p(0);
        auto a = p.tail<3>();
        ll_intention += std::log(intention_probs(theta(i), p(0), a, intention(i)));
    }

    return ll_intention;
}

double log_target(const arma::vec &arma_p, void *ll_data) {
    Eigen::Vector4d p = {
        arma_p[0],
        arma_p[1],
        arma_p[2],
        arma_p[3]
    };
    p *= RW_SCALING;
    p(0) = std::exp(p(0));
    auto a = p.bottomRows<3>();
    apply_softmax(a);

    static int i = 0;
    auto ll = log_prior(p) + log_likelihood(p);// / static_cast<double>(intention.size());// / intention.n_rows;
    if (++i % 10000 == 0) {
        std::cout << i << ": " << ll << std::endl;
        printf("* c:     %.4f\n", p(0));
        printf("* a1:    %.4f\n", p(1));
        printf("* a2:    %.4f\n", p(2));
        printf("* a3:    %.4f\n", p(3));
        printf("* ll_intention: %.4f\n", log_likelihood(p));
    }
    return ll;
}

std::tuple<Eigen::VectorXd, Eigen::Matrix3Xd> run(mcmc::algo_settings_t settings) {
    arma::vec4 initial(arma::fill::randu);
    initial.row(0) *= 2;
    arma::mat arma_draws;
    std::cout << "Starting MCMC - Chain Number " << omp_get_thread_num() << std::endl;
    mcmc::rwmh(initial, arma_draws, log_target, nullptr, settings);
    std::cout << "MCMC Done - Chain Number " << omp_get_thread_num() << std::endl;

    // Convert to Eigen and display results
    auto draws = Eigen::Map<Eigen::MatrixXd>(arma_draws.memptr(), arma_draws.n_rows, arma_draws.n_cols);
    draws *= RW_SCALING;
    auto c = (draws.col(0)).array().exp();
    auto a = draws.rightCols(3).transpose();
    apply_softmax(a);
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