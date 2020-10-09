#include "mcmc.hpp"

#define STATS_ENABLE_ARMA_WRAPPERS
#include <armadillo>
#include <cmath>
#include <iostream>

#include "csv.hpp"
#include "gcem.hpp"
#include "stats.hpp"
#include <omp.h>
#include "matplotlibcpp.h"

constexpr auto P = 4;
constexpr auto N_chains = 10;

constexpr double PI = 3.14;

constexpr double THETA = PI / 4;


std::tuple<arma::vec, arma::Col<int>> load_csv() {
    std::vector<double> theta_vec;
    std::vector<int> intention_vec;
    csv::CSVReader reader("data.csv");
    int i = 0;
    for (csv::CSVRow &row : reader) {
        theta_vec.push_back(row[0].get<double>());
        intention_vec.push_back(row[1].get<int>());
        if (++i >= 1000) {
            break;
        }
    }
    auto theta = arma::vec(theta_vec);
    auto intention = arma::Col<int>(intention_vec);
    return std::tuple(theta, intention);
}

auto data = load_csv();

arma::mat softmax(const arma::mat& p) {
    auto exp = arma::exp(p);
    auto sum = arma::sum(exp, 0);
    return exp * 1.0 / arma::repelem(sum, 3, 1);
}

arma::vec3 map_probs(double theta, double c, const arma::vec3& alpha) {
    const arma::mat33 m =
        theta <= 0.0
            ? arma::mat33({{(std::exp(-c * std::abs(theta))), 1.0 / 2.0, 0.0},
                           {(1 - std::exp(c * theta)), 1.0 / 2.0, 0.0},
                           {(0.0), 0.0, 1.0}})
            : arma::mat33({{(std::exp(-c * std::abs(theta))), 1.0 / 2.0, 0.0},
                           {(0.0), 0.0, 1.0},
                           {(1 - std::exp(-c * theta)), 1.0 / 2.0, 0.0}});
    return m * alpha;
}

double ldirchilet(const arma::vec3& x, const arma::vec3& p) {
    return arma::sum(
                   (p - arma::vec3(arma::fill::ones)) % arma::log(x) - arma::vec3({std::lgamma(p[0]), std::lgamma(p[1]), std::lgamma(p[2])})) +
           std::lgamma(arma::accu(p));
}

double lgamma(double x, double shape, double scale) {
    return (shape - 1) * std::log(x) - x / scale - gcem::lgamma(shape) -
           shape * std::log(scale);
}

double log_likelihood(const arma::vec &p, void *ll_data) {
    auto c = std::exp(p[0]);
    auto a = softmax(p.rows(1,3));

    auto &[theta, intention] = data;

    static auto probs = arma::mat(3, theta.n_rows);
    #pragma omp parallel for
    for (int i = 0; i < theta.n_rows; ++i) {
        probs.col(i) = map_probs(theta[i], c, a);
    }
    arma::mat lprobs = arma::log(probs);
    auto lp_c = lgamma(c*8, 7.5, 1.0);
    auto lp_a = ldirchilet(a, {1, 0.3, 0.1});
    auto ll_intention = 0.0;
    #pragma omp parallel for reduction(+:ll_intention)
    for (int i = 0; i < intention.n_rows; ++i) {
        ll_intention += lprobs.col(i)[intention[i]];
    }
    static int i = 0;
    auto ll = lp_c + lp_a + ll_intention;// / intention.n_rows;
    if (++i % 10000 == 0) {
        std::cout << i << ": " << ll << std::endl;
        printf("* c:     %.4f - %.4f\n", c, lp_c);
        printf("* a1:    %.4f - %.4f\n", a[0], lp_a);
        printf("* a2:    %.4f - %.4f\n", a[1], lp_a);
        printf("* a3:    %.4f - %.4f\n", a[2], lp_a);
        printf("* ll_intention: %.4f\n", ll_intention);
    }
    return ll;
}

std::tuple<arma::vec, arma::mat> run(mcmc::algo_settings_t settings) {
    arma::vec4 initial = {1.0, 0.34, 0.33, 0.33};
    arma::mat draws;
    std::cout << "Starting MCMC - Chain Number " << omp_get_thread_num() << std::endl;
    mcmc::rwmh(initial, draws, log_likelihood, nullptr, settings);
    std::cout << "MCMC Done - Chain Number " << omp_get_thread_num() << std::endl;
    arma::vec c = arma::exp(draws.col(0));
    arma::mat a = softmax(draws.cols(1, 3).t());
    std::cout << "c: " << arma::mean(c) << std::endl;
    std::cout << "a: " << arma::mean(a, 1) << std::endl;
    return {c, a};
}

namespace plt = matplotlibcpp;
int main(int argc, char** argv) {
    mcmc::algo_settings_t settings;
    settings.rwmh_n_draws = 5e5;
    auto [c, a] = run(settings);
    auto c_raw = arma::conv_to<std::vector<double>>::from(c);
    auto a0_raw = arma::conv_to<std::vector<double>>::from(a.row(0));
    auto a1_raw = arma::conv_to<std::vector<double>>::from(a.row(1));
    auto a2_raw = arma::conv_to<std::vector<double>>::from(a.row(2));
    plt::subplot(2, 2, 1);
    plt::hist(c_raw, 50);
    plt::title("C");
    plt::subplot(2, 2, 2);
    plt::hist(a0_raw, 50);
    plt::title("A0");
    plt::subplot(2, 2, 3);
    plt::hist(a1_raw, 50);
    plt::title("A1");
    plt::subplot(2, 2, 4);
    plt::hist(a2_raw, 50);
    plt::title("A2");
    plt::save("./c.png");
    return 0;
}