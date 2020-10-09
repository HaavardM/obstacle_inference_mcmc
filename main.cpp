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
#include <algorithm>


constexpr auto P = 4;
constexpr auto N_chains = 10;

constexpr double PI = 3.14;

constexpr double THETA = PI / 4;
constexpr double RW_SCALING = 1.0 / 50.0;


std::tuple<arma::vec, arma::Col<int>> load_csv() {
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

    auto c = std::exp(p[0] * RW_SCALING);
    auto a = softmax(p.rows(1,3) * RW_SCALING);

    auto &[theta, intention] = data;

    static auto probs = arma::mat(3, theta.n_rows);
    #pragma omp parallel for
    for (int i = 0; i < theta.n_rows; ++i) {
        probs.col(i) = map_probs(theta[i], c, a);
    }
    arma::mat lprobs = arma::log(probs);
    auto lp_c = lgamma(c*10, 7.5, 1.0);
    auto lp_a = ldirchilet(a, {20, 4, 2.5});
    auto ll_intention = 0.0;
    #pragma omp parallel for simd reduction(+:ll_intention)
    for (int i = 0; i < intention.n_rows; ++i) {
        ll_intention += lprobs.col(i)[intention[i]];
    }
    static int i = 0;
    auto ll = lp_c + lp_a;// + ll_intention;// / static_cast<double>(intention.size());// / intention.n_rows;
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
    arma::vec4 initial(arma::fill::randu);
    initial.row(0) *= 2;
    arma::mat draws;
    std::cout << "Starting MCMC - Chain Number " << omp_get_thread_num() << std::endl;
    mcmc::rwmh(initial, draws, log_likelihood, nullptr, settings);
    std::cout << "MCMC Done - Chain Number " << omp_get_thread_num() << std::endl;
    arma::vec c = arma::exp(draws.col(0)*RW_SCALING);
    arma::mat a = softmax(draws.cols(1, 3).t()*RW_SCALING);
    std::cout << "c: " << arma::mean(c) << std::endl;
    std::cout << "a: " << arma::mean(a, 1) << std::endl;
    std::cout << "Accept: " << settings.rwmh_accept_rate << std::endl;
    return {c, a};
}

namespace plt = matplotlibcpp;
int main(int argc, char** argv) {
    mcmc::algo_settings_t settings;
    settings.rwmh_n_draws = 3e5;
    auto [c, a] = run(settings);
    auto res_c = arma::conv_to<std::vector<double>>::from(c);
    auto res_a0 = arma::conv_to<std::vector<double>>::from(a.row(0));
    auto res_a1 = arma::conv_to<std::vector<double>>::from(a.row(1));
    auto res_a2 = arma::conv_to<std::vector<double>>::from(a.row(2));
    std::cout << res_c.size() << std::endl;


    std::vector<double> x, pc, pa0, pa1, pa2;
    auto N = 1000;
    for (double i = 0.0; i <= 1.0; i += 1.0 / N) {
        x.push_back(i);
        pc.push_back(std::exp(lgamma(i*10, 7.5, 1.0))*static_cast<double>(settings.rwmh_n_draws) / 2.5);
    }


    plt::subplot(2, 2, 1);
    plt::plot(x, pc, "r");
    plt::xlim(0.0, 1.0);
    plt::title("C");
    plt::subplot(2, 2, 2);
    plt::hist(res_a0, 50);
    plt::xlim(0.0, 1.0);
    plt::title("A0");
    plt::subplot(2, 2, 3);
    plt::hist(res_a1, 50);
    plt::xlim(0.0, 1.0);
    plt::title("A1");
    plt::subplot(2, 2, 4);
    plt::hist(res_a2, 50);
    plt::xlim(0.0, 1.0);
    plt::title("A2");
    plt::tight_layout();
    plt::save("./c.png");
    return 0;
}