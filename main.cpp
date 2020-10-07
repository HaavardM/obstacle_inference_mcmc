#include "mcmc.hpp"

#define STATS_ENABLE_ARMA_WRAPPERS
#include "stats.hpp"
#include <armadillo>
#include <iostream>
#include <cmath>

#include "gcem.hpp"
#include "csv.hpp"

constexpr auto P = 4;
constexpr auto N_chains = 10;

constexpr double PI = 3.14;

constexpr double THETA = PI / 4;


std::tuple<arma::vec, arma::Col<int>> load_csv() {

    std::vector<double> theta_vec;
    std::vector<int> intention_vec;
    csv::CSVReader reader("data.csv");
    int i = 0;
    for (csv::CSVRow& row : reader) {
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





arma::mat softmax(arma::mat p) {
    auto exp = arma::exp(p);
    auto sum = arma::sum(exp, 0);
    return exp * 1.0 / arma::repelem(sum, 3, 1);
}

arma::vec3 map_probs(double theta, double c, arma::vec3 alpha) {
    const arma::mat33 m = theta >= 0.0 ? arma::mat33({
                               {(std::exp(-c*std::abs(theta))), 1.0 / 2.0, 0.0},
                               {(1 - std::exp(-c*theta))      , 1.0 / 2.0, 0.0}, 
                               {(0.0)                       , 0.0, 1.0}
                            }) : arma::mat33({
                               {(std::exp(-c*std::abs(theta))), 1.0 / 2.0, 0.0},
                               {(0.0)                       , 0.0, 1.0}, 
                               {(1 - std::exp(c*theta))       , 1.0 / 2.0, 0.0}
                              });
    return m*alpha;
}


double ll_dirchilet(arma::vec3 x, arma::vec3 p) {
    return arma::sum(arma::vec3(
                        (p-arma::vec3(arma::fill::ones)) % arma::log(x) 
                        - arma::vec3({
                                        gcem::lgamma(p[0]),
                                        gcem::lgamma(p[1]), 
                                        gcem::lgamma(p[2])
                                    }
                        )
                     ))
                        + gcem::lgamma(arma::sum(p));
}


double ll_gamma(double x, double shape, double scale) {
    return (shape - 1)*std::log(x) - x / scale - gcem::lgamma(shape) - shape*std::log(scale);
}

double log_likelihood(const arma::vec& p, void* ll_data) {
    auto c = std::exp(p[0]);
    auto a = softmax(arma::vec3({p[1], p[2], p[3]}));

    auto& [theta, intention] = data;

    auto probs = arma::mat(3, theta.n_rows);
    for (int i = 0; i < theta.n_rows; ++i) {
        probs.col(i) = map_probs(theta[i], c, a);
    }
    arma::mat lprobs = arma::log(probs);
    auto ll_c = ll_gamma(c, 2.0,  2.0);
    auto ll_a = ll_dirchilet(a, {4.0, 2.0, 1.0});
    auto ll_intention = 0.0;
    for (int i = 0; i < intention.n_rows; ++i) {
        ll_intention += lprobs.col(i)[intention[i]];
    }
    auto ll =  ll_c + ll_a + ll_intention / intention.n_rows;
    static int i = 0;
    if (i++ % 1000 == 0) {
        std::cout << i << ": " << ll << std::endl;
        printf("* c: %.4f - %.4f\n", c, ll_c);
        printf("* a1: %.4f - %.4f\n", a[0], ll_a);
        printf("* a2: %.4f - %.4f\n", a[1], ll_a);
        printf("* a3: %.4f - %.4f\n", a[2], ll_a);
        printf("* ll_intention: %.4f\n", ll_intention);
    } 
    return ll;
}


int main(int argc, char** argv) {

    arma::vec4 initial = {
        1.0, 0.34, 0.33, 0.33
    };

    arma::mat draws;
    std::cout << "Starting MCMC" << std::endl;
    mcmc::rwmh(initial, draws, log_likelihood, nullptr);
    std::cout << "MCMC Done" << std::endl;
    arma::vec c = arma::exp(draws.col(0));
    arma::mat a = softmax(draws.cols(1, 3).t());
    std::cout << "c: " << arma::mean(c) << std::endl;
    std::cout << "a: " << arma::mean(a, 1) << std::endl; 
    return 0;
}