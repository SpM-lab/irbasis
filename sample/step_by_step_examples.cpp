#include <iostream>
#include <vector>
#include "irbasis.hpp"

double rho_omega_metal(double omega) {
    return (2/M_PI) * std::sqrt(1-omega*omega);
}

int main() {
    double beta = 100.0;
    double wmax = 1.0;
    double Lambda = wmax * beta;
    irbasis::basis b = irbasis::load("F", Lambda, "./irbasis.h5");

    int dim = b.dim();

    // (1) Semicircular DOS on [-1, 1]
    // (2) Pole at \omega = -1,+1
    // Comment out one of them
    std::string model = "Metal";
    //std::string model = "Insulator";

    /*
     * Compute rho_l from rho(omega)
     */
    std::vector<double> rho_l(dim, 0.0);
    if (model == "Metal") {
        // We use a simple numerical integration on a uniform mesh
        //  nomega: Number of omega points for numerical integration.
        // Please use an adaptive numerical integration method (such as quad in GSL) for better accuracy!
        int nomega = 100000;
        double dw = 2.0/nomega;
        for (int l=0; l<dim; ++l) {
            for (int w=0; w<nomega; ++w) {
                double omega = 2.0*w/(nomega-1) - 1.0;
                double y = omega;
                rho_l[l] += rho_omega_metal(omega) * b.vly(l, y);
           }
           rho_l[l] *= std::sqrt(1/wmax) * dw;
        }
    } else if (model == "Insulator") {
        double pole = 1.0;
        for (int l=0; l<dim; ++l) {
            rho_l[l] = 0.5 * std::sqrt(1/wmax)* (b.vly(l, pole/wmax) + b.vly(l, -pole/wmax));
        }
    } else {
        throw std::runtime_error("Uknown model");
    }

    /*
     * Compute g_l
     */
    std::vector<double> Sl(dim), gl(dim);
    for (int l=0; l<dim; ++l) {
        Sl[l] = std::sqrt(0.5 * beta * wmax) * b.sl(l);
        gl[l] = - Sl[l] * rho_l[l];
    }
    
    std::cout << "# rho_l                g_l " << std::endl;
    for (int l=0; l<dim; ++l) {
        std::cout << l << " " << std::setprecision(15) << rho_l[l] << " " << gl[l] << std::endl;
    }

    /*
     * Compute G(tau) on a uniform mesh from tau=0 to beta.
     */
    std::cout << "# G(tau)" << std::endl;
    int n_tau = 50;
    std::vector<double> gtau(n_tau, 0.0);
    for (int t=0; t<n_tau; ++t) {
        double tau = (beta/(n_tau-1)) * t;
        double x = 2*tau/beta - 1;
        for (int l=0; l<dim; ++l) {
            gtau[t] += std::sqrt(2/beta) * b.ulx(l, x) * gl[l];
        }
        std::cout << t << " " << gtau[t] << std::endl;
    }

    /*
     * Compute G(iwn)
     */
    std::cout << "# wn    G(iwn)     1/wn" << std::endl;
    int n_iw = 100;
    std::vector<long long> ns;
    for (int n=0; n<n_iw; ++n) {
        ns.push_back(n);
    }
    std::vector<std::vector<std::complex<double> > > unl = b.compute_unl(ns);
    for (int n=0; n<n_iw; ++n) {
        std::complex<double> giwn = 0.0;
        for (int l=0; l<dim; ++l) {
            giwn += unl[n][l] * gl[l];
        }
        giwn *= std::sqrt(beta);

        double wn = (2 * n + 1) * M_PI/beta;
        std::cout << wn << "  " << giwn << "  " << 1/wn << std::endl;
    }
    
}
