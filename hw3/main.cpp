#include <iostream>
#include <iomanip> // for setprecision
#include <random>
#include <cstdlib> // for atof
#include <chrono> // for high_resolution_clock
#include <cmath> // for pow

int main(int argc, char* argv[]) {
    long long int n;
    if (argc == 1) {
        n = int(1e6); // Default number of samples (default format is decimal)
    }
    else if (argc == 2) {
        n = int(std::atof(argv[1])); // Number of samples (default format is decimal)
    }
    else {
        std::cerr << "Usage: " << argv[0] << " [n]" << std::endl; // argv = ['./mc_pi', 'n']
        return 1;
    }

    auto exec_start = std::chrono::high_resolution_clock::now(); // Start timer
    std::random_device rd; // Random seed from hardware
    std::mt19937 gen(rd()); // Mersenne twister engine
    std::uniform_real_distribution<> rand(0., 1.);

    long long int h = 0;
    auto sample_start = std::chrono::high_resolution_clock::now(); // Start timer
    for (long long int i = 0; i < n; ++i) {
        // Get random points
        const double x = rand(gen);
        const double y = rand(gen);
        // Check if point is inside the circle
        if (x*x + y*y <= 1.) h++;
    }
    auto sample_end = std::chrono::high_resolution_clock::now(); // End timer

    double pi_est = 4. * double(h) / double(n);
    auto exec_end = std::chrono::high_resolution_clock::now(); // End timer

    auto sample_time = std::chrono::duration_cast<std::chrono::microseconds>(sample_end-sample_start).count() * 1e-6 / n; // Duration in seconds
    auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(exec_end-exec_start).count() * 1e-6; // Duration in seconds

    std::cout << std::fixed << std::setprecision(6) << pi_est;
    std::cout << "\t" << std::scientific << std::setprecision(0) << double(n);
    std::cout << "\t" << std::setprecision(6) << exec_time;
    std::cout << "\t" << 1./sample_time;
    std::cout << "\t" << sample_time << std::endl;
    return 0;
}