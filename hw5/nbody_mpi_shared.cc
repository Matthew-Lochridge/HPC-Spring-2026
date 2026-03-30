#include <iostream> // Standard I/O
#include <fstream> // File I/O
#include <random> // Random number generators
#include <vector> // Vector (dynamic array)
#include <tuple> // Tuple (multiple return values)
#include <chrono> // Time utilities
#include <mpi.h> // MPI (parallelism)

// Global constants
static int N = 128; // Number of masses
static const int D = 3; // Dimensionality
static int ND = N * D; // Size of the state vectors
static const double G = 0.5; // Gravitational constant
static const double dt = 1e-3; // Time step size
static const int T = 300; // Number of time steps
static const double t_max = static_cast<double>(T) * dt; // Maximum time
static const double x_min = 0.; // Minimum position
static const double x_max = 1.; // Maximum position
static const double v_min = 0.; // Minimum velocity
static const double v_max = 0.; // Maximum velocity
static const double m_0 = 1.; // Mass value
static const double epsilon = 0.01; // Softening parameter
static const double epsilon2 = epsilon * epsilon; // Softening parameter^2
// Note that epsilon must be greater than zero!

using Vec = std::vector<double>; // Vector type
using Vecs = std::vector<Vec>; // Vector of vectors type

// Random number generator
static std::mt19937 gen; // Mersenne twister engine
static std::uniform_real_distribution<> ran(0., 1.); // Uniform distribution

static int rank, n_ranks; // Process rank and number of processes
static std::vector<int> counts, displs; // Counts and displacements for MPI_Allgatherv
static std::vector<int> countsD, displsD; // State counts and displacements for MPI_Allgatherv
static int N_beg, N_end, N_local; // Mass range for each process [N_beg, N_end)
static int ND_beg, ND_end, ND_local; // State vector range for each process [ND_beg, ND_end)

// Shared memory for masses, positions, velocities, and accelerations
static double *m, *x, *v, *a, *x_next, *v_next; // Shared memory
static MPI_Win win_m, win_x, win_v, win_a, win_x_next, win_v_next; // Shared windows

// Set up parallelism
void setup_parallelism() {
    MPI_Init(NULL, NULL); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Unique process rank
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Get the current time and convert it to an integer
    auto now = std::chrono::high_resolution_clock::now();
    auto now_cast = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto now_int = now_cast.time_since_epoch().count();

    // Pure MPI version
    gen.seed(now_int ^ rank); // Seed the random number generator

    // Divide the masses among the processes (needed for MPI_Allgatherv)
    counts.resize(n_ranks); // Counts for each process
    displs.resize(n_ranks); // Displacements for each process
    countsD.resize(n_ranks); // State counts for each process
    displsD.resize(n_ranks); // State displacements for each process
    const int remainder = N % n_ranks; // Remainder of the division
    for (int i = 0; i < n_ranks; ++i) {
        counts[i] = N / n_ranks; // Divide the masses among the processes
        displs[i] = i * counts[i]; // Displacements where each segment begins
        if (i < remainder) {
            counts[i] += 1; // Correct the count
            displs[i] += i; // Correct the displacement
        } else {
            displs[i] += remainder; // Correct the displacement
        }
        countsD[i] = counts[i] * D; // State counts for each process
        displsD[i] = displs[i] * D; // State displacements for each process
    }

    // Set up the local mass ranges
    N_beg = displs[rank]; // Mass range for each process [N_beg, N_end)
    N_end = N_beg + counts[rank]; // Mass range for each process [N_beg, N_end)
    ND_beg = N_beg * D; // State vector range for each process [ND_beg, ND_end)
    ND_end = N_end * D; // State vector range for each process [ND_beg, ND_end)
    N_local = N_end - N_beg; // Local number of masses
    ND_local = ND_end - ND_beg; // Local size of the state vectors

    // Allocate shared memory for positions, velocities, and accelerations
    if (rank == 0) {
        MPI_Win_allocate_shared(N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_allocate_shared(ND * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v_next, &win_v_next);
    } else {
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &m, &win_m);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x, &win_x);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v, &win_v);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win_a);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x_next, &win_x_next);
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &v_next, &win_v_next);
    }
}

// Print a vector to a file
template <typename T>
void save(const std::vector<T>& vec, const std::string& filename, const std::string& header = "") {
    std::ofstream file(filename); // Open the file
    if (file.is_open()) { // Check for successful opening
        if (!header.empty())
            file << "# " << header << std::endl; // Write the header
        for (const auto& elem : vec)
            file << elem << " "; // Write each element
        file << std::endl; // Write a newline
        file.close(); // Close the file
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

// Generate random initial conditions for N masses
void initial_conditions() {
    const double dx = x_max - x_min; // Position range
    const double dv = v_max - v_min; // Velocity range
    for (int i = ND_beg; i < ND_end; ++i) {
        x[i] = ran(gen) * dx + x_min; // Random initial positions
        v[i] = ran(gen) * dv + v_min; // Random initial velocities
    }
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    return;
}

// Compute the acceleration of all masses
// a_i = G * sum_{ji} m_j * (x_j - x_i) / |x_j - x_i|^3
void acceleration() {
    for (int i = N_beg; i < N_end; ++i) {
        const int iD = i * D; // Flatten the index
        double dx[D]; // Difference in position
        for (int j = 0; j < N; ++j) {
            const int jD = j * D; // Flatten the index
            double dx2 = epsilon2; // Distance^2 (softened)
            for (int k = 0; k < D; ++k) {
                dx[k] = x[jD + k] - x[iD + k]; // Difference in position
                dx2 += dx[k] * dx[k]; // Distance^2
            }
            const double Gm_dx3 = G * m[j] / (dx2 * sqrt(dx2)); // G * m_j / |dx|^3
            for (int k = 0; k < D; ++k) {
                const int iDk = iD + k; // Flatten the index
                a[iDk] += Gm_dx3 * dx[k]; // Acceleration
            }
        }
    }
    return;
}

// Compute the next position and velocity for all masses
void timestep() {
    acceleration(); // Calculate particle accelerations
    for (int i = ND_beg; i < ND_end; ++i) {
        v_next[i] = a[i] * dt + v[i]; // New velocity
        x_next[i] = v_next[i] * dt + x[i]; // New position
    }
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    if (rank == 0) {
        std::swap(x, x_next); // Update the positions and velocities
        std::swap(v, v_next); // By swapping the pointers to the arrays
    }
    return;
}

double kinetic_energy() {
    // Calculate the total kinetic energy of the system
    double KE_n = 0.; // Kinetic energy
    for (int i = N_beg; i < N_end; ++i) {
        double v2 = 0.; // Velocity magnitude
        for (int j = 0; j < D; ++j) {
            const int k = i * D + j; // Flatten the index
            v2 += v[k] * v[k]; // Velocity magnitude
        }
        KE_n += 0.5 * m[i] * v2; // Kinetic energy
    }
    return KE_n; // Kinetic energy
}

// Main function
int main(int argc, char** argv) {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Set up the problem
    if (argc > 1) {
        N = std::atoi(argv[1]); // Update the number of masses
        ND = N * D; // Update the size of the state vectors
    }

    setup_parallelism(); // Initialize MPI

    // Prepare vectors for time points, masses, positions, velocities, accelerations, and kinetic energy
    Vec t(T+1); // Time points
    for (int i = 0; i <= T; ++i)
        t[i] = double(i) * dt; // Time points
        
    m = (double *)malloc(N * sizeof(double)); // Masses (all equal)
    x = (double *)malloc(ND * sizeof(double)); // Positions
    v = (double *)malloc(ND * sizeof(double)); // Velocities
    a = (double *)malloc(ND * sizeof(double)); // Accelerations
    x_next = (double *)malloc(ND * sizeof(double)); // Position updates
    v_next = (double *)malloc(ND * sizeof(double)); // Velocity updates

    Vec KE(T+1); // Kinetic energy
    initial_conditions(); // Set up initial conditions
    KE[0] = kinetic_energy(); // Calculate initial kinetic energy

    // Simulate the motion of N masses in D-dimensional space
    for (int n = 0; n < T; ++n) {
        timestep(); // Time step
        KE[n] = kinetic_energy(); // Kinetic energy
    }

    if (rank == 0) {
        // Reduce the kinetic energies
        MPI_Reduce(MPI_IN_PLACE, KE.data(), T+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // Protected save and print logic ...
        save(KE, "nbody_output/KE_" + std::to_string(N) + ".txt", "Kinetic Energy");
        save(t, "nbody_output/time_" + std::to_string(N) + ".txt", "Time");
        /*
        // Output the results
        std::cout << "Total Kinetic Energy = [" << KE[0];
        const int T_skip = T / 50; // Skip every T_skip time steps
        for (int n = 1; n <= T; n += T_skip)
            std::cout << ", " << KE[n];
        std::cout << "]" << std::endl;
        */
    } else {
        // Send the kinetic energies
        MPI_Reduce(KE.data(), NULL, T+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Free memory
    free(v_next);
    free(x_next);
    free(a);
    free(v);
    free(x);
    free(m);
    v_next = NULL;
    x_next = NULL;
    a = NULL;
    v = NULL;
    x = NULL;
    m = NULL;

    // Positions, velocities, and accelerations (reversed order)
    MPI_Win_free(&win_v_next);
    MPI_Win_free(&win_x_next);
    MPI_Win_free(&win_a);
    MPI_Win_free(&win_v);
    MPI_Win_free(&win_x);
    MPI_Win_free(&win_m);
    MPI_Finalize(); // Finalize MPI

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000.;
    // std::cout << "Runtime = " << elapsed << " s for N = " << N << std::endl;
    return 0;
}