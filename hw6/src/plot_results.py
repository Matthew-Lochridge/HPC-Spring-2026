import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

def power_law_fit(h, y_error):
    '''
    Fit a power law of the form y_error = A * h^p to the data (h, y_error).
    Parameters:
        h: array of step sizes (powers of 2)
        y_error: array of error values
    Returns:
        fit: array of fitted error values based on the power law
        fit_uncertainty: array of uncertainties in the fitted error values
        A: coefficient A in the power law
        A_uncertainty: uncertainty in the coefficient A
        p: exponent p in the power law
        p_uncertainty: uncertainty in the exponent p
    '''
    coeffs, cov = np.polyfit(np.log2(h), np.log2(y_error), 1, cov=True)
    slope = coeffs[0]
    intercept = coeffs[1]
    p = slope
    A = 2 ** intercept
    fit = A * h ** p
    slope_uncertainty = np.sqrt(cov[0, 0])
    intercept_uncertainty = np.sqrt(cov[1, 1])
    p_uncertainty = slope_uncertainty
    A_uncertainty = np.abs(2 ** (intercept - 1) * intercept * intercept_uncertainty)
    fit_uncertainty = np.sqrt( (A_uncertainty * h ** p) ** 2 + (A * h ** (p-1) * p * p_uncertainty) ** 2 )
    return fit, fit_uncertainty, A, A_uncertainty, p, p_uncertainty

def rk_stability_contour(z, a, b):
    '''
    Computes the stability contour for a given Runge-Kutta method defined by tensors a and b.
    For Forward Euler, R = 1 + z.
    For RK2, R = 1 + z + 1/2 * z**2.
    For RK4, R = 1 + z + 1/2 * z**2 + 1/6 * z**3 + 1/24 * z**4.
    Parameters:
        z: complex number representing the scaled eigenvalue (h*lambda)
        a: Runge-Kutta matrix
        b: Runge-Kutta weights
    Returns:        
        R: stability function value at z, minus 1 (so that the contour where R=0 corresponds to the stability boundary)
    '''
    R = 1
    for s in range(len(b)):
        R = R + b @ np.linalg.matrix_power(a, s) @ np.ones_like(b) * z ** (s+1)
    return abs(R) - 1

if __name__ == "__main__":

    # Problem 1: Implement and verify the convergence of explicit solvers

    if os.path.exists(f'results/problem1_solutions_numpy.txt'):
        print(f"Found results/problem1_solutions_numpy.txt...")
        # Load and parse data
        solution_data = np.loadtxt(f'results/problem1_solutions_numpy.txt', delimiter=' ', skiprows=1)
        t = solution_data[:, 0]
        y_rk1 = solution_data[:, 1]
        y_rk2 = solution_data[:, 2]
        y_rk4 = solution_data[:, 3]
        # Sanity check: Plot the numerical solutions and the analytical solution
        plt.figure()
        plt.plot(t, np.exp(-t), label=r'$y(t)=e^{-t}$')
        plt.plot(t, y_rk1, '--', label='RK1 Solution')
        plt.plot(t, y_rk2, '-.', label='RK2 Solution')
        plt.plot(t, y_rk4, ':', label='RK4 Solution')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$y(t)$')
        plt.legend()
        plt.savefig(f"figures/problem1_solutions_numpy.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_solutions_numpy.pdf.")

    if os.path.exists(f'results/problem1_solutions_cupy.txt'):
        print(f"Found results/problem1_solutions_cupy.txt...")
        # Load and parse data
        solution_data = np.loadtxt(f'results/problem1_solutions_cupy.txt', delimiter=' ', skiprows=1)
        t = solution_data[:, 0]
        y_rk1 = solution_data[:, 1]
        y_rk2 = solution_data[:, 2]
        y_rk4 = solution_data[:, 3]
        # Sanity check: Plot the numerical solutions and the analytical solution
        plt.figure()
        plt.plot(t, np.exp(-t), label=r'$y(t)=e^{-t}$')
        plt.plot(t, y_rk1, '--', label='RK1 Solution')
        plt.plot(t, y_rk2, '-.', label='RK2 Solution')
        plt.plot(t, y_rk4, ':', label='RK4 Solution')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$y(t)$')
        plt.legend()
        plt.savefig(f"figures/problem1_solutions_cupy.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_solutions_cupy.pdf.")
    
    if os.path.exists(f'results/problem1_errors_numpy.txt'):
        print(f"Found results/problem1_errors_numpy.txt...")
        # Load and parse data
        convergence_data = np.loadtxt(f'results/problem1_errors_numpy.txt', delimiter=' ', skiprows=1)
        h = convergence_data[:, 0]
        error_rk1 = convergence_data[:, 1]
        error_rk2 = convergence_data[:, 2]
        error_rk4 = convergence_data[:, 3]
        # Fit power laws to the error data
        fit_rk1, fit_uncertainty_rk1, A_rk1, A_uncertainty_rk1, p_rk1, p_uncertainty_rk1 = power_law_fit(h, error_rk1)
        fit_rk2, fit_uncertainty_rk2, A_rk2, A_uncertainty_rk2, p_rk2, p_uncertainty_rk2 = power_law_fit(h, error_rk2)
        fit_rk4, fit_uncertainty_rk4, A_rk4, A_uncertainty_rk4, p_rk4, p_uncertainty_rk4 = power_law_fit(h, error_rk4)
        # Plot error vs step size
        plt.figure()
        plt.plot(h, error_rk1, 'ok', label='RK1 Error')
        plt.plot(h, fit_rk1, '-k', label=f'$| y(1) - e^{{{-1}}} | = ({A_rk1:.3f} \pm {A_uncertainty_rk1:.3f}) \Delta t^{{ ({p_rk1:.3f} \pm {p_uncertainty_rk1:.3f} )}} $')
        plt.plot(h, error_rk2, 'ob', label='RK2 Error')
        plt.plot(h, fit_rk2, '-b', label=f'$| y(1) - e^{{{-1}}} | = ({A_rk2:.3f} \pm {A_uncertainty_rk2:.3f}) \Delta t^{{ ({p_rk2:.3f} \pm {p_uncertainty_rk2:.3f} )}} $')
        plt.plot(h, error_rk4, 'or', label='RK4 Error')
        plt.plot(h, fit_rk4, '-r', label=f'$| y(1) - e^{{{-1}}} | = ({A_rk4:.3f} \pm {A_uncertainty_rk4:.3f}) \Delta t^{{ ({p_rk4:.2f} \pm {p_uncertainty_rk4:.2f} )}} $')
        plt.xlabel(r'$\Delta t$')
        plt.ylabel(r'$\left| y(1) - e^{-1} \right|$')
        ax = plt.gca()
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"figures/problem1_error_vs_h_numpy.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_error_vs_h_numpy.pdf.")
        # Convergence order table
        df = DataFrame({
            "Method": ["RK1", "RK2", "RK4"],
            "Convergence Order": [f"{p_rk1:.3f}", f"{p_rk2:.3f}", f"{p_rk4:.2f}"],
            "Notes": [f"Uncertainty = ±{p_uncertainty_rk1:.3f}", f"Uncertainty = ±{p_uncertainty_rk2:.3f}", f"Uncertainty = ±{p_uncertainty_rk4:.2f}"]
        })
        plt.figure()
        ax = plt.gca()
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        plt.savefig(f"figures/problem1_convergence_table_numpy.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_convergence_table_numpy.pdf.")
    
    if os.path.exists(f'results/problem1_errors_cupy.txt'):
        print(f"Found results/problem1_errors_cupy.txt...")
        # Load and parse data
        convergence_data = np.loadtxt(f'results/problem1_errors_cupy.txt', delimiter=' ', skiprows=1)
        h = convergence_data[:, 0]
        error_rk1 = convergence_data[:, 1]
        error_rk2 = convergence_data[:, 2]
        error_rk4 = convergence_data[:, 3]
        # Fit power laws to the error data
        fit_rk1, fit_uncertainty_rk1, A_rk1, A_uncertainty_rk1, p_rk1, p_uncertainty_rk1 = power_law_fit(h, error_rk1)
        fit_rk2, fit_uncertainty_rk2, A_rk2, A_uncertainty_rk2, p_rk2, p_uncertainty_rk2 = power_law_fit(h, error_rk2)
        fit_rk4, fit_uncertainty_rk4, A_rk4, A_uncertainty_rk4, p_rk4, p_uncertainty_rk4 = power_law_fit(h, error_rk4)
        # Plot error vs step size
        plt.figure()
        plt.plot(h, error_rk1, 'ok', label='RK1 Error')
        plt.plot(h, fit_rk1, '-k', label=f'$| y(1) - e^{{{-1}}} | = ({A_rk1:.3f} \pm {A_uncertainty_rk1:.3f}) \Delta t^{{ ({p_rk1:.3f} \pm {p_uncertainty_rk1:.3f} )}} $')
        plt.plot(h, error_rk2, 'ob', label='RK2 Error')
        plt.plot(h, fit_rk2, '-b', label=f'$| y(1) - e^{{{-1}}} | = ({A_rk2:.3f} \pm {A_uncertainty_rk2:.3f}) \Delta t^{{ ({p_rk2:.3f} \pm {p_uncertainty_rk2:.3f} )}} $')
        plt.plot(h, error_rk4, 'or', label='RK4 Error')
        plt.plot(h, fit_rk4, '-r', label=f'$| y(1) - e^{{{-1}}} | = ({A_rk4:.3f} \pm {A_uncertainty_rk4:.3f}) \Delta t^{{ ({p_rk4:.2f} \pm {p_uncertainty_rk4:.2f} )}} $')
        plt.xlabel(r'$\Delta t$')
        plt.ylabel(r'$\left| y(1) - e^{-1} \right|$')
        ax = plt.gca()
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"figures/problem1_error_vs_h_cupy.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_error_vs_h_cupy.pdf.")
        # Convergence order table
        df = DataFrame({
            "Method": ["RK1", "RK2", "RK4"],
            "Convergence Order": [f"{p_rk1:.3f}", f"{p_rk2:.3f}", f"{p_rk4:.2f}"],
            "Notes": [f"Uncertainty = ±{p_uncertainty_rk1:.3f}", f"Uncertainty = ±{p_uncertainty_rk2:.3f}", f"Uncertainty = ±{p_uncertainty_rk4:.2f}"]
        })
        plt.figure()
        ax = plt.gca()
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        plt.savefig(f"figures/problem1_convergence_table_cupy.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_convergence_table_cupy.pdf.")

    if os.path.exists(f'results/problem1_runtimes_numpy.txt') and os.path.exists(f'results/problem1_runtimes_cupy.txt'):
        print(f"Found results/problem1_runtimes_numpy.txt and results/problem1_runtimes_cupy.txt...")
        # Load and parse data
        runtime_data_cpu = np.loadtxt(f'results/problem1_runtimes_numpy.txt', delimiter=' ', skiprows=1)
        runtime_data_gpu = np.loadtxt(f'results/problem1_runtimes_cupy.txt', delimiter=' ', skiprows=1)
        h = runtime_data_cpu[:, 0]
        runtime_rk1_cpu = runtime_data_cpu[:, 1]
        runtime_rk2_cpu = runtime_data_cpu[:, 2]
        runtime_rk4_cpu = runtime_data_cpu[:, 3]
        runtime_rk1_gpu = runtime_data_gpu[:, 1]
        runtime_rk2_gpu = runtime_data_gpu[:, 2]
        runtime_rk4_gpu = runtime_data_gpu[:, 3]
        # CPU runtime vs step size
        plt.figure()
        plt.plot(h, runtime_rk1_cpu, label='RK1 CPU')
        plt.plot(h, runtime_rk2_cpu, label='RK2 CPU')
        plt.plot(h, runtime_rk4_cpu, label='RK4 CPU')
        plt.xlabel(r'$\Delta t$')
        plt.ylabel('Execution Time (s)')
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.legend()
        plt.savefig(f"figures/problem1_cpu_runtime_vs_h.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_cpu_runtime_vs_h.pdf.")
        # GPU runtime vs step size
        plt.figure()
        plt.plot(h, runtime_rk1_gpu, label='RK1 GPU')
        plt.plot(h, runtime_rk2_gpu, label='RK2 GPU')
        plt.plot(h, runtime_rk4_gpu, label='RK4 GPU')
        plt.xlabel(r'$\Delta t$')
        plt.ylabel('Execution Time (s)')
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.legend()
        plt.savefig(f"figures/problem1_gpu_runtime_vs_h.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_gpu_runtime_vs_h.pdf.")
        # Speedup vs step size
        speedup_rk1 = runtime_rk1_cpu / runtime_rk1_gpu
        speedup_rk2 = runtime_rk2_cpu / runtime_rk2_gpu
        speedup_rk4 = runtime_rk4_cpu / runtime_rk4_gpu
        plt.figure()
        plt.plot(h, speedup_rk1, label='RK1')
        plt.plot(h, speedup_rk2, label='RK2')
        plt.plot(h, speedup_rk4, label='RK4')
        plt.xlabel(r'$\Delta t$')
        plt.ylabel('Speedup (CPU Time / GPU Time)')
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        plt.legend()
        plt.savefig(f"figures/problem1_speedup_vs_h.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_speedup_vs_h.pdf.")
        # Speedup table
        df = DataFrame({
            "Method": ["RK1", "RK2", "RK4"],
            "CPU Time (s)": [f"{runtime_rk1_cpu[-1]:.6f}", f"{runtime_rk2_cpu[-1]:.6f}", f"{runtime_rk4_cpu[-1]:.6f}"],
            "GPU Time (s)": [f"{runtime_rk1_gpu[-1]:.6f}", f"{runtime_rk2_gpu[-1]:.6f}", f"{runtime_rk4_gpu[-1]:.6f}"],
            "Speedup (CPU/GPU)": [f"{speedup_rk1[-1]:.0f}", f"{speedup_rk2[-1]:.0f}", f"{speedup_rk4[-1]:.0f}"],
            "Notes": ["", "", ""]
        })
        plt.figure()
        ax = plt.gca()
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        plt.savefig(f"figures/problem1_speedup_table.pdf", bbox_inches="tight")
        print(f"Saved figures/problem1_speedup_table.pdf.")

    # Problem 2: Stability regions of explicit methods

    if os.path.exists(f'results/problem2_rk1_parameters_numpy.txt') and os.path.exists(f'results/problem2_rk2_parameters_numpy.txt') and os.path.exists(f'results/problem2_rk4_parameters_numpy.txt'):
        print(f"Found results/problem2_rk1_parameters_numpy.txt")
        # Load and parse data
        rk1_parameters = np.loadtxt(f'results/problem2_rk1_parameters_numpy.txt', delimiter=' ', skiprows=1)
        b_rk1 = np.array([rk1_parameters[0]])
        a_rk1 = np.array([rk1_parameters[1:]])
        print(f"Found results/problem2_rk2_parameters_numpy.txt")
        # Load and parse data
        rk2_parameters = np.loadtxt(f'results/problem2_rk2_parameters_numpy.txt', delimiter=' ', skiprows=1)
        b_rk2 = rk2_parameters[:, 0]
        a_rk2 = rk2_parameters[:, 1:]
        print(f"Found results/problem2_rk4_parameters_numpy.txt")
        # Load and parse data
        rk4_parameters = np.loadtxt(f'results/problem2_rk4_parameters_numpy.txt', delimiter=' ', skiprows=1)
        b_rk4 = rk4_parameters[:, 0]
        a_rk4 = rk4_parameters[:, 1:]
        # Plot stability contours
        x = np.linspace(-5, 5, 100)
        x_grid, y_grid = np.meshgrid(x, x)
        R_rk1 = rk_stability_contour(z=x_grid + y_grid*1j, a=a_rk1, b=b_rk1)
        R_rk2 = rk_stability_contour(z=x_grid + y_grid*1j, a=a_rk2, b=b_rk2)
        R_rk4 = rk_stability_contour(z=x_grid + y_grid*1j, a=a_rk4, b=b_rk4)
        plt.figure()
        contour_rk1 = plt.contour(x_grid, y_grid, R_rk1, levels=[0], linestyles='solid', colors='black')
        contour_rk2 = plt.contour(x_grid, y_grid, R_rk2, levels=[0], linestyles='solid', colors='red')
        contour_rk4 = plt.contour(x_grid, y_grid, R_rk4, levels=[0], linestyles='solid', colors='blue')
        plt.grid()
        plt.axis('equal')
        plt.xlabel(r'Re($\lambda \Delta t$)')
        plt.ylabel(r'Im($\lambda \Delta t$)')
        id_rk1, _ = contour_rk1.legend_elements()
        id_rk2, _ = contour_rk2.legend_elements()
        id_rk4, _ = contour_rk4.legend_elements()
        plt.legend([id_rk1[0], id_rk2[0], id_rk4[0]], ["RK1", "RK2", "RK4"])
        plt.savefig(f"figures/problem2_stability_contours_numpy.pdf", bbox_inches="tight")
        print(f"Saved figures/problem2_stability_contours_numpy.pdf.")

    if os.path.exists(f'results/problem2_rk1_parameters_cupy.txt') and os.path.exists(f'results/problem2_rk2_parameters_cupy.txt') and os.path.exists(f'results/problem2_rk4_parameters_cupy.txt'):
        print(f"Found results/problem2_rk1_parameters_cupy.txt")
        # Load and parse data
        rk1_parameters = np.loadtxt(f'results/problem2_rk1_parameters_cupy.txt', delimiter=' ', skiprows=1)
        b_rk1 = np.array([rk1_parameters[0]])
        a_rk1 = np.array([rk1_parameters[1:]])
        print(f"Found results/problem2_rk2_parameters_numpy.txt")
        # Load and parse data
        rk2_parameters = np.loadtxt(f'results/problem2_rk2_parameters_cupy.txt', delimiter=' ', skiprows=1)
        b_rk2 = rk2_parameters[:, 0]
        a_rk2 = rk2_parameters[:, 1:]
        print(f"Found results/problem2_rk4_parameters_numpy.txt")
        # Load and parse data
        rk4_parameters = np.loadtxt(f'results/problem2_rk4_parameters_cupy.txt', delimiter=' ', skiprows=1)
        b_rk4 = rk4_parameters[:, 0]
        a_rk4 = rk4_parameters[:, 1:]
        # Plot stability contours
        x = np.linspace(-5, 5, 100)
        x_grid, y_grid = np.meshgrid(x, x)
        R_rk1 = rk_stability_contour(z=x_grid + y_grid*1j, a=a_rk1, b=b_rk1)
        R_rk2 = rk_stability_contour(z=x_grid + y_grid*1j, a=a_rk2, b=b_rk2)
        R_rk4 = rk_stability_contour(z=x_grid + y_grid*1j, a=a_rk4, b=b_rk4)
        plt.figure()
        contour_rk1 = plt.contour(x_grid, y_grid, R_rk1, levels=[0], linestyles='solid', colors='black')
        contour_rk2 = plt.contour(x_grid, y_grid, R_rk2, levels=[0], linestyles='solid', colors='red')
        contour_rk4 = plt.contour(x_grid, y_grid, R_rk4, levels=[0], linestyles='solid', colors='blue')
        plt.grid()
        plt.axis('equal')
        plt.xlabel(r'Re($\lambda \Delta t$)')
        plt.ylabel(r'Im($\lambda \Delta t$)')
        id_rk1, _ = contour_rk1.legend_elements()
        id_rk2, _ = contour_rk2.legend_elements()
        id_rk4, _ = contour_rk4.legend_elements()
        plt.legend([id_rk1[0], id_rk2[0], id_rk4[0]], ["RK1", "RK2", "RK4"])
        plt.savefig(f"figures/problem2_stability_contours_cupy.pdf", bbox_inches="tight")
        print(f"Saved figures/problem2_stability_contours_cupy.pdf.")
    
    # Problem 3: GPU speedup via large-batch ensemble integration

    if os.path.exists(f'results/problem3_runtimes_numpy.txt') and os.path.exists(f'results/problem3_runtimes_cupy.txt'):
        print(f"Found results/problem3_runtimes_numpy.txt and results/problem3_runtimes_cupy.txt...")
        # Load and parse data
        runtime_data_cpu = np.loadtxt(f'results/problem3_runtimes_numpy.txt', delimiter=' ', skiprows=1)
        runtime_data_gpu = np.loadtxt(f'results/problem3_runtimes_cupy.txt', delimiter=' ', skiprows=1)
        N = runtime_data_cpu[:, 0]
        runtime_cpu = runtime_data_cpu[:, 1]
        runtime_gpu = runtime_data_gpu[:, 1]
        # Runtime vs batch size
        plt.figure()
        plt.plot(N, runtime_cpu, label='CPU')
        plt.plot(N, runtime_gpu, label='GPU')
        plt.xlabel(r'$N$')
        plt.ylabel('Execution Time (s)')
        plt.xscale('log', base=10)
        plt.yscale('log', base=10)
        plt.legend()
        plt.savefig(f"figures/problem3_runtime_vs_N.pdf", bbox_inches="tight")
        print(f"Saved figures/problem3_runtime_vs_N.pdf.")
        # Speedup vs batch size
        plt.figure()
        speedup = runtime_cpu / runtime_gpu
        plt.plot(N, speedup)
        plt.xlabel(r'$N$')
        plt.ylabel('Speedup (CPU Time / GPU Time)')
        plt.xscale('log', base=10)
        plt.yscale('log', base=10)
        plt.savefig(f"figures/problem3_speedup_vs_N.pdf", bbox_inches="tight")
        print(f"Saved figures/problem3_speedup_vs_N.pdf.")
        # Throughput (trajectories x steps / sec) vs N
        throughput_cpu = N * 1e4 / runtime_cpu
        throughput_gpu = N * 1e4 / runtime_gpu
        plt.figure()
        plt.plot(N, throughput_cpu, label='CPU')
        plt.plot(N, throughput_gpu, label='GPU')
        plt.xlabel(r'$N$')
        plt.ylabel('Throughput (1/s)')
        plt.xscale('log', base=10)
        plt.yscale('log', base=10)
        plt.legend()
        plt.savefig(f"figures/problem3_throughput_vs_N.pdf", bbox_inches="tight")
        print(f"Saved figures/problem3_throughput_vs_N.pdf.")
    
    # Problem 4: Stiff decay modes (TR vs. TRBDF2)

    # Task 2: Demonstrate stiffness by showing that explicit methods would require ∆t to be small enough to resolve αmax for stability. 
    # Show that too large of a ∆t leads to instability with a plot. 
    if os.path.exists(f'results/problem4_task2_numpy.txt'):
        print(f"Found results/problem4_task2_numpy.txt...")
        # Load and parse data
        stability_data = np.loadtxt("results/problem4_task2_empirical_numpy.txt", delimiter=' ', skiprows=1)
        alpha = stability_data[:, 0]
        yf_RK1 = stability_data[:, 1]
        yf_RK2 = stability_data[:, 2]
        yf_RK4 = stability_data[:, 3]
        yf_exact = stability_data[:, 4]
        # Explicit RK stability, step size vs. alpha
        plt.figure()
        plt.plot(alpha, 2./alpha, label='RK1, RK2')
        plt.plot(alpha, 2.79/alpha, label='RK4')
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\Delta t$")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.legend()
        plt.savefig(f"figures/problem4_task2_theoretical.pdf")
        print(f"Saved figures/problem4_task2_theoretical.pdf.")
        # Global error vs alpha, with h = 2^-3
        error_RK1 = np.abs(yf_RK1 - yf_exact)
        error_RK2 = np.abs(yf_RK2 - yf_exact)
        error_RK4 = np.abs(yf_RK4 - yf_exact)
        plt.figure()
        plt.plot(alpha, error_RK1, label='RK1')
        plt.plot(alpha, error_RK2, label='RK2')
        plt.plot(alpha, error_RK4, label='RK4')
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$|y(1) - e^{-\alpha}|$")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.legend()
        plt.savefig(f"figures/problem4_task2_empirical.pdf")
        print(f"Saved figures/problem4_task2_empirical.pdf.")

    # Task 3: Demonstrate lack of L-stability for TR on very stiff modes, 
    # i.e. for a large ∆t, plot a small, medium, and large α and show that TR does not strongly damp the stiff decay modes compared to TRBDF2.
    if os.path.exists(f'results/problem4_task3_numpy.txt'):
        print(f"Found results/problem4_task3_numpy.txt...")
        # Load and parse data
        stability_data = np.loadtxt("results/problem4_task3_numpy.txt", delimiter=' ', skiprows=1)
        alpha = stability_data[:, 0]
        yf_TR = stability_data[:, 1]
        yf_TRBDF2 = stability_data[:, 2]
        yf_exact = stability_data[:, 3]
        # Global error vs alpha, with h = 1/2
        error_TR = np.abs(yf_TR - yf_exact)
        error_TRBDF2 = np.abs(yf_TRBDF2 - yf_exact)
        plt.figure()
        plt.plot(alpha, error_TR, label='TR')
        plt.plot(alpha, error_TRBDF2, label='TRBDF2')
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$|y(1) - e^{-\alpha}|$")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.legend()
        plt.savefig(f"figures/problem4_task3_numpy.pdf")
        print(f"Saved figures/problem4_task3_numpy.pdf.")

    # Task 4: GPU benchmark: set d large enough to exploit GPU parallelism (e.g. d = 1e6 in float32). 
    # Compare NumPy vs. CuPy runtimes for TRBDF2 specifically, and explain why this case is GPU-friendly.

    if os.path.exists(f"results/problem4_task4_numpy.txt") and os.path.exists(f"results/problem4_task4_cupy.txt"):
        print(f"Found results/problem4_task4_numpy.txt...")
        # Load and parse data
        cpu_data = np.loadtxt("results/problem4_task4_numpy.txt", delimiter=' ', skiprows=1)
        h = cpu_data[:, 0]
        runtime_cpu = cpu_data[:, 1]
        print(f"Found results/problem4_task4_cupy.txt...")
        # Load and parse data
        gpu_data = np.loadtxt("results/problem4_task4_cupy.txt", delimiter=' ', skiprows=1)
        h = gpu_data[:, 0]
        runtime_gpu = gpu_data[:, 1]
        # Runtime vs h
        plt.figure()
        plt.plot(h, runtime_cpu, label='NumPy')
        plt.plot(h, runtime_gpu, label='CuPy')
        plt.xlabel(r"$\Delta t$")
        plt.ylabel("Runtime (s)")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.legend()
        plt.savefig(f"figures/problem4_task4.pdf")
        print(f"Saved figures/problem4_task4.pdf.")

    # Task 5: Produce a summary table discussing insights from each method, 
    # e.g., TR vs. TRBDF2, large vs. small ∆t, accuracy (e.g. max relative error at t = 1 vs. exact e−αit), runtime, and the notes on stability.

        df = DataFrame({
            "Method": [r"TR with large $\Delta t$", r"TR with small $\Delta t$", r"TRBDF2 with large $\Delta t$", r"TRBDF2 with small $\Delta t$"],
            "CPU Time (s)": [f"", f"", f"{runtime_cpu[0]:.6f}", f"{runtime_cpu[-1]:.6f}"],
            "GPU Time (s)": [f"", f"", f"{runtime_gpu[0]:.6f}", f"{runtime_gpu[-1]:.6f}"],
            "Speedup": [f"", f"", f"{runtime_cpu[0]/runtime_gpu[0]:.6f}", f"{runtime_cpu[-1]/runtime_gpu[-1]:.6f}"],
            "Max. Error": [f"{error_TR[-1]}", f"", f"{error_TRBDF2[-1]}", f""],
            "Notes": [r"Not stable for large $\alpha$", r"Stable for large $\alpha$", r"Stable for large $\alpha$", r"Stable for large $\alpha$"]
        })
        plt.figure()
        ax = plt.gca()
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        plt.savefig(f"figures/problem4_task5.pdf", bbox_inches="tight")
        print(f"Saved figures/problem4_task5.pdf.")