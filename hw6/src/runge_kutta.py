def runge_kutta(f, t0, tf, h, y0, order, backend):
    '''
    Runge-Kutta method for solving ODEs.
    Parameters:
        f: function representing the ODE (dy/dt = f(t, y))
        t0: initial time
        tf: final time
        h: step size
        y0: initial value
        order: order of the Runge-Kutta method (1, 2, or 4)
        backend: Backend object containing the xp module (numpy or cupy) and other info
    Returns:
        t: array of time points
        y: array of solution values
        a: Runge-Kutta matrix
        b: Runge-Kutta weights
    '''
    
    # Initialize t and y
    n_steps = int((tf - t0) / h)
    t = backend.xp.linspace(t0, tf, n_steps+1)
    y = backend.xp.empty_like(t)
    y[0] = y0

    # Initialize RK parameters based on the order of the method
    match order:
        case 1: # forward Euler method
            a = backend.xp.array([[0.]])
            b = backend.xp.array([1.])
            c = backend.xp.array([0.])
        case 2:
            alpha = 1/2 # midpoint method
            a = backend.xp.array([[0, 0], 
                                [alpha, 0]])
            b = backend.xp.array([1 - 1/(2*alpha), 1/(2*alpha)])
            c = backend.xp.array([0, alpha])
        case 4:
            a = backend.xp.array([[0, 0, 0, 0], 
                                [1/2, 0, 0, 0], 
                                [0, 1/2, 0, 0], 
                                [0, 0, 1, 0]])
            b = backend.xp.array([1/6, 1/3, 1/3, 1/6])
            c = backend.xp.array([0, 1/2, 1/2, 1])
        case _:
            raise ValueError("Unsupported order. Use 1, 2, or 4.")
    k = backend.xp.empty_like(b)

    # Time-stepping loop
    for n in range(n_steps):
        for s, _ in enumerate(k):
            k[s] = f(t[n] + h * c[s], y[n] + h * backend.xp.dot(a[s, :], k))
        y[n+1] = y[n] + h * backend.xp.dot(b, k)

    return t, y, a, b

# Problem 1: Implement and verify the convergence of explicit solvers
# Problem 2: Stability regions of explicit methods
if __name__ == "__main__":
    import sys
    from backend import get_backend, sync, to_cpu, Timer

    prefer_gpu = True # Default
    if len(sys.argv) > 1: # Parse command line
        prefer_gpu = int(sys.argv[1])

    # Initialize backend (GPU if available and preferred, else CPU)
    backend = get_backend(prefer_gpu=prefer_gpu)  # or get_backend(prefer_gpu=False) for CPU-only

    # Initialize basic parameters
    t0 = 0.
    tf = 1.
    y0 = 1.
    f = lambda t, y: -y

    orders = [1, 2, 4]
    p = backend.xp.array([-4., -5., -6., -7., -8., -9., -10.])
    h = backend.xp.power(2, p)

    # Preallocate output arrays
    y1_errors = backend.xp.empty([len(p), 3]) # columns: RK1, RK2, RK4
    runtimes = backend.xp.empty_like(y1_errors)

    if backend.name == "cupy":
        # Warm up JIT compiler
        _, _, _, _ = runge_kutta(f=f, t0=t0, tf=tf, h=h[0], y0=y0, order=orders[0], backend=backend)

    # Time GPU/CPU operations
    for i, o_i in enumerate(orders):
        for j, h_j in enumerate(h):
            with Timer(backend) as timer:
                t, y, a, b = runge_kutta(f=f, t0=t0, tf=tf, h=h_j, y0=y0, order=o_i, backend=backend)
                sync(backend) # Synchronize before stopping the timer to ensure accurate measurement
            runtimes[j, i] = timer.dt
            y1_errors[j, i] = backend.xp.abs(y[-1] - backend.xp.exp(-tf))
            if h_j == h[-1]:
                match o_i:
                    case 1:
                        y_rk1 = y
                        a_rk1 = a
                        b_rk1 = b
                    case 2:
                        y_rk2 = y
                        a_rk2 = a
                        b_rk2 = b
                    case 4:
                        y_rk4 = y
                        a_rk4 = a
                        b_rk4 = b
            print(f"RK{o_i} with h = 2^{backend.xp.log2(h_j):.0f} completed in {runtimes[j, i]:.6f} s")

    # Convert GPU results back to CPU NumPy arrays
    cpu_t, cpu_y_rk1, cpu_a_rk1, cpu_b_rk1, cpu_y_rk2, cpu_a_rk2, cpu_b_rk2, cpu_y_rk4, cpu_a_rk4, cpu_b_rk4, cpu_h, cpu_y1_errors, cpu_runtimes = to_cpu(t), to_cpu(y_rk1), to_cpu(a_rk1), to_cpu(b_rk1), to_cpu(y_rk2), to_cpu(a_rk2), to_cpu(b_rk2), to_cpu(y_rk4), to_cpu(a_rk4), to_cpu(b_rk4), to_cpu(h), to_cpu(y1_errors), to_cpu(runtimes)

    # Save data
    with open(f"results/problem1_solutions_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# t y_rk1 y_rk2 y_rk4\n")
        for i in range(len(cpu_t)):
            file.write(f"{cpu_t[i]} {cpu_y_rk1[i]} {cpu_y_rk2[i]} {cpu_y_rk4[i]}\n")

    with open(f"results/problem1_errors_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# h end_error_rk1 end_error_rk2 end_error_rk4\n")
        for i in range(len(cpu_h)):
            file.write(f"{cpu_h[i]} {cpu_y1_errors[i, 0]} {cpu_y1_errors[i, 1]} {cpu_y1_errors[i, 2]}\n")

    with open(f"results/problem1_runtimes_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# h runtime_rk1 runtime_rk2 runtime_rk4\n")
        for i in range(len(cpu_h)):
            file.write(f"{cpu_h[i]} {cpu_runtimes[i, 0]} {cpu_runtimes[i, 1]} {cpu_runtimes[i, 2]}\n")

    with open(f"results/problem2_rk1_parameters_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# b A\n")
        for i in range(len(cpu_b_rk1)):
            file.write(f"{cpu_b_rk1[i]}")
            for j in range(len(cpu_b_rk1)):
                file.write(f" {cpu_a_rk1[i, j]}")
            file.write("\n")

    with open(f"results/problem2_rk2_parameters_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# b A\n")
        for i in range(len(cpu_b_rk2)):
            file.write(f"{cpu_b_rk2[i]}")
            for j in range(len(cpu_b_rk2)):
                file.write(f" {cpu_a_rk2[i, j]}")
            file.write("\n")

    with open(f"results/problem2_rk4_parameters_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# b A\n")
        for i in range(len(cpu_b_rk4)):
            file.write(f"{cpu_b_rk4[i]}")
            for j in range(len(cpu_b_rk4)):
                file.write(f" {cpu_a_rk4[i, j]}")
            file.write("\n")
