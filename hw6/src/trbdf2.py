def TR(f, t0, tf, h, y0, alpha, backend):
    '''
    Trapezoidal rule for numerical integration, with optional BDF2 correction.
    Parameters:
        f: function to integrate
        t0: initial time
        tf: final time
        h: step size
        y0: array of initial values
        alpha: array of ODE parameters
        backend: Backend object containing the xp module (numpy or cupy) and other info
    Returns:
        y: array of solution values at tf
    '''
    n_steps = int((tf - t0) / h)
    y = y0

    gamma = 2. - 2. ** 0.5

    for _ in range(n_steps):
        G = ((2. - gamma*alpha*h)/(2. + gamma*alpha*h) - (1. - gamma)**2. ) / (gamma*(2. - gamma) + (1. - gamma)*gamma*alpha*h)
        y = G * y

    return y

if __name__ == "__main__":
    import sys
    from backend import get_backend, sync, to_cpu, Timer

    prefer_gpu = True # Default
    if len(sys.argv) > 1: # Parse command line
        prefer_gpu = int(sys.argv[1])

    # Initialize backend (GPU if available and preferred, else CPU)
    backend = get_backend(prefer_gpu=prefer_gpu)  # or get_backend(prefer_gpu=False) for CPU-only

    # Problem 4: Stiff decay modes (TR vs. TRBDF2)

    # Task 4: GPU benchmark: set d large enough to exploit GPU parallelism (e.g. d = 1e6 in float32). 
    # Compare NumPy vs. CuPy runtimes for TRBDF2 specifically, and explain why this case is GPU-friendly.

    t0 = 0.
    tf = 1.

    p = backend.xp.array([-1., -2., -3., -4., -5., -6.])
    h = 2. * backend.xp.power(10., p)
    
    d = int(1e6)
    alpha = backend.xp.logspace(0, 6, d)
    y0 = backend.xp.ones_like(alpha)

    f = lambda t, y, alpha: -alpha * y

    runtime = backend.xp.empty_like(h)

    # Time GPU/CPU operations
    for i, h_i in enumerate(h):
        with Timer(backend) as timer:
            print(f"Running TRBDF2 on {backend.name} with h = {h_i:e}...")
            y_TRBDF2 = TR(f=f, t0=t0, tf=tf, h=h_i, y0=y0, alpha=alpha, backend=backend)
            sync(backend) # Synchronize before stopping the timer to ensure accurate measurement
        runtime[i] = timer.dt

    # Convert GPU results back to CPU NumPy arrays
    cpu_h, cpu_runtime = to_cpu(h), to_cpu(runtime)

    with open(f"results/problem4_task4_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# h runtime (s)\n")
        for i in len(cpu_h):
            file.write(f"{cpu_h[i]} {cpu_runtime[i]}\n")

    # Produce a summary table discussing insights from each method, 
    # e.g., TR vs. TRBDF2, large vs. small ∆t, accuracy (e.g. max relative error at t = 1 vs. exact e−αit), runtime, and the notes on stability.
