def rk4_batch(f, t0, tf, h, y0, backend):
    '''
    Fourth-order vectorized Runge-Kutta method for solving batches of independent ODEs.
    Parameters:
        f: function representing the ODE (dy/dt = f(t, y))
        t0: initial time
        tf: final time
        h: step size
        y0: array of initial values
        backend: Backend object containing the xp module (numpy or cupy) and other info
    Returns:
        y: array of solution values at tf
    '''
    
    # Initialize t and y
    n_steps = int((tf - t0) / h)
    t = backend.xp.linspace(t0, tf, n_steps+1)
    y = y0

    # Initialize RK4 parameters
    a = backend.xp.array([[0, 0, 0, 0], 
                        [1/2, 0, 0, 0], 
                        [0, 1/2, 0, 0], 
                        [0, 0, 1, 0]])
    b = backend.xp.array([1/6, 1/3, 1/3, 1/6])
    c = backend.xp.array([0, 1/2, 1/2, 1])

    k = backend.xp.empty((len(b), len(y)), dtype=y0.dtype)
    k_next = backend.xp.empty_like(k)

    # Time-stepping loop
    for n in range(n_steps):
        for s, _ in enumerate(k):
            k_next[s, :] = f(t[n] + h * c[s], y + h * backend.xp.einsum('i,ij->j', a[s, :], k))
        k = k_next
        y = y + h * backend.xp.einsum('i,ij->j', b, k)

    return y

if __name__ == "__main__":
    import sys
    from backend import get_backend, sync, to_cpu, Timer

    prefer_gpu = True # Default
    if len(sys.argv) > 1: # Parse command line
        prefer_gpu = int(sys.argv[1])

    # Initialize backend (GPU if available and preferred, else CPU)
    backend = get_backend(prefer_gpu=prefer_gpu)  # or get_backend(prefer_gpu=False) for CPU-only

    # Problem 3: GPU speedup via large-batch ensemble integration
    t0 = 0
    tf = 10
    h = 1e-3
    r = 2.
    f = lambda t, y: r * y * (1 - y)

    p = backend.xp.array([3., 4., 5., 6., 7.])
    N = (backend.xp.power(10, p)).astype(int)

    runtime_per_N = backend.xp.empty(len(N))

    if backend.name == "cupy":
        # Warm up JIT compiler
        y0 = backend.xp.linspace(0.01, 0.99, N[0].item())
        _ = rk4_batch(f=f, t0=t0, tf=tf, h=h, y0=y0, backend=backend)

    # Time GPU/CPU operations
    for i, N_i in enumerate(N):
        with Timer(backend) as timer:
            print(f"Timing RK4 using {backend.name} with N = {N_i:.0e}...")
            y0 = backend.xp.linspace(0.01, 0.99, N_i.item())
            y = rk4_batch(f=f, t0=t0, tf=tf, h=h, y0=y0, backend=backend)
            sync(backend) # Synchronize before stopping the timer to ensure accurate measurement
        runtime_per_N[i] = timer.dt

    # Convert GPU results back to CPU NumPy arrays
    cpu_N, cpu_runtime_per_N = to_cpu(N), to_cpu(runtime_per_N)

    # Save data
    with open(f"results/problem3_runtimes_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# N, runtime (s)\n")
        for i in range(len(cpu_N)):
            file.write(f"{cpu_N[i]} {cpu_runtime_per_N[i]}\n")
