def RK(f, t0, tf, h, y0, alpha, order, backend):
    '''
    Vectorized Runge-Kutta method for solving batches of independent ODEs.
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

    k = backend.xp.empty((len(b), len(y)), dtype=y0.dtype)
    k_next = backend.xp.empty_like(k)

    # Time-stepping loop
    for n in range(n_steps):
        for s, _ in enumerate(k):
            k_next[s, :] = f(t[n] + h * c[s], y + h * backend.xp.einsum('i,ij->j', a[s, :], k), alpha)
        k = k_next
        y = y + h * backend.xp.einsum('i,ij->j', b, k)

    return y

def TR(f, t0, tf, h, y0, alpha, bdf2, backend):
    '''
    Trapezoidal rule for numerical integration, with optional BDF2 correction.
    Parameters:
        f: function to integrate
        t0: initial time
        tf: final time
        h: step size
        y0: array of initial values
        alpha: array of ODE parameters
        bdf2: Boolean parameter indicating whether to implement BDF2 correction using gamma = 2 - sqrt(2)
        backend: Backend object containing the xp module (numpy or cupy) and other info
    Returns:
        y: array of solution values at tf
    '''
    n_steps = int((tf - t0) / h)
    y = y0

    gamma = 1.
    if bdf2 == True:
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

    # Task 2: Demonstrate stiffness by showing that explicit methods would require ∆t to be small enough to resolve αmax for stability. 
    # Show that too large of a ∆t leads to instability with a plot. 

    t0 = 0.
    tf = 1.
    f = lambda t, y, alpha: -alpha * y
    
    d = int(7)
    alpha = backend.xp.logspace(0, 6, d)
    y0 = backend.xp.ones_like(alpha)
    yf_exact = backend.xp.exp(-alpha*tf)

    h = 0.5
    n_steps = int((tf - t0) / h)
    t = backend.xp.linspace(t0, tf, n_steps+1)

    yf_RK1= RK(f=f, t0=t0, tf=tf, h=h, y0=y0, alpha=alpha, order=1, backend=backend)
    sync(backend)

    yf_RK2= RK(f=f, t0=t0, tf=tf, h=h, y0=y0, alpha=alpha, order=2, backend=backend)
    sync(backend)

    yf_RK4= RK(f=f, t0=t0, tf=tf, h=h, y0=y0, alpha=alpha, order=4, backend=backend)
    sync(backend)

    cpu_alpha, cpu_yf_RK1, cpu_yf_RK2, cpu_yf_RK4, cpu_yf_exact = to_cpu(alpha), to_cpu(yf_RK1), to_cpu(yf_RK2), to_cpu(yf_RK4), to_cpu(yf_exact)

    with open(f"results/problem4_task2_empirical_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# alpha yf_RK1 yf_RK2 yf_RK4 yf_exact\n")
        for i in range(len(cpu_alpha)):
            file.write(f"{cpu_alpha[i]} {cpu_yf_RK1[i]} {cpu_yf_RK2[i]} {cpu_yf_RK4[i]} {cpu_yf_exact[i]}\n")

    # Task 3: Demonstrate lack of L-stability for TR on very stiff modes, 
    # i.e. for a large ∆t, plot a small, medium, and large α and show that TR does not strongly damp the stiff decay modes compared to TRBDF2.

    h = 0.5
    n_steps = int((tf - t0) / h)
    t = backend.xp.linspace(t0, tf, n_steps+1)

    yf_TR = TR(f=f, t0=t0, tf=tf, h=h, y0=y0, alpha=alpha, bdf2=False, backend=backend)
    sync(backend)

    yf_TRBDF2 = TR(f=f, t0=t0, tf=tf, h=h, y0=y0, alpha=alpha, bdf2=True, backend=backend)
    sync(backend)

    cpu_alpha, cpu_yf_TR, cpu_yf_TRBDF2, cpu_yf_exact = to_cpu(alpha), to_cpu(yf_TR), to_cpu(yf_TRBDF2), to_cpu(yf_exact)

    with open(f"results/problem4_task3_{backend.name}.txt", "w", encoding="utf-8") as file:
        file.write("# alpha yf_TR yf_TRBDF2 yf_exact\n")
        for i in range(len(cpu_alpha)):
            file.write(f"{cpu_alpha[i]} {cpu_yf_TR[i]} {cpu_yf_TRBDF2[i]} {cpu_yf_exact[i]}\n")