import numpy as np
import matplotlib.pyplot as plt

############################################
# 1. Your existing functions
############################################

def load_array(n_people, p):
    if p == 1:
        p = "1.0"
    arrs = np.load(f"/Users/josephpieper/MCM Competition/StepData/Arrays{p}__{n_people}.npy")

    arr = []
    for val in arrs:
        arr.append(val[0])
        arr.append(val[1])
        arr.append(val[2])
    return arr

data_dict = {}
for p in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    for n in [1, 2, 3, 4, 5, 10]:
        data_dict[n, p] = load_array(n, p)

def stitch_dataset(combs):
    arr = []
    for n, p in combs:
        for i in data_dict[n, p]:
            arr.append(i)
    return arr

def laplace(f):
    f_padded = np.pad(f, 1, mode='edge')  # Add 1-pixel border
    return (f_padded[2:, 1:-1] +
            f_padded[:-2, 1:-1] +
            f_padded[1:-1, 2:] +
            f_padded[1:-1, :-2] -
            4 * f_padded[1:-1, 1:-1])

def build_dataset(p_up_down, size, p_t):
    i_s = 0
    i = 0
    arr = []
    while i < size:
        p_step = p_t[i_s]
        i_s = (i_s + 1) % len(p_t)
        if np.random.uniform() < p_step ** 2:
            a = data_dict[2, p_up_down]
            idx = np.random.randint(0, len(a))
            arr.append(a[idx])
            i += 1
        elif np.random.uniform() < p_step:
            a = data_dict[1, p_up_down]
            idx = np.random.randint(0, len(a))
            arr.append(a[idx])
            i += 1
    return arr

def forward_model(theta, dt, T, arrs, nx=50, ny=50,
                  mat_params=[0.5, 1e7, 1e-6, 1e-4, 1e-7,
                              5e5, 1e5, 3e5, 7e5]):
    # theta = [p, ud_rate]
    p, ud_rate = theta
    # Unpack mat_params
    k, gamma_val, C, D, diff_const, sig_u, sig_s, sig_min, sig_max = mat_params
    
    dx = 1 / nx
    x = np.linspace(0, 1, nx+1)
    y = np.linspace(0, 1, ny+1)
    X_grid, Y_grid = np.meshgrid(x, y, indexing='ij')
    
    # Initialize
    h_sim = np.zeros_like(X_grid)
    sigma = np.random.normal(sig_u, sig_s, size=X_grid.shape)
    sigma = np.clip(sigma, sig_min, sig_max)

    num_steps = int(T / dt)
    
    for i in range(num_steps):
        # Here, we just pick arrs[i]; ignoring p and ud_rate
        if np.random.uniform() < ud_rate:
            P = arrs[i]
        else:
            P = arrs[i]

        # Plastic deformation update
        X_ = np.maximum(P - sigma, 0)
        h_impulse = (-k * X_) * dt / gamma_val
        h_diffusion = diff_const * laplace(h_sim) * dt / dx**2
        
        h_sim += h_impulse + h_diffusion
        
        # Yield stress update
        sigma = sigma * (1 - C * dt) - D * dt * P
    
    return h_sim

def plot_results(result):
    x = np.linspace(0, 0.4, result.shape[0])
    y = np.linspace(0, 1, result.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig = plt.figure(figsize=(20, 10))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=20, azim=240)
    surf1 = ax1.plot_surface(X, Y, result, cmap='viridis', rstride=1, cstride=1)
    ax1.set_title('Plastic Deformation')
    ax1.set_xlabel('Y position')
    ax1.set_ylabel('X position')
    ax1.set_zlabel('Deformation')
    ax1.set_zlim(bottom=-0.5, top=result.max())
    ax1.set_box_aspect((1, 1, 1))
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(122)
    cont = ax2.contourf(X, Y, result, cmap='viridis', levels=20)
    ax2.set_title('Top View')
    ax2.set_xlabel('Y position')
    ax2.set_ylabel('X position')
    ax2.set_aspect('equal')
    plt.colorbar(cont, ax=ax2, shrink=0.8)

    plt.show()


############################################
# 2. Example base settings
############################################

base_mat_params = [
    1e-7,  # k
    1.0,   # gamma_val
    1e-6,  # C
    1e-4,  # D
    1e-6,  # diff_const
    4e6,   # sig_u
    1e5,   # sig_s
    1e5,  # sig_min
    1e10    # sig_max
]

# Build one default dataset (small size to test quickly)
Dd = build_dataset(0.4, 10000, [1]*4)
res = forward_model([1, 0], dt=1, T=len(Dd), arrs=Dd,
                    nx=39, ny=99, mat_params=base_mat_params)
print("Test SSE:", np.sum(np.square(res)))


############################################
# 3. Parameter sweeps, but skipping large sizes
############################################
p_up_down     = [0.2, 0.4, 0.6, 0.8, 1.0]

# Let's remove 1e6 and 1e7 here to avoid huge computations:
sizes         = [1, 1e1, 1e2, 1e3, 1e4, 1e5]  

k_values      = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
C_values      = [1e-4, 1e-5, 1e-6, 1e-7]
D_values      = [1e-1, 1e-2, 1e-3, 1e-4]
diff_values   = [1e-3, 1e-4, 1e-5, 1e-6]
gamma_values  = [1]        
sig_u_values  = [1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8]      
sig_s_values  = [1e5]      
sig_min_vals  = [1e5]      
sig_max_vals  = [1e10]      

results_dict = {}

###############################################################################
# A) Sweep p_up_down
###############################################################################
p_up_down_sse = {}
for val in p_up_down:
    # Rebuild dataset for each p_up_down
    Dd_temp = build_dataset(val, 10000, [1]*4)  
    res = forward_model([1, 0], dt=1, T=len(Dd_temp),
                        arrs=Dd_temp, nx=39, ny=99,
                        mat_params=base_mat_params)
    sse = np.sum(np.square(res))
    p_up_down_sse[val] = sse

results_dict['p_up_down'] = p_up_down_sse
print("p_up_down sweep:", p_up_down_sse)

###############################################################################
# B) Sweep sizes, skip or slice if too big
###############################################################################
size_sse = {}
max_sim_steps = 50000  # limit number of steps in forward_model
for val in sizes:
    size_int = int(val)
    if size_int > max_sim_steps:
        # Option 1: skip
        print(f"Skipping size={size_int} because it's too large.")
        continue
        
        # OR Option 2: (example) slice the dataset
        # Dd_temp = build_dataset(0.4, size_int, [1]*4)
        # Dd_temp = Dd_temp[:max_sim_steps]
        # res = forward_model([1, 0], dt=1, T=max_sim_steps,
        #                     arrs=Dd_temp, nx=39, ny=99,
        #                     mat_params=base_mat_params)
        # sse = np.sum(np.square(res))
        # size_sse[size_int] = sse

    else:
        # Build full dataset
        Dd_temp = build_dataset(0.4, size_int, [1]*4)
        res = forward_model([1, 0], dt=1, T=len(Dd_temp),
                            arrs=Dd_temp, nx=39, ny=99,
                            mat_params=base_mat_params)
        sse = np.sum(np.square(res))
        size_sse[size_int] = sse

results_dict['size'] = size_sse
print("Size sweep:", size_sse)

###############################################################################
# C) Sweep k-values; we can reuse the default dataset Dd
###############################################################################
k_sse = {}
for k_val in k_values:
    mat_params_new = base_mat_params.copy()
    mat_params_new[0] = k_val  # index 0 => 'k'

    res = forward_model([1, 0], dt=1, T=len(Dd),
                        arrs=Dd, nx=39, ny=99,
                        mat_params=mat_params_new)
    sse = np.sum(np.square(res))
    k_sse[k_val] = sse

results_dict['k'] = k_sse
print("k sweep:", k_sse)

###############################################################################
# D) Sweep C-values similarly
###############################################################################
C_sse = {}
for c_val in C_values:
    mat_params_new = base_mat_params.copy()
    mat_params_new[2] = c_val  # index 2 => 'C'

    res = forward_model([1, 0], dt=1, T=len(Dd),
                        arrs=Dd, nx=39, ny=99,
                        mat_params=mat_params_new)
    sse = np.sum(np.square(res))
    C_sse[c_val] = sse

results_dict['C'] = C_sse
print("C sweep:", C_sse)

# ... similarly for D, diff_const, etc. if desired

print("All results:", results_dict)

###############################################################################
# 4. Plot the results for each parameter
###############################################################################
def plot_sweeps(results_dict):
    """
    Create a separate plot for each parameter in results_dict.
    The x-axis = parameter values, y-axis = sum of squares (SSE).
    """
    for param_name, param_results in results_dict.items():
        # param_results = {param_value: sse, ...}
        if not param_results:
            print(f"No data to plot for {param_name}. Skipping.")
            continue
        
        # Sort by the parameter value (key)
        x_vals, y_vals = zip(*sorted(param_results.items(), key=lambda x: x[0]))
        
        # Create new figure
        plt.figure(figsize=(6,4))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-')
        
        plt.xlabel(param_name)
        plt.ylabel('Sum of Squares (SSE)')
        plt.title(f'SSE vs {param_name}')
        
        # If needed, you could use a log scale on x:
        # plt.xscale('log')
        
        plt.tight_layout()
        plt.show()

# Finally, call the plotting function
plot_sweeps(results_dict)
