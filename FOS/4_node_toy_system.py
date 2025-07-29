import numpy as np
import matplotlib.pyplot as plt
import time 
def solve_fractional_system(alpha, A, x0, t, mode = 'single', K=None):

    n_nodes = len(alpha)
    n_points = len(t)
    control_step = 1000 # timepoint for control input (single)
    x_setpoint = 1.0 # what value we want to drive our nodes toward
    # x_setpoint = np.array([1.0, 2.0, 3.0, 4.0]) if we want to drive towards diff values for each node
    x = np.zeros((n_points, n_nodes))
    x[0, :] = x0 # Set initial conditions
    h = t[1] - t[0] # Time step 1/500
    # coefficients for frac derivative (Grunwald-Letnikov)
    coeffs = {}
    for a_val in alpha: 
        coeffs[a_val] = [1.0] 

        for k in range(1, n_points):
            coeffs[a_val].append(coeffs[a_val][k-1] * (1 - (a_val + 1) / k))

    for i in range(1, n_points): 
        for j in range(n_nodes): 
            # coupling effect from other nodes 
            linear_term = A[j, :] @ x[i - 1, :]

            control_input = 0.0
            if K is not None: 
                if mode == 'continuous':
                    error = x[i - 1, :] - x_setpoint
                    control_input = -K[j, :] @ error
                elif mode == 'single_ctrl' and i == control_step: 
                    error = x[i - 1, :] - x_setpoint
                    control_input = -K[j, :] @ error
            
            rhs = -x[i - 1, j] + linear_term + control_input

            # Memory term using prev states
            memory_term = 0.0
            for k in range(1, i + 1):
                memory_term += coeffs[alpha[j]][k] * x[i - k, j]
            # Update state 
            x[i, j] = rhs * (h**alpha[j]) - memory_term

    return x

# two with low to 0 memory 
alpha = np.array([1.0, 0.95, 0.7, 0.4])
K = np.diag([10, 10, 10, 10])
n_nodes = len(alpha)

# A matrix with unstable dynamics 
A = np.array([[0.1, 0.05, 0.0, 0.0],
              [0.05, 0.1, 0.0, 0.0],
              [0.0, 0.0, 0.2, 0.05],
              [0.0, 0.0, 0.05, 0.2]])

A_convergent = np.array([[-0.5, 0.1, 0.0, 0.0],
                         [0.1, -0.5, 0.0, 0.0],
                         [0.0, 0.0, -0.3, 0.05],
                         [0.0, 0.0, 0.05, -0.3]])

A_oscillatory = np.array([[0.5, 1.0, 0.0, 0.0],
                          [-1.0, 0.5, 0.0, 0.0], 
                          [0.0, 0.0, 0.5, 0.8],
                          [0.0, 0.0, -0.8, 0.5]])

A_divergent = np.array([[1.0, 0.5, 0.0, 0.0], 
                        [0.5, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.2, 0.6],
                        [0.0, 0.0, 0.6, 1.2]])

# A = np.zeros((4, 4))
# A matrix to isolate memory effects of alpha 
# A = np.diag([0.3, 0.3, 0.3, 0.3])

# Initial conditions for each node (random values between 0 and 1)
# np.random.seed(42)
# x0 = np.random.rand(n_nodes) * 2
x0 = np.ones(4)
t_start = 0.0
t_end = 8
frequency = 500.0 
n_points = int((t_end - t_start) * frequency) + 1
t = np.linspace(t_start, t_end, n_points)

A_matrix_choice = A_oscillatory #change this based on what A matrix to test
print(f"Simulation parameters:")
print(f"  Initial conditions (x0): {x0}")
print(f"  A matrix:\n{A_matrix_choice}")

x_unctrl = solve_fractional_system(alpha, A_matrix_choice, x0, t)
# single control input 
# x_ctrled_single = solve_fractional_system(alpha, A_matrix_choice, x0, t, 'single', K)
start = time.time()
# continuous control input 
# x_ctrled_continuous = solve_fractional_system(alpha, A_matrix_choice, x0, t, 'single_ctrl', K)
end = time.time()

print(end-start, "seconds")

plt.figure(figsize=(10, 6)) # Single figure for all uncontrolled signals 
for j in range(n_nodes):
    plt.plot(t, x_unctrl[:, j], label=f'Node {j} (α = {alpha[j]})')
plt.title('Uncontrolled System Dynamics')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.ylim(-1, 1.2) # change 
plt.legend()
plt.grid(True)
plt.show()

# for j in range(n_nodes):
#     plt.figure()
#     plt.plot(t, x_unctrl[:, j], label='Uncontrolled')
#     plt.plot(t, x_ctrled_continuous[:, j], label='Controlled')
#     plt.title(f'Node {j} (α = {alpha[j]})')
#     plt.xlabel('Time (s)')
#     plt.ylabel('State')
#     plt.ylim(-.7, 1.2)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"images/single_controlled_alpha-{alpha[j]}.png")
# plt.show()

