# Interval Signal Temporal Logic for Robust Optimal Control
# Author: Luke Baird
# (c) Georgia Institute of Technology 2024

import numpy as np
import interval
import scipy
import blimp
import time

from stlpy.systems import LinearSystem
from stlpy.STL import LinearPredicate
from stlpy.solvers import GurobiIntervalOptimalControl
from matplotlib import pyplot as plt
from matplotlib import rc

# One function from another repo, inclusion.
from numpy import clip, empty, inf
def d_positive (B, separate=True) :
    Bp = clip(B, 0, inf); Bn = clip(B, -inf, 0)
    if separate :
        return Bp, Bn
    else :
        n,m = B.shape
        ret = empty((2*n,2*m))
        ret[:n,:m] = Bp; ret[n:,m:] = Bp
        ret[:n,m:] = Bn; ret[n:,:m] = Bn
        return ret


plt.rcParams['pdf.fonttype'] = 42 # Avoid type 3 font issue.
plt.rcParams['ps.fonttype'] = 42

def p_error(msg):
    print(f'\033[91m[ERROR]\033[00m {msg}')

def create_phi_no_uncertain_predicates(dT, N, Tinv):
    # Create the STL formula phi defined in my notebook.
    
    x_a = np.zeros((1, 12)); x_a[0, 6] = 1; x_a = x_a @ Tinv
    y_a = np.zeros((1, 12)); y_a[0, 7] = 1; y_a = y_a @ Tinv

    # Create the target set.
    # The target set is a 0.5 x 0.5 box located at (5,5), uncertain size!
    x_lb_target = LinearPredicate(x_a, 4.7)
    x_ub_target = LinearPredicate(-x_a, -5.3)
    y_lb_target = LinearPredicate(y_a, 4.7)
    y_ub_target = LinearPredicate(-y_a, -5.3)
    target_set = x_lb_target & x_ub_target & y_lb_target & y_ub_target

    # Create the negation
    x_lb_target_not = LinearPredicate(x_a, 4.7).negation()
    x_ub_target_not = LinearPredicate(-x_a, -5.3).negation()
    y_lb_target_not = LinearPredicate(y_a, 4.7).negation()
    y_ub_target_not = LinearPredicate(-y_a, -5.3).negation()
    target_set_not = x_lb_target_not | x_ub_target_not | y_lb_target_not | y_ub_target_not

    always_clause = target_set.always(0, int(1.5/dT)).eventually(0, int(6/dT))
    eventually_clause = (target_set_not | target_set_not.eventually(0, int(3/dT))).always(0, int(6.5/dT))
    phi_A = always_clause & eventually_clause
    # OK try something infeasible
    # phi_A = (target_set & target_set_not).always(0, int(5/dT)) # lol

    # phi_A = (target_set.always(0, int(1/dT)) & target_set_not.eventually(int(2/dT), int(3/dT))).eventually(0, int(5/dT))

    # Create the safe return set.
    # The safe return set is a 0.7 x 0.7 box located at (0,0)
    x_lb_return = LinearPredicate(-x_a, -0.35)
    x_ub_return = LinearPredicate(x_a, -0.35)
    y_lb_return = LinearPredicate(-y_a, -0.35)
    y_ub_return = LinearPredicate(y_a, -0.35)

    phi_B = (x_lb_return & x_ub_return & y_lb_return & y_ub_return).eventually(int(19/dT), N)

    xpy_a = np.zeros((1, 12)); xpy_a[0, 6] = 2; xpy_a[0, 7] = 1; xpy_a = xpy_a @ Tinv
    xmy_a = np.zeros((1, 12)); xmy_a[0, 6] = 2; xmy_a[0, 7] = -1; xmy_a = xmy_a @ Tinv

    # Charging Station 1.
    # The vertices of the triangle are: (0.6, 3.6), (1.4, 3.6), (1, 4.4)
    station1_side1 = LinearPredicate(y_a, 3.6)
    station1_side2 = LinearPredicate(xmy_a, -2.4)
    station1_side3 = LinearPredicate(-xpy_a, -6.4)
    station_1_spec = station1_side1 & station1_side2 & station1_side3

    # Charging Station 2.
    # The vertices of the triangle are: (3.9, 2.7), (4.7, 2.7), (4.3, 3.5)
    station2_side1 = LinearPredicate(y_a, 2.7)
    station2_side2 = LinearPredicate(xmy_a, 5.1)
    station2_side3 = LinearPredicate(-xpy_a, -12.1)
    station_2_spec = station2_side1 & station2_side2 & station2_side3

    # Charging Station 3.
    x_lb_station3 = LinearPredicate(x_a, 2.7)
    x_ub_station3 = LinearPredicate(-x_a, -3.3)
    y_lb_station3 = LinearPredicate(y_a, 0.7)
    y_ub_station3 = LinearPredicate(-y_a, -1.3)
    station_3_spec = x_lb_station3 & x_ub_station3 & y_lb_station3 & y_ub_station3
    # (station_1_spec | station_2_spec | station_3_spec)
    # phi_D = station_1_spec.eventually(0, int(10/dT)).always(0, N - int(10/dT)) # for now: change timing?
    phi_D_frequency = int(8./dT)
    phi_D = (station_1_spec | station_2_spec | station_3_spec).eventually(
            0, phi_D_frequency
        ).always(0, N - phi_D_frequency)
    
    phi_D = (station_1_spec | station_2_spec | station_3_spec).eventually(
        int(1/dT), int(6/dT) 
    ) & (station_1_spec | station_2_spec | station_3_spec).eventually(int(8/dT),int(12/dT)) & (station_1_spec | station_2_spec | station_3_spec).eventually(int(14/dT),int(20/dT))
    # Every six seconds, we msut return to phi_D

    return phi_A & phi_B & phi_D


    

def create_phi(dT, N, Tinv):
    # Create the STL formula phi defined in my notebook.
    
    x_a = np.zeros((1, 12), dtype=np.interval); x_a[0, 6] = 1; x_a = x_a @ Tinv
    y_a = np.zeros((1, 12), dtype=np.interval); y_a[0, 7] = 1; y_a = y_a @ Tinv

    # Create the target set.
    # The target set is a 0.5 x 0.5 box located at (5,5), uncertain size!
    x_lb_target = LinearPredicate(x_a, 4.7 + np.interval(-0.1, 0.1))
    x_ub_target = LinearPredicate(-x_a, -5.3 + np.interval(-0.1, 0.1))
    y_lb_target = LinearPredicate(y_a, 4.7 + np.interval(-0.1, 0.1))
    y_ub_target = LinearPredicate(-y_a, -5.3 + np.interval(-0.1, 0.1))
    target_set = x_lb_target & x_ub_target & y_lb_target & y_ub_target

    # Create the negation
    x_lb_target_not = LinearPredicate(x_a, 4.7 + np.interval(-0.1, 0.1)).negation()
    x_ub_target_not = LinearPredicate(-x_a, -5.3 + np.interval(-0.1, 0.1)).negation()
    y_lb_target_not = LinearPredicate(y_a, 4.7 + np.interval(-0.1, 0.1)).negation()
    y_ub_target_not = LinearPredicate(-y_a, -5.3 + np.interval(-0.1, 0.1)).negation()
    target_set_not = x_lb_target_not | x_ub_target_not | y_lb_target_not | y_ub_target_not

    always_clause = target_set.always(0, int(1.5/dT)).eventually(0, int(6/dT))
    eventually_clause = (target_set_not | target_set_not.eventually(0, int(3/dT))).always(0, int(6.5/dT))
    phi_A = always_clause & eventually_clause
    # OK try something infeasible
    # phi_A = (target_set & target_set_not).always(0, int(5/dT)) # lol

    # phi_A = (target_set.always(0, int(1/dT)) & target_set_not.eventually(int(2/dT), int(3/dT))).eventually(0, int(5/dT))

    # Create the safe return set.
    # The safe return set is a 0.7 x 0.7 box located at (0,0)
    x_lb_return = LinearPredicate(-x_a, -0.35)
    x_ub_return = LinearPredicate(x_a, -0.35)
    y_lb_return = LinearPredicate(-y_a, -0.35)
    y_ub_return = LinearPredicate(y_a, -0.35)

    phi_B = (x_lb_return & x_ub_return & y_lb_return & y_ub_return).eventually(int(19/dT), N)

    xpy_a = np.zeros((1, 12)); xpy_a[0, 6] = 2; xpy_a[0, 7] = 1; xpy_a = xpy_a @ Tinv
    xmy_a = np.zeros((1, 12)); xmy_a[0, 6] = 2; xmy_a[0, 7] = -1; xmy_a = xmy_a @ Tinv

    # Charging Station 1.
    # The vertices of the triangle are: (0.6, 3.6), (1.4, 3.6), (1, 4.4)
    station1_side1 = LinearPredicate(y_a, 3.6)
    station1_side2 = LinearPredicate(xmy_a, -2.4)
    station1_side3 = LinearPredicate(-xpy_a, -6.4)
    station_1_spec = station1_side1 & station1_side2 & station1_side3

    # Charging Station 2.
    # The vertices of the triangle are: (3.9, 2.7), (4.7, 2.7), (4.3, 3.5)
    station2_side1 = LinearPredicate(y_a, 2.7)
    station2_side2 = LinearPredicate(xmy_a, 5.1)
    station2_side3 = LinearPredicate(-xpy_a, -12.1)
    station_2_spec = station2_side1 & station2_side2 & station2_side3

    # Charging Station 3.
    x_lb_station3 = LinearPredicate(x_a, 2.7)
    x_ub_station3 = LinearPredicate(-x_a, -3.3)
    y_lb_station3 = LinearPredicate(y_a, 0.7)
    y_ub_station3 = LinearPredicate(-y_a, -1.3)
    station_3_spec = x_lb_station3 & x_ub_station3 & y_lb_station3 & y_ub_station3
    # (station_1_spec | station_2_spec | station_3_spec)
    # phi_D = station_1_spec.eventually(0, int(10/dT)).always(0, N - int(10/dT)) # for now: change timing?
    phi_D_frequency = int(8./dT)
    phi_D = (station_1_spec | station_2_spec | station_3_spec).eventually(
            0, phi_D_frequency
        ).always(0, N - phi_D_frequency)
    
    phi_D = (station_1_spec | station_2_spec | station_3_spec).eventually(
        int(1/dT), int(6/dT) 
    ) & (station_1_spec | station_2_spec | station_3_spec).eventually(int(8/dT),int(12/dT)) & (station_1_spec | station_2_spec | station_3_spec).eventually(int(14/dT),int(20/dT))
    # Every six seconds, we msut return to phi_D

    # return phi_D
    return phi_A & phi_B & phi_D

def plot_phi(ax, labels=False):
    # Plot the target set, safe return set, and obstacle set.
    # If labels, then label the sets with $\mathcal{A}$, $\mathcal{B}$, and $\mathcal{O}$, respectively.
    # The target set is a 0.5 x 0.5 box located at (5,5)
    # Fill between the preceding two rectangles.
    ax.fill_between([4.6, 5.4], 4.6, 5.4, color='b', alpha=0.3)
    ax.fill_between([4.7, 5.3], 4.7, 5.3, color='w')
    ax.add_patch(plt.Rectangle((4.7, 4.7), 0.6, 0.6, fill=False, color='b', linewidth=1))
    ax.add_patch(plt.Rectangle((4.6, 4.6), 0.8, 0.8, fill=False, color='b', linewidth=1))

    ax.add_patch(plt.Polygon([(0.6, 3.6), (1.4, 3.6), (1, 4.4)], fill=False, color='m', linewidth=2))
    ax.add_patch(plt.Polygon([(3.9, 2.7), (4.7, 2.7), (4.3, 3.5)], fill=False, color='m', linewidth=2))

    # The safe return set is a 0.7 x 0.7 box located at (0,0)
    ax.add_patch(plt.Rectangle((-0.35, -0.35), 0.7, 0.7, fill=False, color='g', linewidth=2))
    # The obstacle to avoid is a 0.2 x 0.2 box located at (2,2)
    # ax.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, fill=False, color='r', linewidth=2))

    # Create two squares for charging stations.
    # ax.add_patch(plt.Rectangle((0.7, 3.7), 0.6, 0.6, fill=False, color='m', linewidth=2))
    # ax.add_patch(plt.Rectangle((4.0, 2.7), 0.6, 0.6, fill=False, color='m', linewidth=2))
    ax.add_patch(plt.Rectangle((2.7, 0.7), 0.6, 0.6, fill=False, color='m', linewidth=2))

    if labels:
        ax.text(5-.17, 5-.12, '$\\mathcal{A}$', fontsize=12, color='b')
        ax.text(-0.14, -.12, '$\\mathcal{B}$', fontsize=12, color='g')
        # ax.text(2-.13, 2-.12, '$\\mathcal{O}$', fontsize=12, color='r')
        ax.text(1-.14, 4-.22, '$\\mathcal{C}$', fontsize=12, color='m')
        ax.text(4.3-.14, 3-.12, '$\\mathcal{C}$', fontsize=12, color='m')
        ax.text(3-.13, 1-.12, '$\\mathcal{C}$', fontsize=12, color='m')

        # Add blimp-screenshot.png above the green box.
        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # arr_img = plt.imread('blimp-screenshot.png', format='png')
        # imagebox = OffsetImage(arr_img, zoom=0.2)
        # ab = AnnotationBbox(imagebox, (0.1, 1.0), frameon=False)
        # ax.add_artist(ab)


def plot_mission_setup():
    # Create a plot of the mission setup. This is a main function.
    fig, ax = plt.subplots()
    plot_phi(ax, True)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.set_title('Mission setup')

    # Set the figure height
    fig.set_figheight(4)

    # Make the figure a square
    ax.set_aspect('equal', 'box')
    
    plt.savefig('output/cdc/mission_setup.pdf')
    plt.show()

def main(do_w=True, do_ip=True, detailed=False):
    rc('text', usetex=True)

    dT = None
    blimp_analysis = None
    if detailed:
        dT = .25
        blimp_analysis = scipy.io.loadmat('K25.mat')
    else:
        dT = 0.5
        blimp_analysis = scipy.io.loadmat('K50.mat')

    # We are using a blimp model with 12 states and 4 inputs.

    hrz = 20 # seconds
    N = int(hrz/dT) # Number of time steps.

    # Create u as a 4x1 interval vector.
    u = np.zeros((4, 1), dtype=np.interval)
    u[:] = np.interval(-1, 1) #* 0.6 # Try 0.6 or 1.0.

    # Create the system from blimp.py
    the_blimp = blimp.Blimp()
    the_blimp.setup_discrete_lti_model(dT)
    A_nom = the_blimp.A_discrete
    B = the_blimp.B_discrete

    # Create a matrix G that maps disturbance in R^6 to a vector in R^12.
    # This is a 12x6 matrix, an identity and a zero matrix concatenated.
    G = np.zeros((12, 6))
    G[:6, :6] = np.eye(6)
    C = np.eye(12) # Nominal system dynamics output (unused)
    D = np.zeros((12, 4))

    # Get T and K
    T = np.real(blimp_analysis['T'])#; T = np.eye(12) + np.random.rand(12, 12) * .01
    K = np.real(blimp_analysis['K'])

    the_blimp.dT = dT
    the_blimp.stable_physics_model()

    A_K = A_nom - B @ K

    # Check if T is invertible.
    if np.linalg.det(T) == 0:
        p_error('T is not invertible.')
        return

    T_inv = np.linalg.inv(T)
    G_xf = T @ G
    A_xf = T @ A_nom @ T_inv
    A_K_xf = T @ A_K @ T_inv
    B_xf = T @ B
    C_xf = C @ T_inv
    D_xf = T @ D

    Ae = d_positive(A_K_xf, False)
    Be = np.vstack((B_xf, B_xf)) # in R^{24x4}
    Ce = d_positive(C_xf, False) # in R^{24x24}
    De = np.vstack((D_xf, D_xf)) # in R^{24x4}

    # Create the embedding system.
    embedding = LinearSystem(Ae, Be, Ce, De)
    # nominal = LinearSystem(A_K,B,C,D)
    nominal = LinearSystem(A_xf, B_xf, C_xf, D_xf)

    # Check the eigenvalues of Ae.
    print('Eigenvalues of Ae')
    print(np.linalg.eigvals(Ae))
    print(np.linalg.eigvals(A_xf))

    # Create the STL formula.
    phi = None
    if do_ip:
        phi = create_phi(dT, N, np.eye(12))#np.eye(12)) # T_inv
    else:
        phi = create_phi_no_uncertain_predicates(dT, N, np.eye(12))

    # N is computed from the horizon.
    print('Error dimensions')
    print(f'p: {embedding.p}')
    print(f'n: {embedding.n}')
    print(f'm: {embedding.m}')

    print('Nominal system dimensions')
    print(f'p: {nominal.p}')
    print(f'n: {nominal.n}')
    print(f'm: {nominal.m}')


    # Create the additive disturbance interval.
    # This is in R^{24x1} ultimately, but the salient part is in R^6.
    w = np.zeros((6, 1), dtype=np.interval)
    if do_w: # otherwise make it zero.
        w[:] = np.interval(-0.001, 0.001) * dT * .2

    w = G_xf @ w # Use the advantages of interval arithmetic.
    w_l, w_u = interval.get_lu(w)
    w = np.vstack((w_l, w_u))


    # Define x0
    x0 = np.zeros((12, 1), dtype=np.interval)
    x0_nominal = np.zeros((12, 1)) # For now.
    x0_nominal[8,0] = -1.5 # m
    
    # Transform the coordinates.
    x0 = T @ x0
    x0_nominal = T @ x0_nominal

    x0_l, x0_u = interval.get_lu(x0)
    x0 = np.vstack((x0_l, x0_u))

    T_inv_d_positive = d_positive(T_inv, False)

    # Construct state bounds, for the vertical position. It must lie in [-1.5, 0].
    state_bounds = np.zeros((12, 1), dtype=np.interval)
    # state_bounds[8, 0] = np.interval(-2.0, 0)
    state_bounds[8,0] = np.interval(-2.5, 0.)
    # Make the rest [-np.inf, np.inf]
    state_bounds[0:8, 0] = np.interval(-np.inf, np.inf)
    state_bounds[9:, 0] = np.interval(-np.inf, np.inf)
    state_bounds_l, state_bounds_u = interval.get_lu(state_bounds)

    # phi, psi
    solver=None
    if do_w:
        print('embedding system construction - do_w = True')
        solver = GurobiIntervalOptimalControl(spec=(phi), sys=embedding, sys_nominal=nominal, x0=x0,
                                        x0_nominal=x0_nominal, T=N, horizon=N, K=K, verbose=False,
                                        M=100, w=w, intervals=True, V_inv=T_inv, V_inv_d=T_inv_d_positive)#T_inv)
    elif do_ip and not do_w:
        print('embedding system construction - do_ip = True, do_w = False')
        solver = GurobiIntervalOptimalControl(spec=(phi), sys=nominal, sys_nominal=None, x0=x0_nominal,
                                        x0_nominal=None, T=N, horizon=N, K=K, verbose=False,
                                        M=100, w=None, intervals=True, V_inv=T_inv, V_inv_d=T_inv_d_positive)#T_inv)
    else:
        solver = GurobiIntervalOptimalControl(spec=(phi), sys=nominal, sys_nominal=None, x0=x0_nominal, x0_nominal=None, T=N, horizon=N, K=K, verbose=False,
                                          M=100, w=None, intervals=False, V_inv=T_inv) # Set intervals to True if create_phi is used.
    
    # Add bounds to the state.
    # solver.AddStateBounds(state_bounds_l, state_bounds_u) # Keep the blimp from flying away.

    # Create a cost function for the solver.
    R = np.eye(4) 
    # Let Q be a block matrix, of a 3x3 identity in the upper-left corner with the rest zero.
    Q = np.zeros((12,12))
    Q[:3, :3] = np.eye(3)
    solver.AddQuadraticCost(Q=Q, R=R) # Prefer small velocities and control inputs.

    # Create control actuation constraints.
    u_min, u_max = interval.get_lu(u)
    solver.AddControlBounds(u_min, u_max)
    
    # Solve the optimal control problem.
    if False: # Make this true for the comparison.
        for j in range(10):
            solver.RemoveX0Constraints()

            # Perturb x0_nominal elements 6,7.
            x0_nominal = np.zeros((12, 1)) # For now.
            x0_nominal[8,0] = -1.5 # m
            x0_nominal[6,0] = np.random.uniform(-0.1, 0.1)
            x0_nominal[7,0] = np.random.uniform(-0.1, 0.1)
            x0_nominal = T @ x0_nominal
            print(f'The new starting location is {x0_nominal[6,0]}, {x0_nominal[7,0]} (try #{j}).')
            if do_w:
                solver.SetX0Constraint(x0, x0_nominal)
            else:
                solver.SetX0Constraint(x0_nominal, None)
            current_time = time.time()
            solution = solver.Solve()
            print(f'Time to solve: {time.time() - current_time} seconds.')
    else: # Just solve the problem.
        solution = solver.Solve()

    eta = solution[0]
    u_xf = solution[1]
    eta_nominal = solution[2]
    rho = solution[3]
    print(rho)
    cost = solution[5]

    print(f'eta shape: {eta.shape}')
    print(f'u_xf shape: {u_xf.shape}')
    print(f'Cost: {cost}')

    from pathlib import Path
    Path('output/cdc').mkdir(parents=True, exist_ok=True)

    # Save the results.
    np.save(f'output/cdc/eta{dT}{do_ip}{do_w}.npy', eta)
    np.save(f'output/cdc/u_xf{dT}{do_ip}{do_w}.npy', u_xf)
    np.save(f'output/cdc/eta_nominal{dT}{do_ip}{do_w}.npy', eta_nominal)
    np.save(f'output/cdc/rho{dT}{do_ip}{do_w}.npy', rho)
    np.save(f'output/cdc/cost{dT}{do_ip}{do_w}.npy', cost)

    if do_w:
        # Do this properly.
        T_inv_p, T_inv_n = d_positive(T_inv)
        x_lb = T_inv_p @ eta[:12, :] + T_inv_n @ eta[12:, :]
        x_ub = T_inv_p @ eta[12:, :] + T_inv_n @ eta[:12, :]

        x_nom = T_inv @ eta_nominal # nominal trajectory
        x_lower = x_lb + x_nom # lower bound on the trajectory
        x_upper = x_ub + x_nom # upper bound on the trajectory

        # Check the robustness of phi.
        print(f'robustness of lower bound: {phi.robustness(x_lower, 0)}')
        print(f'robustness of upper bound: {phi.robustness(x_upper, 0)}')
        print(f'robustness of nominal tra: {phi.robustness(x_nom, 0)}')

        plt.figure()

        # Get the current axis handle.
        ax = plt.gca()
        plot_phi(ax) # We do this early to avoid accidentally overwriting the plot.

        plt.plot(x_nom[6,:], x_nom[7,:], 'k.-')

        # Plot a sequence of boxes that represent the interval bounds.
        # The corners are given by x_lower[6,:], x_lower[7,:], x_upper[6,:], x_upper[7,:].
        # Only plot every 5th box.
        for i in range(0, N+1):
            plt.fill([x_lower[6,i], x_upper[6,i], x_upper[6,i], x_lower[6,i], x_lower[6,i]],
                    [x_lower[7,i], x_lower[7,i], x_upper[7,i], x_upper[7,i], x_lower[7,i]], 'm-', alpha=0.1)
            plt.plot([x_lower[6,i], x_upper[6,i], x_upper[6,i], x_lower[6,i], x_lower[6,i]],
                    [x_lower[7,i], x_lower[7,i], x_upper[7,i], x_upper[7,i], x_lower[7,i]], 'm-', alpha=0.3)
        
        # Add a green box around the 25th and 34th boxes.
        box_7_idx = 24 # Analyze the plots manually and fill this in based on that.
        box_15_idx = 34
        plt.fill([x_lower[6,box_7_idx], x_upper[6,box_7_idx], x_upper[6,box_7_idx], x_lower[6,box_7_idx], x_lower[6,box_7_idx]],
                    [x_lower[7,box_7_idx], x_lower[7,box_7_idx], x_upper[7,box_7_idx], x_upper[7,box_7_idx], x_lower[7,box_7_idx]], 'g', alpha=0.7)
        plt.fill([x_lower[6,box_15_idx], x_upper[6,box_15_idx], x_upper[6,box_15_idx], x_lower[6,box_15_idx], x_lower[6,box_15_idx]],
                    [x_lower[7,box_15_idx], x_lower[7,box_15_idx], x_upper[7,box_15_idx], x_upper[7,box_15_idx], x_lower[7,box_15_idx]], 'g', alpha=0.4)

        plt.xlabel('$x$ (m)')
        plt.ylabel('$y$ (m)')
        
        plt.title('Blimp horizontal plane trajectory with interval bounds')
        plt.savefig(f'output/cdc/interval_bounds{dT}{do_ip}{do_w}.pdf')


        # Plot the results.
        plt.figure()
        # Plot the interval bounds with respect to time, and fill in the space in between with lines
        plt.fill_between(np.arange(N+1) * dT, x_lb[6,:], x_ub[6,:], alpha=0.3)
        plt.plot(np.arange(N+1) * dT, x_lb[6, :], 'b.')
        plt.plot(np.arange(N+1) * dT, x_ub[6, :], 'r.') # upper bound
        plt.xlabel('t (s)')
        plt.ylabel('x (m)')
        plt.title('Error dynamics in the original coordinates')
        # plt.show()

        # Repeat the above but use eta.
        plt.figure()
        plt.fill_between(np.arange(N+1) * dT, eta[6, :], eta[18, :], alpha=0.3)
        plt.plot(np.arange(N+1) * dT, eta[6, :], 'b.')
        plt.plot(np.arange(N+1) * dT, eta[18, :], 'r.') # Upper bound
        plt.xlabel('t (s)')
        plt.ylabel('$\\eta$ (m)')
        plt.title('Error dynamics in the transformed coordinate space')
        # plt.show()

        # OK: now plot x vs. y
        # Make two subplots. The second has input vs time.
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x_lb[6,:], x_lb[7,:], 'b.-')
        plt.plot(x_ub[6,:], x_ub[7,:], 'r.-')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(N+1) * dT, np.linalg.norm(u_xf, axis=0), 'g.-') # Plot the first input.
        plt.xlabel('t (s)')
        plt.ylabel('|u| ($\\frac{m}{s^2}$)')



        plt.figure()
        plt.plot(np.arange(N+1) * dT, x_lower[8,:], 'c.-')
        plt.plot(np.arange(N+1) * dT, x_upper[8,:], 'm.-')
        plt.xlabel('t (s)')
        plt.ylabel('z (m)')
        plt.title('z vs. t')

        plt.show()
    else:
        x = T_inv @ eta
        print(f'robustness of nominal trajectory: {phi.robustness(x, 0)}')
        # print(f'robustness of nominal trajectory: {phi.robustness(eta, 0)}')
        plt.figure()
        ax = plt.gca()
        plot_phi(ax)
        plt.plot(x[6,:], x[7,:], 'k.-') # The solved trajectory.
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('$(x,y)$ Blimp trajectory (no disturbance)')
        plt.savefig(f'output/cdc/trajectory_no_disturbance{dT}{do_ip}{do_w}.pdf')
        plt.show()

if __name__ == '__main__':
    # plot_mission_setup()
    main(do_w=True, do_ip=True, detailed=True) # do disturbance, do interval predicates, detailed=True is 0.25s time step while False is 0.5s time step.