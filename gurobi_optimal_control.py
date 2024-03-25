from ..base import STLSolver
from ...STL import LinearPredicate, NonlinearPredicate
import numpy as np
from numpy import clip, empty, inf

import gurobipy as gp
from gurobipy import GRB, min_ #, QuadExpr

from interval import get_lu

import time

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

class GurobiIntervalOptimalControl(STLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`,
    solve the optimization problem

    .. math::

        \min & -\\rho^{\\varphi}(y_0,y_1,\dots,y_T) + \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = A x_t + B u_t

        & y_{t} = C x_t + D u_t

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    with Gurobi using mixed-integer convex programming. This gives a globally optimal
    solution, but may be computationally expensive for long and complex specifications.
    
    .. note::

        This class implements the algorithm described in

        Belta C, et al.
        *Formal methods for control synthesis: an optimization perspective*.
        Annual Review of Control, Robotics, and Autonomous Systems, 2019.
        https://dx.doi.org/10.1146/annurev-control-053018-023717.

    :param spec:            A tuple of :class:`.STLFormula` describing the specification, logically conjoined.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param model:           A :class:`Model` describing the system dynamics as an LTV system.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    :param presolve:        (optional) A boolean indicating whether to use Gurobi's
                            presolve routines. Default is ``True``.
    :param verbose:         (optional) A boolean indicating whether to print detailed
                            solver info. Default is ``True``.
    """

    def __init__(self, spec, sys, sys_nominal, x0, x0_nominal, T, M=1000, robustness_cost=False, 
            hard_constraint=True, presolve=True, verbose=True, K=None, horizon=0, rho_min=0.0, N=1, tube_mpc_buffer=None,
            tube_mpc_enabled=False, w=None, intervals=False, V_inv=None, V_inv_d=None):
        assert M > 0, "M should be a (large) positive scalar"
        if type(spec) is not tuple:
            spec = (spec,)
        super().__init__(spec, sys, x0, T, verbose)
        self.sys_nominal = sys_nominal
        self.intervals=intervals
        self.horizon = horizon # useful for our specific problem

        self.M = float(M)
        self.presolve = presolve

        # Create a variable representing the number of future steps to require rho>0 on.
        self.N = N
        self.tubeMPCBuffer=tube_mpc_buffer
        self.tubeMPCEnabled=tube_mpc_enabled

        if K is not None:
            self.K = K

        # Create a gurobi model to govern the optimization problem.
        self.model = gp.Model("STL_MICP")
        
        # Initialize the cost function, which will added to self.model right before solving
        self.cost = 0.0

        # Initialize a place to hold constraints
        self.dynamics_constraints = []
        self.lp_constraints = []
        self.x0_constraints = []

        self.initialization_point = None

        # Dummy start point - it's not that useful...
        #self.start_point = 2*self.horizon
        self.start_writing_ics = 0

        # Set some model parameters
        if not self.presolve:
            self.model.setParam('Presolve', 0)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        if self.verbose:
            print("Setting up optimization problem...")
            st = time.time()  # for computing setup time
        
        # Create optimization variables
        self.y = self.model.addMVar((self.sys.p, self.T), lb=-float('inf'), name='y') # Embedding system output
        self.x = self.model.addMVar((self.sys.n, self.T), lb=-float('inf'), name='x') # Embedding system state
        self.u = self.model.addMVar((self.sys.m, self.T), lb=-float('inf'), name='u') # Embedding system input

        # x and x_nominal are in a transformed coordinate space (if V is not None).

        if V_inv is not None:
            self.V_inv = V_inv # Transformation matrix from x to eta. That is, eta := V @ x
            self.V_inv_d = V_inv_d # Decomposition of V_inv.

        self.s = self.model.addMVar((self.sys.m, self.T), lb=-float('inf'), name='s')
        self.rho = self.model.addMVar((len(spec), self.T), name='rho', lb=rho_min) # lb sets minimum robustness
        if w is not None:
            # self.w = self.model.addMVar((self.sys.n, self.T), lb=-float('inf'), name='w')
            self.x_nominal = self.model.addMVar((self.sys_nominal.n, self.T), lb=-float('inf'), name='x_nominal') # Nominal system state
            # self.model.addConstr(self.x_nominal[:, 0] == x0_nominal[:, 0])
            self.SetX0Constraint(self.x0, x0_nominal)
            self.AddNominalDynamicsConstraints()
        else:
            self.SetX0Constraint(self.x0, None)
        self.w=w # disturbance polytope - may be None.

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        if hard_constraint:
            self.AddRobustnessConstraint(rho_min=rho_min)
        if robustness_cost:
            self.ξ = self.model.addMVar((len(spec), self.T), name='ξ', lb=0.0) # slack variable for the robustness
            self.AddRobustnessCost()
            self.AddSoftRobustnessConstraint()

        # Set the model tolerance and maximum computation time.
        # self.model.setParam('IntFeasTol', 1e-3)
        # self.model.setParam('MIPGapAbs', 1e-2)
        # self.model.setParam('OptimalityTol', 1e-3)
        # self.model.setParam('TimeLimit', dT * 0.9)
            

        if self.verbose:
            print(f"Setup complete in {time.time()-st} seconds.")

    def AddControlBounds(self, u_min, u_max):
        # if type(u_max) in [float, int]:
        for t in range(self.T): # range(self.start_point, self.T)
            self.model.addConstr( u_min <= self.u[:,t] )
            self.model.addConstr( self.u[:,t] <= u_max )
        # else:
        #     print('[WARN] bounds must be float or int.')
    def AddControlBoundsTubeMPC(self, u_min, u_max):
        assert( len(u_min) == self.N )
        for t in range(self.T): # Start enforcing TubeMPC input constraints at self.horizon-1.
            t_a = min(t - (self.horizon - 1), self.N-1) # use the last value for everything past t + self.N-1
            t_a = max(0, t_a) # use the first value before self.horizon - 1.
            self.model.addConstr( u_min[t_a] <= self.u[:, t] )
            self.model.addConstr( self.u[:,t] <= u_max[t_a] )

    def AddStateBounds(self, x_min, x_max):
        # x_min and x_max are not transformed.
        V_inv_p, V_inv_n = d_positive(self.V_inv) # Decompose V_inv into positive and negative parts.
        for t in range(self.T):
            # if self.intervals:
            #     self.model.addConstr( V_inv_p @ self.x[:self.sys_nominal.n, t] + V_inv_n @ self.x[self.sys_nominal.n:, t] + self.V_inv @ self.x_nominal[:,t] >= x_min )
            #     self.model.addConstr( V_inv_n @ self.x[:self.sys_nominal.n, t] + V_inv_p @ self.x[self.sys_nominal.n:, t] + self.V_inv @ self.x_nominal[:,t] <= x_max )
            # else:
            #     self.model.addConstr( x_min <= self.V_inv @ self.x[:,t] )
            #     self.model.addConstr( self.V_inv @ self.x[:,t] <= x_max )
            self.model.addConstr( x_min <= self.y[self.sys_nominal.n:,t] )
            self.model.addConstr( self.y[:self.sys_nominal.n,t] <= x_max )

    def AddObstacleConstraint(self, P, p):
        # Let P, p be a set of n > 2 polytopes.
        # We will enforce that the bounds y is in at least one of these polytopes.
        # We will use a binary variable to enforce this.
        n = len(P)
        z = self.model.addMVar(n, vtype=GRB.BINARY)
        for j in range(n):
            self.model.addConstr( P[j] @ self.y <= p[j] + self.M*(1-z[j]) )
        self.model.addConstr( sum(z) >= 1 )

    def AddStateBoundsPolytope(self, P, p):
        for t in range(self.T):
            self.model.addConstr( P[t] @ self.x[:, t] <= p[t] )

    # Add a polytopic constraint to the disturbances.
    def AddDisturbanceBounds(self, P, p):
        for t in range(self.T):
            self.model.addConstr( P[t] @ self.w[:, t] <= p[t])

    def AddQuadraticCost(self, Q=None, R=None):
        # self.cost += self.x[:,0]@Q@self.x[:,0] + self.u[:,0]@R@self.u[:,0]
        if self.w is not None:
            if Q is not None:
                for t in range(self.T):
                    self.cost += self.x_nominal[:,t]@Q@self.x_nominal[:,t]
            if R is not None:
                for t in range(self.T):
                    self.cost += self.u[:,t]@R@self.u[:,t]
        else:
            if Q is not None:
                for t in range(self.T):
                    self.cost += self.x[:,t]@Q@self.x[:,t]
            if R is not None:
                for t in range(self.T):
                    self.cost += self.u[:,t]@R@self.u[:,t]


        if self.verbose:
            print(type(self.cost))
    
    def AddIntervalRobustnessCost(self, index=0, gamma=1):
        # Reward the robustness at a specific STL formula index (if there are multiple)
        self.cost -= gamma * self.rho[index, 0] # We reward larger positive lower bounds on the robustness.

    def AddOneNormInputCost(self, coeffs=None):
        if coeffs is None:
            coeffs = np.ones((1, self.sys.m))
        for t in range(self.T):
            self.cost += coeffs @ self.s[:, t:t+1]

    def AddOneNormInputConstraints(self):
        self.model.addConstr( self.u <= self.s )
        self.model.addConstr( -self.u <= self.s )
    
    # Fix c in R
    # TODO: encode this efficiently!
    def AddInputOneNormConstraint(self, c):
        # Add another dummy variable.
        # d = self.model.addMVar((self.sys.m, self.T), vtype=GRB.CONTINUOUS)
        # # (self.sys.m, self.T), lb=-float('inf')
        # for j in range(self.T): # For each self.u[:, j]
        #     # enforce the arguments are <= d, as is their sum
        #     self.model.addConstr( d[j] >= sum(self.u[j,:] ) )
        #     self.model.addConstr( d[j] >= -sum(self.u[j,:]) )
        for i in range(self.T):
            self.model.addConstr( c >= self.u[:, i]@self.u[:, i] )

    def RemoveLPConstraints(self):
        if len(self.lp_constraints) > 0:
            self.model.remove(self.lp_constraints)
        self.lp_constraints = []
        return
    
    def RemoveX0Constraints(self):
        if len(self.x0_constraints) > 0:
            self.model.remove(self.x0_constraints)
        self.x0_constraints = []
    
    def SetX0Constraint(self, x0=None, x0_nominal=None):
        if x0_nominal is not None:
            self.x0_constraints.append(self.x_nominal[:, 0] == x0_nominal[:, 0])
        if x0 is not None:
            self.x0_constraints.append(self.x[:, 0] == x0[:, 0])
        
        self.x0_constraints = self.model.addConstrs(_ for _ in self.x0_constraints)
        return

    def AddControlCost(self, u_hat):
        for t in range(self.T):
            self.cost += (u_hat - self.u[:, t]) * (self.T - t)
    
    def AddRobustnessCost(self):
        for j in range(len(self.spec)):
            for t in range(self.T):# - self.horizon): # range(self.N + self.horizon):
                if j == 0:
                    gamma_t = 1000
                else:
                    gamma_t = 500
                if t < self.horizon:
                    gamma_t *= 1
                else:
                    pass
                    # gamma_t *= (0.5 ** (t - self.horizon))
                    # gamma_t = 0.5 ** (t - self.horizon)
                self.cost += gamma_t * self.ξ[j,t]
            # self.cost -= 1*self.rho

    def AddUnweightedRobustnessCost(self):
        for j in range(len(self.spec)):
            self.cost -= self.rho[j, self.horizon]

    def AddRobustnessConstraint(self, rho_min=0.0):
        self.model.addConstr( self.rho >= rho_min )
    
    def AddSoftRobustnessConstraint(self):
        self.model.addConstr(self.rho >= -self.ξ)

    def Solve(self):
        # Set the cost function now, right before we solve.
        # This is needed since model.setObjective resets the cost.
        # print(type(self.cost))
        # print(self.cost)
        self.model.setObjective(self.cost, GRB.MINIMIZE)
        # print(self.model.getVars())
        # Do the actual solving
        self.model.optimize()
        success = None

        if self.model.status == GRB.OPTIMAL:
            if self.verbose:
                print("\nOptimal Solution Found!\n")
            x = self.x.X
            u = self.u.X
            # Ensure that x_nominal is defined
            if self.w is not None:
                x_nominal = self.x_nominal.X
            else:
                x_nominal = self.x.X
            rho = self.rho.X

            # Report optimal cost and robustness
            if self.verbose:
                print("Solve time: ", self.model.Runtime)
                print("Optimal robustness: ", rho)
                print("")
                # for i in range(len(self.z_specs)):
                #     print(f'Resultant z_specs[{i}]: ', self.z_specs[i].X)
            
            success = True
            objective = self.model.getObjective().getValue()
        else:
            if self.verbose:
                print(f"\nOptimization failed with status {self.model.status}.\n")
            x = None
            u = None
            x_nominal = None
            rho = -np.inf
            success = False
            objective = np.inf
        
        if self.verbose:
            print(self.model.status)
        return (x,u,x_nominal,rho,self.model.Runtime,objective,success)

    def AddNominalDynamicsConstraints(self):
        # Create constraints to model nominal dynamics of an embedding system.
        for t in range(self.T-1):
            # Use a variable self.x_nominal to represent the nominal dynamics.
            self.model.addConstr( self.x_nominal[:, t+1] == self.sys_nominal.A@self.x_nominal[:, t] + self.sys_nominal.B@self.u[:, t] )


    # Note: the indices were extensively verified as of 7/9/23 to be correct.
    def AddDynamicsConstraints(self):
        # Dynamics (that are not updated with each update to the historical states)
        if self.w is not None:
            for t in range(self.T-1): #range(self.horizon-1, self.T-1):
                if self.verbose:
                    print(f'initializing index {t}, x[:,{t+1}], y[:,{t}] using u[:,{t}]')
                if type(self.w) is np.ndarray:
                    # self.model.addConstr(
                    #         self.x[:,t+1] == self.sys.A@self.x[:,t] +
                    #                          self.sys.B@self.u[:,t] +
                    #                          self.w[:,0]
                    #  )
                    
                    # Another attempt. This is the error dynamics.
                    self.model.addConstr(
                        self.x[:, t+1] == self.sys.A @ self.x[:, t] +
                                        self.w[:,0]
                    )
                else:
                    print('Alert: w is not an np.ndarray.')
                    self.model.addConstr(
                            self.x[:,t+1] == self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t] + self.w[:,t] )
            # self.model.addConstr( # \oly per AddSTLConstraints
            #     self.y[:self.sys_nominal.n, :] == self.V_inv @ self.x[:self.sys_nominal.n, :] + self.V_inv @ self.x_nominal # Noting that our dynamics is written as [\ulx, \olx], opposite of y.
            # )
            # self.model.addConstr( # \uly per AddSTLConstraints
            #     self.y[self.sys_nominal.n:, :] == self.V_inv @ self.x[self.sys_nominal.n:, :] + self.V_inv @ self.x_nominal
            # )
            V_inv_p, V_inv_n = d_positive(self.V_inv) # Decompose V_inv into positive and negative parts.
            self.model.addConstr( # \uly
                self.y[self.sys_nominal.n:, :] == V_inv_p @ self.x[self.sys_nominal.n:, :] + V_inv_n @ self.x[:self.sys_nominal.n, :] + self.V_inv @ self.x_nominal
            )
            self.model.addConstr( # \oly
                self.y[:self.sys_nominal.n, :] == V_inv_n @ self.x[self.sys_nominal.n:, :] + V_inv_p @ self.x[:self.sys_nominal.n, :] + self.V_inv @ self.x_nominal
            )
        else:
            for t in range(self.T-1):
                if self.verbose:
                    print(f'initializing index {t}, x[:,{t+1}], y[:,{t}] using u[:,{t}]')
                self.model.addConstr(
                        self.x[:,t+1] == self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t]
                )
            for t in range(self.T):
                self.model.addConstr(
                    self.y[:,t] == self.sys.C@self.x[:,t] + self.sys.D@self.u[:,t]
                )

    def RemoveDynamicsConstraints(self):
        # remove previously added dynamics constraints.
        if len(self.dynamics_constraints) > 0:
            self.model.remove(self.dynamics_constraints)
        
        # reset the array
        self.dynamics_constraints = []

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Recursively traverse the tree defined by the specification
        # to add binary variables and constraints that ensure that
        # rho is the robustness value

        #
        self.z_spec = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
        for j in range(len(self.spec)):
            temp_spec = self.spec[j]
            self.AddFormulaConstraintsNotRecursive(temp_spec, self.z_spec, 0, 0, j)

        self.model.addConstr( self.z_spec == 1)

    def AddFormulaConstraintsNotRecursive(self, topLevelFormula, topZ, topT, topIdx, specIdx):
        # start with a stack data structure
        stack = []
        stack.append((topLevelFormula, topZ, topT, topIdx))
        while len(stack) > 0:
            (formula, z, t, idx) = stack.pop()
            if isinstance(formula, LinearPredicate):
                if t < 0: continue # the index is invalid most likely due to a past time formula.
                if self.intervals:
                    if formula.b.dtype == np.interval :
                        _b, b_ = get_lu(formula.b)
                    else :
                        _b = formula.b; b_ = formula.b

                    if formula.a.dtype == np.interval :
                        _a, a_ = get_lu(formula.a)
                    else :
                        _a = formula.a; a_ = formula.a

                    ps = _a.shape[0] # The length of the output vector, non-decomposed.
                    # ps = 1
                    s_js = []
                    for j in range(ps): # Through the a-vectors.
                        # if np.all(np.abs([_a, a_]) < 1e-3): continue # Do not bother: everything is zero.
                        _a__y = self.model.addVar(lb=-self.M)
                        _a_y_ = self.model.addVar(lb=-self.M)
                        a___y = self.model.addVar(lb=-self.M)
                        a__y_ = self.model.addVar(lb=-self.M)

                        # print(f't = {t}')
                        # print(f'j = {j}')
                        # print(f'shape of self.y[j, t:t+1] is {self.y[j, t:t+1].shape}')

                        self.model.addConstr( _a__y == _a[j] * self.y[j, t:t+1] )
                        self.model.addConstr( a___y == a_[j] * self.y[j, t:t+1])
                        if self.w is not None:
                            self.model.addConstr( _a_y_ == _a[j] * self.y[j + ps, t:t+1])
                            self.model.addConstr( a__y_ == a_[j] * self.y[j + ps, t:t+1])
                        else:
                            self.model.addConstr( _a_y_ == _a[j] * self.y[j, t:t+1])
                            self.model.addConstr( a__y_ == a_[j] * self.y[j, t:t+1])

                        s_j = self.model.addVar(lb=-self.M)
                        if (_a[j] > 0):
                            self.model.addConstr(s_j == min_(_a__y, a___y))
                        elif (a_[j] < 0):
                            self.model.addConstr(s_j == min_(_a_y_, a__y_))
                        else:
                            self.model.addConstr(s_j == min_(_a__y, _a_y_, a___y, a__y_))
                        s_js.append(s_j)

                    s = self.model.addVar(lb=-self.M)
                    self.model.addConstr(s <= sum(s_js))
                    
                    # Now, for the constraints.
                    self.model.addConstr( s - b_ + (1-z) * self.M >= self.rho[specIdx, t])
                elif self.tubeMPCEnabled and t >= self.horizon: # enforced on y, not u, hence self.horizon and not self.horizon-1.
                    tubeIdx = min(t-self.horizon, self.N-2)
                    if np.all(formula.a >= 0):
                        self.model.addConstr( formula.a.T @ (self.y[:,t:t+1] - self.tubeMPCBuffer[tubeIdx]) - \
                            formula.b + (1-z)*self.M  >= self.rho[specIdx, t] )
                    else:
                        self.model.addConstr( formula.a.T @ (self.y[:,t:t+1] + self.tubeMPCBuffer[tubeIdx]) - \
                            formula.b + (1-z)*self.M  >= self.rho[specIdx, t] )
                else:
                    # print('intervals was false.')
                    self.model.addConstr( formula.a.T @ self.y[:,t:t+1] - formula.b + (1-z)*self.M  >= self.rho[specIdx, t] )
                b = self.model.addMVar(1,vtype=GRB.BINARY)
                self.model.addConstr(z == b)
            elif isinstance(formula, NonlinearPredicate):
                raise TypeError("Mixed integer programming does not support nonlinear predicates")
            else:
                if formula.combination_type == "and":
                    for i, subformula in enumerate(formula.subformula_list):
                        t_sub = formula.timesteps[i]
                        z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                        stack.append((subformula, z_sub, t+t_sub, idx))
                        self.model.addConstr( z <= z_sub )
                else:  # combination_type == "or":
                    z_subs = []
                    for i, subformula in enumerate(formula.subformula_list):
                        z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                        z_subs.append(z_sub)
                        t_sub = formula.timesteps[i]
                        stack.append((subformula, z_sub, t+t_sub, idx)) # Negative times are handled by the predicate.

                    self.model.addConstr( z <= sum(z_subs) )
