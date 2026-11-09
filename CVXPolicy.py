import torch
import torch.nn as nn
import cvxpy as cp
#Imports the differentiable optimization-layer class.
from cvxpylayers.torch import CvxpyLayer 
from ImplicitNets import DefaultPNet
from ImplicitNets import Phi
from Stochastic2D import Stochastic2D
import numpy as np

class CVXPolicy_S2D(nn.Module):
    """
    A policy network that uses CVXPYLayers to solve for the optimal control u* using 
    Jacobian based backpropagation for the Stochastic 2D problem
    """
    def __init__(self, state_dim, control_dim, soc_problem, p_net=None, tol = 1e-3, dev='cpu'):
        super().__init__()
        self.state_dim = state_dim 
        self.control_dim = control_dim
        self.soc_problem = soc_problem

        self.tol = tol

        if p_net is not None:
            self.p_net = p_net 
        else:
            self.p_net = DefaultPNet(state_dim, hidden_dim=100, dev = dev)

        # use CVXPY Layer to solve for control u* that minimize the Hamiltonian
        # H = -tr(sigma^T q) - f^T p - L(u)

        # the unknown variable
        u_cp = cp.Variable(control_dim)
        # the parameters that we supply which can change between solver
        p_cp = cp.Parameter(state_dim)
        q_cp = cp.Parameter(state_dim, state_dim)
        sigma = soc_problem.sigma.detach().cpu().numpy()
        # Objective function 
        # defined the objective function
        obj_cp = cp.Minimize(p_cp @ u_cp +cp.sum(cp.multiply(sigma, q_cp)) +0.5*cp.sum_squares(u_cp)+(1/20)*cp.power(cp.sum_squares(u_cp)+1e-3,1.5))
        # turns the problem into a differentiable layer 
        self.u_layer = CvxpyLayer(cp.Problem(obj_cp),[p_cp, q_cp], [u_cp])

    def forward(self, z, t, track_all_fp_iters=False):
        u0 = torch.zeros(z.shape[0], self.control_dim, device=z.device, dtype=z.dtype)
        p, q = self.soc_problem.compute_q(t,z,u0, self.p_net)
        q_flat = q.reshape(q.shape[0], -1)
        # provide p and solve for u.
        ustar, = self.u_layer(p, q_flat.detach(), solver_args={'eps': self.tol})

        return ustar.to(dtype=z.dtype)

def example_Stochastic2D(batch_size, state_dim, control_dim, z0):
    problem = Stochastic2D(batch_size=batch_size)
    policy = CVXPolicy_S2D(state_dim, control_dim, problem)
    u_star = policy(z0, 0.0)
    # direct layer, written independently of the policy: minimize f^T p + tr(sigma^T q) + L(u), f = u
    u_cp = cp.Variable(control_dim)
    p_cp = cp.Parameter(state_dim)
    q_cp = cp.Parameter((state_dim, problem.noise_dim))
    sigma = problem.sigma.detach().cpu().numpy()
    obj_cp = cp.Minimize(cp.sum(p_cp @ u_cp + cp.sum(cp.multiply(sigma, q_cp))
                                + problem.alphaL*(0.5*cp.sum_squares(u_cp)+(1/20)*cp.power(cp.sum_squares(u_cp)+1e-3,1.5))))
    u_layer = CvxpyLayer(cp.Problem(obj_cp), [p_cp, q_cp], [u_cp])
    # p = V_z and q = V_zz sigma from ImplicitSOC.compute_q (sigma is constant, so u only needs the right shape)
    p, q = problem.compute_q(0.0, z0, torch.zeros(batch_size, control_dim), policy.p_net)
    ustar_dir, = u_layer(p, q.detach(), solver_args={"eps": 1e-3}) 
 
    print(f'difference: {torch.norm(ustar_dir - u_star)}')

def main():
    batch_size = 10
    state_dim = 2
    control_dim = 2
    z0 = torch.randn(batch_size, state_dim)
    example_Stochastic2D(batch_size, state_dim, control_dim, z0)

if __name__ == '__main__':
    main()

    
