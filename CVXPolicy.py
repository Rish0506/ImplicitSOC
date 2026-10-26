import torch
import torch.nn as nn
import cvxpy as cp
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
    def __init__(self, state_dim, control_dim, p_net=None, tol = 1e-3, dev='cpu'):
        super().__init__()
        self.state_dim = state_dim 
        self.control_dim = control_dim

        self.tol = tol

        if p_net is not None:
            self.p_net = p_net 
        else:
            self.p_net = DefaultPNet(state_dim, hidden_dim=100, dev = dev)

        # use CVXPY Layer to solve for control u* that minimize the Hamiltonian
        # H = tr(sigma^T q) + f^T p - L(u)

        u_cp = cp.Variable(control_dim)
        p_cp = cp.Parameter(state_dim)
        # Objective function 
        obj_cp = cp.Minimize(-p_cp @ u_cp + 0.5*cp.sum_squares(u_cp)+(1/20)*cp.power(cp.sum_squares(u_cp)+1e-3,1.5))
        self.u_layer = CvxpyLayer(cp.Problem(obj_cp),[p_cp], [u_cp])

    def forward(self, z, t, track_all_fp_iters=False):
        p = self.p_net(t,z)
        ustar, = self.u_layer(p, solver_args={'eps': self.tol})

        return ustar.to(dtype=z.dtype)

def example_Stochastic2D(batch_size, state_dim, control_dim, z0):
    policy = CVXPolicy_S2D(state_dim, control_dim)
    u_star = policy(z0, 0.0)
    u_cp = cp.Variable(control_dim)
    p_cp = cp.Parameter(state_dim)
    obj_cp = cp.Minimize(cp.sum(-p_cp @ u_cp  + 0.5*cp.sum_squares(u_cp)+(1/20)*cp.power(cp.sum_squares(u_cp)+1e-3,1.5)))
    u_layer = CvxpyLayer(cp.Problem(obj_cp), [p_cp], [u_cp])
    p = policy.p_net(0.0,z0)
    ustar_dir, = u_layer(p, solver_args={"eps": 1e-3}) 

    print(f'difference: {torch.norm(ustar_dir - u_star)}')

def main():
    batch_size = 10
    state_dim = 2
    control_dim = 2
    z0 = torch.randn(batch_size, state_dim)
    example_Stochastic2D(batch_size, state_dim, control_dim, z0)

if __name__ == '__main__':
    main()

    
