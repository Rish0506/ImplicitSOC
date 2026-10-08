import torch
import math
from utils import StochasticGradientTester
from utils import GradientTester_Taylors

torch.set_default_dtype(torch.float64)

class TrajectorySOC:
    def __init__(self, batch_size=4, t_initial=0.0, t_final=1.0, num_steps=100, device='cpu'):
        self.batch_size = batch_size
        self.state_dim = 2
        self.control_dim = 2
        self.noise_dim = 2
        self.device = device
        self.dt = (t_final - t_initial) / num_steps
        self.z_target = torch.tensor([1.5,1.5], device=self.device)
        self.sigma = torch.tensor([[0.2,-0.4],[-0.4,0.2]], device=self.device)

    def Q(self, z):
        return  50.0/(0.8*math.pi)*torch.exp(-0.5*torch.sum(z**2, dim=-1)/0.4)
    def compute_grad_Q(self, z):
        z_sq = torch.sum(z**2, dim=-1, keepdim=True)
        return -50.0/(0.8*math.pi)*torch.exp(-0.5*z_sq/0.4)*z/0.4
    def compute_f(self, t, z, u):
        return u 
    def compute_grad_f_u(self, t, z, u):
        return torch.eye(self.control_dim, device=self.device).expand(z.shape[0], -1, -1)
    def compute_grad_f_z(self, t, z, u):
        return torch.zeros(z.shape[0], self.state_dim, self.state_dim, device=self.device)
    def compute_lagrangian(self, t, z, u):
        u_sq = torch.sum(u**2, dim=-1)
        return 0.5*u_sq+(1/20)*((u_sq) +0.001)**1.5+ self.Q(z)
    def compute_grad_lagrangian(self, t, z, u):
        u_sq = torch.sum(u**2, dim=-1, keepdim=True)
        return u + u*(3/20)*((u_sq) +0.001).sqrt() 
    def compute_grad_L_z(self, t, z, u):
        return self.compute_grad_Q(z)
    def compute_sigma(self, t, z, u):
        return self.sigma
 
    def compute_grad_sigma_z(self, t, z, u):      # (B, state, noise, state)
        return torch.zeros(z.shape[0], 2, 2, 2, device=self.device)
 
    def compute_grad_sigma_u(self, t, z, u):      # (B, state, noise, control)
        return torch.zeros(z.shape[0], 2, 2, 2, device=self.device)
 
    # ---- terminal cost ----
    def compute_G(self, t, z, u=None):
        return 50.0*torch.sum((z - self.z_target)**2, dim=-1)
 
    def compute_grad_G_z(self, t, z, u=None):
        return 100.0*(z - self.z_target)
 
 
if __name__ == "__main__":
    problem = TrajectorySOC(batch_size=4)
    B = problem.batch_size
    z = math.sqrt(0.4)*torch.randn(B, 2) - 1.5   # z0 ~ N(-1.5, 0.4 I)
    u = torch.randn(B, 2)
    t = torch.zeros(B)                           # tensor (not float) so vmap can batch over it
    StochasticGradientTester.run_all_tests(problem, z=z, u=u, t=t)


    taylor_checker = GradientTester_Taylors()

    z0 = z[0]
    u0 = u[0]
    t0 = t[0]
    #w = torch.tensor([0.6,-0.4], device=problem.device)

    taylor_checker.check_gradient(cost_func=lambda u_: problem.compute_lagrangian(t0, z0, u_),
                                 grad_func=lambda u_: problem.compute_grad_lagrangian(t0, z0, u_),x=u0,)

    taylor_checker.check_gradient( cost_func=lambda z_: problem.compute_lagrangian(t0, z_, u0),
                                  grad_func=lambda z_: problem.compute_grad_L_z(t0, z_, u0), x=z0,)

    taylor_checker.check_gradient( cost_func=lambda z_: problem.compute_G(t0, z_),
                                  grad_func=lambda z_: problem.compute_grad_G_z(t0, z_),x=z0,)


    #taylor_checker.check_gradient(cost_func=lambda z_: problem.compute_f(t0, z_, u0),
    #                              grad_func=lambda z_: problem.compute_grad_f_z(t0, z_, u0)[0],x=z0,)

    #taylor_checker.check_gradient(cost_func=lambda u_: problem.compute_f(t0, z0, u_),
    #                              grad_func=lambda u_: problem.compute_grad_f_u(t0, z0, u_),x=u0,)

    #taylor_checker.check_gradient(cost_func=lambda z_: problem.compute_sigma(t0, z_, u0),
    #                              grad_func=lambda z_: problem.compute_grad_sigma_z(t0, z_, u0),x=z0,)

    #taylor_checker.check_gradient(cost_func=lambda u_: problem.compute_sigma(t0, z0, u_),
    #                              grad_func=lambda u_: problem.compute_grad_sigma_u(t0, z0, u_),x=u0,)
    
    taylor_checker.check_gradient(cost_func=lambda z_: problem.Q(z_),
                                  grad_func=lambda z_: problem.compute_grad_Q(z_),x=z0,)