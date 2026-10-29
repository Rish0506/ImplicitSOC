import torch
import math
from ImplicitSOC import ImplicitSOC
from ImplicitNets import ImplicitNetSOC
from ImplicitNets import Phi
import matplotlib.pyplot as plt
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Stochastic2D(ImplicitSOC):
    def __init__(self, batch_size=50, t_initial=0.0, t_final=1.0, nt=250, alphaL=1.0, alphaG=1.0, device='cpu', ic_mean=[-1.5,-1.5], ic_var = 0.4, sigma=None, pen_pos = False):
        state_dim =2 
        control_dim = 2
        noise_dim = 2
        super().__init__(state_dim, control_dim,noise_dim,  batch_size, t_initial, t_final, nt, alphaL, alphaG, device)
        self.soc_problem_name = "Stochastic 2D"
        self.ic_mean = torch.as_tensor(ic_mean, dtype = torch.float32, device = self.device).view(1,2)
        self.ic_var = ic_var
        self.z_target = torch.tensor([1.5,1.5], dtype=torch.float32, device=self.device).repeat(batch_size, 1)
        self.sigma = torch.tensor([[0.2,-0.4],[-0.4,0.2]], dtype=torch.float32, device=self.device)

    def Q(self, z):
        z = z.view(-1, self.state_dim)
        return  50.0/(0.8*math.pi)*torch.exp(-0.5*torch.sum(z**2, dim=-1)/0.4)
    def compute_grad_Q_z(self, z):
        z = z.view(-1, self.state_dim)
        z_sq = torch.sum(z**2, dim=-1, keepdim=True)
        return -50.0/(0.8*math.pi)*torch.exp(-0.5*z_sq/0.4)*z/0.4
    def compute_f(self, t, z, u):
        u = u.view(-1, self.control_dim)
        #u_sq = torch.sum(u**2, dim=-1)
        return  u
    def compute_grad_f_u(self, t, z, u):
        return torch.eye(self.control_dim, device=self.device).unsqueeze(0).expand(z.shape[0], -1, -1).clone()
        #return 2*u
    def compute_grad_f_z(self, t, z, u):
        return torch.zeros(z.shape[0], self.state_dim, self.state_dim, device=self.device)
        #return 2*z
    # Non-batched derivative of f w.r.t. u
    # def compute_grad_f_u(self, z, u, grad_f_u):
    #     grad_f_u.copy_(torch.eye(self.control_dim, device = self.device))
    #     return grad_f_u
    
    def compute_lagrangian(self, t, z, u):
        u_sq = torch.sum(u**2, dim=-1)
        loss = 0.5*u_sq+(1/20)*((u_sq) +0.001)**1.5+ self.Q(z)
        return loss
    def compute_grad_lagrangian(self, t, z, u):
        u = u.view(-1, self.control_dim)
        u_sq = torch.sum(u**2, dim=-1, keepdim=True)
        return u + u*(3/20)*((u_sq) +0.001).sqrt() 
    def compute_grad_L_z(self, t, z, u):
        z = z.view(-1, self.state_dim)
        return self.compute_grad_Q_z(z)
    def compute_sigma(self, t, z, u):
        return self.sigma.unsqueeze(0).expand(z.shape[0],-1,-1)
    
 
    def compute_grad_sigma_z(self, t, z, u):      # (B, state, noise, state)
         return torch.zeros(z.shape[0], 2, 2, 2, device=self.device)
 
    def compute_grad_sigma_u(self, t, z, u):      # (B, state, noise, control)
         return torch.zeros(z.shape[0], 2, 2, 2, device=self.device)
 
    def compute_G(self,z):
        z = z.view(-1, self.state_dim)
        target = self.z_target[:z.shape[0]]
        return 50.0 * torch.sum((z - target)**2, dim=-1)
    def compute_grad_G_z(self, z):
        z = z.view(-1, self.state_dim)
        target = self.z_target[:z.shape[0]]
        return 100.0*(z - target)

    def generate_trajectory(self, u, z0, nt, return_full_trajectory=False):
        """"
        Generates trajectory using Euler integration.
        """
        batch_size = z0.shape[0]
        z = torch.zeros(batch_size, self.state_dim, nt+1, device = self.device)
        z[:,:,0] = z0
        h = (self.t_final-self.t_initial)/nt
        t_i = self.t_initial

        for i in range(nt):
            if hasattr(u, "forward"):
                current_u = u(z[:,:,i], t_i)
            else:
                current_u = u[:,:,i]

            sigma = self.compute_sigma(t_i, z[:,:,i], current_u)
            dW = (h**0.5) * torch.randn(batch_size, sigma.shape[2],device = self.device )
            noise_term = torch.bmm(sigma, dW.unsqueeze(-1)).squeeze(-1)
            z[:,:,i+1] = z[:,:,i] + h*self.compute_f(t_i, z[:,:,i], current_u) + noise_term
            t_i = t_i + h 
        return z if return_full_trajectory else z[:,:,-1]
    
    
    def sample_initial_condition(self):
        noise = torch.randn(int(self.batch_size), self.state_dim, device=self.device)
        return self.ic_mean + math.sqrt(self.ic_var)*noise

    def plot_trajectories(self, z, z_target=None, filename="trajectories.png"):
        if z_target is None:
            z_target = self.z_target
            # Plot: contour of the Gaussian bump Q so the reader sees why the paths bend around the origin
        xs = torch.linspace(-3, 3, 149)
        X, Y = torch.meshgrid(xs, xs, indexing="xy")
        Qv = self.Q(torch.stack([X.reshape(-1), Y.reshape(-1)], 1).to(self.device)).reshape(X.shape)
        plt.contourf(X.numpy(), Y.numpy(), Qv.detach().cpu().numpy(), levels=10, cmap="Reds", alpha=0.5)
        plt.colorbar(label="Q(z)")
        for b in range(z.shape[0]):
            # extract x and y coorindates, detach them from autograd, cpu moves them from GPU if needed, .numpy() converts them fro matplotlib
            x = z[b,0,:].detach().cpu().numpy()
            y = z[b,1,:].detach().cpu().numpy()
            plt.plot(x,y,alpha=0.7)

        # Plot: initial states and final states
        plt.scatter(z[:,0,0].detach().cpu(), z[:,1,0].detach().cpu(), s=12, c="k", label="z(0)")
        plt.scatter(z[:,0,-1].detach().cpu(), z[:,1,-1].detach().cpu(), s=12, c="b", marker="x", label="z(T)")
        target = z_target[0].detach().cpu().numpy()
        plt.scatter(target[0], target[1], marker="*", s=120, c="gold", edgecolors="k", label="Target", zorder=5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Stochastic 2D trajectories")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    nt = 250
    problem = Stochastic2D(batch_size=batch_size, nt=nt, device=device)
    print("\n" + "="*50)
    print("Performing Gradient Check for functions")

    from utils import GradientTester_Taylors as GradientTester
    z = problem.sample_initial_condition()
    u = torch.randn(batch_size, 2, device=device)
    t = torch.zeros(batch_size, device=device)                          # tensor (not float) so vmap can batch over it
    #StochasticGradientTester.run_all_tests(problem, z=z, u=u, t=t)
    taylor_checker = GradientTester()
    # For z= [10,20,30], z[0] = 10 so size is 0. while z[0:1] = [10] size is 1.
    # For z = ([[1,2], [3,4]]), z[0] = ([1,2]), z[0:1] = ([[1,2]]) 
    z0 = z[0:1]
    u0 = u[0:1]
    t0 = t[0:1]

    z0_check = z0.double()
    u0_check = u0.double()
    t0_check = t0.double()

    taylor_checker.check_gradient(cost_func=lambda u_: problem.compute_lagrangian(t0_check, z0_check, u_),
                                grad_func=lambda u_: problem.compute_grad_lagrangian(t0_check, z0_check, u_),x=u0_check,)

    taylor_checker.check_gradient( cost_func=lambda z_: problem.compute_lagrangian(t0_check, z_, u0_check),
                                 grad_func=lambda z_: problem.compute_grad_L_z(t0_check, z_, u0_check), x=z0_check,)

    # taylor_checker.check_gradient( cost_func=lambda u_: problem.compute_f(t0, z0, u_).sum(),
    #                               grad_func=lambda u_: problem.compute_grad_f_u(t0, z0, u_),x=u0,)

    # taylor_checker.check_gradient( cost_func=lambda z_: problem.compute_f(t0, z_, u0),
    #                               grad_func=lambda z_: problem.compute_grad_f_z(t0, z_, u0),x=z0,)


    # taylor_checker.check_gradient( cost_func=lambda u_: problem.compute_sigma(t0, z0, u_),
    #                               grad_func=lambda u_: problem.compute_grad_sigma_u(t0, z0, u_),x=u0,)  

    
    # taylor_checker.check_gradient( cost_func=lambda z_: problem.compute_sigma(t0, z_, u0),
    #                               grad_func=lambda z_: problem.compute_grad_sigma_z(t0, z_, u0),x=z0,)

    taylor_checker.check_gradient( cost_func=lambda z_: problem.compute_G(z_),
                                   grad_func=lambda z_: problem.compute_grad_G_z(z_),x=z0_check,)

    taylor_checker.check_gradient(cost_func=lambda z_: problem.Q(z_),
                                   grad_func=lambda z_: problem.compute_grad_Q_z(z_),x=z0_check,)

    print("="*50+"\n")
    #================================================
    # dimension of the problem
    state_dim = problem.state_dim
    control_dim = problem.control_dim
    print(f"Running on device: {device}")
    print(f"State Dimension: {state_dim}, Control Dimension: {control_dim}")
    # constructs Phi_net from ImplicitNets
    Phi_net = Phi(3,10,state_dim, dev = device)
    with torch.no_grad():
        Phi_net.A.zero_()
        Phi_net.w.weight.zero_()
        Phi_net.c.weight.zero_()
        Phi_net.c.weight[0, 0] = 4.9
        Phi_net.c.weight[0, 1] = 4.9
    u_net_implicit = ImplicitNetSOC(state_dim = state_dim, control_dim=control_dim,hidden_dim = 512, tol=1e-3, soc_problem = problem, 
                                    p_net = Phi_net, use_aa=False).to(device)
    # plotting the traj
    with torch.no_grad():
        print("u at t=0", u_net_implicit(z0, 0.0))
    z_traj = problem.generate_trajectory(u_net_implicit, z0, nt, return_full_trajectory=True)
    problem.plot_trajectories(z_traj.detach())

    # Creats an Adam optimizer for the network parameters
    optimizer = torch.optim.Adam(u_net_implicit.parameters(), lr=1e-3)
    # Creates a learning rate scheduler that would reduce the leanring rate after the monitoered loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode ='min', factor = 0.8, patience=15)

    plot_frequency = 1
    best_loss = float('inf')
    max_epochs = int(1)

    # Starts the iteration timer and clear gradients left from a previous optimizer step
    for i in range(1, max_epochs +1):
        start_time = time.time()
        optimizer.zero_grad()
        # calls compute_loss from ImplicitSOC 
        z_batch = problem.sample_initial_condition()
        loss, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, largest_grad_H_u, avg_grad_H_u = problem.compute_loss(u_net_implicit, z0)
        # plotting trajectories
        z_traj = problem.generate_trajectory(u_net_implicit, z_batch, nt, return_full_trajectory=True)
        problem.plot_trajectories(z_traj.detach())
        #Use automatic differentiation to compute gradients of the loss with respect to network parameters.
        loss.backward()
        # clips the overall gradient norm at most 1. his helps preventing large parameter updates. 
        torch.nn.utils.clip_grad_norm_(u_net_implicit.parameters(),1.0)
        # Updates the network parameters using Adam, then passed the loss to learning rate scheduler. 
        # The scheduler is using the loss calculated before the parameter update.
        optimizer.step()
        scheduler.step(loss.item())
        if i % plot_frequency == 0:
            z_traj = problem.generate_trajectory(u_net_implicit, z_batch, nt, return_full_trajectory=True)
            problem.plot_trajectories(z_traj.detach())

        # Calculates the combined Euclidean norm of the parameter gradients
        # The gradient are available from optimizer.step(). so this reports the gradients from the update just perfomred. 
        grad_norm = sum(torch.norm(p.grad)**2 for p in u_net_implicit.parameters() if p.grad is not None)**0.5
        iter_time = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        print(f"iter {i}, loss {loss.item():.3e}, grad {grad_norm.item():.3e}, L {running_cost.item():.3e}, G {terminal_cost.item():.3e}, "
              f"HJB {cHJB.item():.3e}, HJBfin {cHJBfin.item():.3e}, Adj {cadj.item():.3e}, Adjfin {cadjfin.item():.3e}, "
              f"time: {iter_time:.3e}, lr: {lr:.3e}")

if __name__ == "__main__":
    main()





