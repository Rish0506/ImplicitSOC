import math
import torch
from ImplicitSOC import ImplicitSOC
import matplotlib.pyplot as plt

class MultiAssetLiquidation(ImplicitSOC):
    def __init__(self, batch_size=50, t_initial = 0.0, t_final = 1.0, nt=250, alphaL=1, alphaG=1, device='cpu', ic_var=0.0):
        self.n_assets = 10 # number of assets
        state_dim = 1+ 2*self.n_assets
        control_dim = self.n_assets 
        noise_dim = self.n_assets
        super().__init__(state_dim, control_dim, noise_dim, batch_size, t_initial, t_final,nt, alphaL, alphaG, device)
        self.noise_dim = noise_dim
        self.soc_problem_name = '10-asset liquidation'
        self.ic_var = float(ic_var)
        self.kappa = 0.1
        self.shi = 0.1
        self.terminal_penalty = 0.5
        n = self.n_assets
        eye = torch.eye(n, device=self.device)
        ones = torch.ones(n,n, device = self.device)
        self.B = 0.02*eye + 0.001 * ones
        self.covariance = 0.04 * (0.8 * eye * 0.2 * ones)
        self.sigma_S = torch.linalg.cholesky(self.covariance)
        self.sigma = torch.cat((torch.zeros(1,n,device=self.device),self.sigma_S, torch.zeros(n,n,device=self.device)), dim=0)
        self.ic_mean = torch.cat((torch.zeros(1,device= self.device),100*torch.ones(n,device = self.device), 5.0*torch.ones(n,device=self.device))).view(1,-1)
        # Inventory target 
        self.inventory_target = torch.zeros(n, device=self.device)

    def _parts(self, z):
        z = z.reshape(-1, self.state_dim)
        n = self.n_assets 
        return z[:,0:1], z[:,1:1+n], z[:,1+n:1+2*n]
    
    @staticmethod
    def _like(constant, reference):
        return constant.to(device = reference.device, dtype = reference.dtype)

    def compute_f(self, t, z, u):
        u = u.reshape(-1,self.control_dim)
        _, S, _ = self._parts(z)
        B = self._like(self.B, u)
        cash_drift = (u*S - self.kappa * u.square()).sum(dim=-1, keepdim = True)
        price_drift = - (u.pow(7) @ B.T)
        return torch.cat((cash_drift, price_drift, -u), dim=-1)

    def compute_grad_f_z(self, t, z, u):
        u = u.reshape(-1,self.control_dim)
        batch = u.shape[0]
        n = self.n_assets 
        cash_row = torch.cat((u.new_zeros(batch,1), u, u.new_zeros(batch,n)), dim=1).unsqueeze(1)
        return torch.cat((cash_row, u.new_zeros(batch, 2*n, self.state_dim)), dim=1)

    def compute_grad_f_u(self, t, z, u):
        u = u.reshape(-1,self.control_dim)
        _, S, _ = self._parts(z)
        n = self.n_assets 
        cash_row = (S-2.0*self.kappa*u).unsqueeze(1)
        B = self._like(self.B,u)
        # entry (i,j) is -7B[i,j]*u[j]**6
        price_rows = -7.0 * B.unsqueeze(0) * u.pow(6).unsqueeze(1)
        inventory_rows = - torch.eye(n, device=u.device, dtype = u.dtype).unsqueeze(0).expand(u.shape[0], -1, -1)
        return torch.cat((cash_row, price_rows, inventory_rows), dim=1)

    def compute_sigma(self, t, z, u):
        z = z.reshape(-1,self.state_dim)
        return  self._like(self.sigma, z).unsqueeze(0).expand(z.shape[0], -1, -1)

    def compute_grad_sigma_z(self, t, z, u):
        z = z.reshape(-1,self.state_dim)
        return z.new_zeros(z.shape[0],self.state_dim, self.noise_dim, self.state_dim)

    def compute_grad_sigma_u(self, t, z, u):
        z = z.reshape(-1,self.state_dim)
        return z.new_zeros(z.shape[0],self.state_dim, self.noise_dim, self.state_dim)

    def compute_lagrangian(self, t, z, u):
        _, _, Q = self._parts(z)
        C = self._like(self.covariance, Q)
        return self.shi * ((Q @ C) * Q).sum(dim=-1)

    def compute_grad_lagrangian(self, t, z, u):
        u = u.reshape(-1, self.control_dim)
        return torch.zeros_like(u)

    def compute_grad_L_z(self, t, z, u):
        _, _, Q = self._parts(z)
        C = self._like(self.covariance, Q)
        return torch.cat((Q.new_zeros(Q.shape[0], 1+self.n_assets), 2*self.shi*(Q @ C)), dim=-1)

    def compute_G(self, z):
        X, S, Q = self._parts(z)
        return (self.terminal_penalty * Q.square().sum(dim=-1)-X.squeeze(-1)-(Q*S).sum(dim=-1))

    def compute_grad_G_z(self, z):
        _, S, Q = self._parts(z)
        return torch.cat((-Q.new_ones(Q.shape[0], 1), -Q, 2*self.terminal_penalty * Q - S), dim = -1)

    def sample_initial_condition(self):
        if self.ic_var == 0:
            return self.ic_mean.expand(self.batch_size, -1).clone()
        z = self.ic_mean + math.sqrt(self.ic_var)*torch.randn(self.batch_size, self.state_dim, device=self.device)
        if (z[:,1+self.n_assets:]<0).any():
            raise ValueError('Sampled negative inventory; reduce ic_var')
        return z

    def generate_trajectory(self, u, z0, nt, return_full_trajectory = False, check_admissibility = True):

        z = z0.reshape(-1, self.state_dim)
        h = (self.t_final - self.t_initial)/nt
        t_i = self.t_initial
        for i in range(nt):
            if hasattr(u, "forward"):
                current_u = u(z[:,:,i], t_i)
            else:
                current_u = u[:,:,i]
            current_u = current_u.reshape(-1,self.control_dim)
            if check_admissibility:
                Q = self._parts(z)[2]
                if (current_u<0).any() or (h*current_u>Q + 1e-7).any():
                    raise ValueError('Inadmissible selling rate: negative or would oversell inventory')
            dW = math.sqrt(h) * torch.randn(z.shape[0], self.noise_dim, dtype = z.dtype, device=z.device)
            noise_term = torch.bmm(self.compute_sigma(t_i,z, current_u), dW.unsqueeze(-1)).squeeze(-1)
            z[:,:,i+1] = z[:,:,i] + h*self.compute_f(t_i, z[:,:,i], current_u) + noise_term
            t_i = t_i + h
        return z if return_full_trajectory else z[:,:,-1]

    def plot_trajectories(self, z, filename="multi_asset_trajectories.png"):
        """Plot a rollout of shape [batch, 1 + 2*n_assets, nt + 1].

        Panels show cash, prices and inventories against time. Colours identify
        assets; multiple paths of an asset use the same colour. The inventory
        target is a soft terminal objective, not a guaranteed endpoint.
        """
        if z.ndim != 3 or z.shape[1] != self.state_dim:
            raise ValueError('Expected z with shape [batch, state_dim, nt + 1]')
        if z.shape[0] == 0 or z.shape[2] < 2:
            raise ValueError('Provide at least one path and two time points')

        z = z.detach().cpu()
        times = torch.linspace(self.t_initial, self.t_final, z.shape[2]).numpy()
        n = self.n_assets
        # State ordering: z = [X, S_1, ..., S_n, Q_1, ..., Q_n].
        X = z[:, 0, :].numpy()
        S = z[:, 1:1+n, :].numpy()
        Q = z[:, 1+n:1+2*n, :].numpy()
        colours = plt.get_cmap('tab10')
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

        for b in range(z.shape[0]):
            axes[0].plot(times, X[b], alpha=0.7,
                        label=f'Path {b+1}' if b < 5 else None)
            for j in range(n):
                label = f'Asset {j+1}' if b == 0 else None
                colour = colours(j % 10)
                axes[1].plot(times, S[b, j], color=colour,
                            alpha=0.7, label=label)
                axes[2].plot(times, Q[b, j], color=colour,
                            alpha=0.7, label=label)

        axes[2].axhline(0.0, color='black', linestyle='--', linewidth=1.5,
                        label='Inventory target = 0')
        axes[0].set_ylabel('Cash X(t)')
        axes[1].set_ylabel('Prices S_i(t)')
        axes[2].set_ylabel('Inventories Q_i(t)')
        axes[2].set_xlabel('Time t')
        axes[0].set_title(f'{self.soc_problem_name}: stochastic trajectories')
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', ncol=3, fontsize=8)

        fig.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

def main():
    from utils import GradientTester_Taylors as GradientTester

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    problem = MultiAssetLiquidation(batch_size=2, nt=250, device=device)
    taylor_checker = GradientTester()
    z0 = problem.sample_initial_condition()[0:1].double()
    u0 = torch.linspace(0.3, 1.2, problem.control_dim,
                        device=device, dtype=torch.float64).view(1, -1)
    t0 = z0.new_zeros(1)


    # actual = problem.compute_grad_f_z(t0, z0, u0).sum(dim=1)
    # expected = torch.zeros_like(z0)
    # expected[:, 1:1 + problem.n_assets] = u0
    # print('Max f_z difference:', (actual - expected).abs().max().item())

    print('Checking f_z')
    taylor_checker.check_gradient(
        cost_func=lambda z_: problem.compute_f(t0, z_, u0).sum(),
        grad_func=lambda z_: problem.compute_grad_f_z(t0, z_, u0).sum(dim=1),
        x=z0)

    print('Checking f_u')
    taylor_checker.check_gradient(
        cost_func=lambda u_: problem.compute_f(t0, z0, u_).sum(),
        grad_func=lambda u_: problem.compute_grad_f_u(t0, z0, u_).sum(dim=1),
        x=u0)

    print('Checking L_z')
    taylor_checker.check_gradient(
        cost_func=lambda z_: problem.compute_lagrangian(t0, z_, u0),
        grad_func=lambda z_: problem.compute_grad_L_z(t0, z_, u0), x=z0)
    print('Checking L_u (zero because L depends only on inventory)')
    taylor_checker.check_gradient(
        cost_func=lambda u_: problem.compute_lagrangian(t0, z0, u_),
        grad_func=lambda u_: problem.compute_grad_lagrangian(t0, z0, u_), x=u0)
    print('Checking G_z')
    taylor_checker.check_gradient(
        cost_func=lambda z_: problem.compute_G(z_),
        grad_func=lambda z_: problem.compute_grad_G_z(z_), x=z0)


if __name__ == '__main__':
    main()
