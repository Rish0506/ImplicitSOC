import torch
import numpy as np
from abc import ABC, abstractmethod

"""
B : batch_size, D = state_dim, M = control_dim, N : Number of time intervals, K = Brownian/noise dim 
Shapes: z = [B,D], u = [B,M], f = [B,D], sigma = [B,D,K], dW = [B,K], p = [B,D]
V_zz = [B,D,D], q = V_zz sigma = [B,D,K]
z_t = [B,D,N+1], u_t = [B,M,N], p_t = [B,D,N+1], dW_t = [B,K,N]

"""


class ImplicitSOC(ABC):
    """
    A general class for Implicit Optimal Control problems.
    This abstract base class provides the structure for solving optimal control problems
    using implicit methods.
    """
    
    def __init__(self, state_dim, control_dim, noise_dim, batch_size, t_initial, t_final, nt, 
                 alphaL, alphaG,  device='cpu', alphaHJB = [0.0,0.0], alphaadj = [0.0,0.0],
                 track_all_fp_iters = False, pen_pos=False): 
        """
        Initialize the Implicit Optimal Control problem.
        
        Args:
            state_dim (int): Dimension of the state vector
            control_dim (int): Dimension of the control vector
            noise_dim (int): Dimension of the noise vector
            batch_size (int): Batch size for trajectory optimization
            t_initial (float): Initial time
            t_final (float): Final time
            nt (int): Number of time steps
            device (str): Device to perform computation on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.t_initial = t_initial
        self.t_final = t_final
        self.nt = nt
        self.device = device
        self.h = (t_final - t_initial) / nt
        self.pen_pos = pen_pos

        self.soc_problem_name = ""

        # Loss function weights
        self.alphaL = alphaL  # Running cost weight
        self.alphaG = alphaG  # Terminal cost weight
        self.alphaHJB = alphaHJB  # HJB weight
        self.alphaadj = alphaadj  # adjoint weight
        self.use_HJB = True if (self.alphaHJB[0] + self.alphaHJB[1]) > 0.0 else False
        
        # Gradient tracking of fixed point iterations
        self.track_all_fp_iters = track_all_fp_iters

    @abstractmethod
    def compute_lagrangian(self, t, z, u):
        """
        Compute the Lagrangian (running cost).
        
        Args:
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Lagrangian values of shape (batch_size,)
        """
        pass
    
    @abstractmethod
    def compute_grad_lagrangian(self, t, z, u):
        """
        Compute the gradient of the Lagrangian with respect to control.
        
        Args:
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Gradient of Lagrangian of shape (batch_size, control_dim)
        """
        pass

    @abstractmethod
    def compute_grad_L_z(self, t, z, u):
        """
        Compute the gradient of the Lagrangian with respect to state.
 
        This is the source term in the stochastic adjoint recursion:
            dp = - (f_z^T p - L_z+sum_{j=1}^d sigma_z_j q_j)ds + q(s)dW(s)
 
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
 
        Returns:
            torch.Tensor: Gradient of Lagrangian w.r.t. z of shape (batch_size, state_dim)
        """
        pass

    @abstractmethod
    def compute_G(self, z):
        """
        Compute the terminal cost.
        
        Args:
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Terminal cost values of shape (batch_size,)
        """
        pass
        
    @abstractmethod
    def compute_sigma(self, t, z, u):
        """
        Compute the diffusion coefficient/matrix sigma(t, z, u).

        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)

        Returns:
            torch.Tensor: Diffusion matrix of shape
                          (batch_size, state_dim, noise_dim)
        """
        pass
        
    
    @abstractmethod
    def compute_f(self, t, z, u):
        """
        Compute the system dynamics.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Time derivative of z (dz/dt) of shape (batch_size, state_dim)
        """
        pass
    
    @abstractmethod
    def compute_grad_f_u(self, t, z, u):
        """
        Compute the gradient of the dynamics with respect to control.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Gradient of f w.r.t. u of shape (batch_size, control_dim, state_dim)
        """
        pass
    
    @abstractmethod
    def compute_grad_f_z(self, t, z, u):
        """
        Compute the gradient of the dynamics with respect to state.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Gradient of f w.r.t. z of shape (batch_size, state_dim, state_dim)
        """
        pass

    @abstractmethod
    def compute_grad_sigma_u(self, t, z, u):
        """
        [NEW] Compute the gradient of sigma with respect to control.
 
        Used in the stochastic adjoint Jacobian:
            (dsigma/du * dW)_{ij} = sum_k  [d sigma_{ik} / d u_j] * dW_k
 
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
        Returns:
            torch.Tensor: d sigma / d u of shape (batch_size, state_dim, noise_dim, control_dim)
        """
        pass

    @abstractmethod
    def compute_grad_sigma_z(self, t, z, u):
        """
        [NEW] Compute the gradient of sigma with respect to state.
 
        Used in the stochastic adjoint Jacobian:
            (dsigma/dz * dW)_{ij} = sum_k  [d sigma_{ik} / d z_j] * dW_k
 
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
        Returns:
            torch.Tensor: d sigma / d z of shape (batch_size, state_dim, noise_dim, state_dim)
        """
        pass

    @abstractmethod
    def compute_grad_G_z(self, t, z, u):
        """
        Compute the gradient of the dynamics with respect to state.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            
        Returns:
            torch.Tensor: Gradient of G w.r.t. z of shape (batch_size, state_dim, state_dim)
        """
        pass

    def sample_dW(self, z, sigma):
        """
        Sample one Brownian increment dW ~ N(0, h I) where h = Delta t for the current batch

        Args:
            z(torch.Tensor): Current state, shape (batch_size, state_dim).
            sigma(torch.Tensor): Diffusion matrix, shape (batch_size, state_dim, noise_dim).

        Returns:
            torch.Tensor: Brownian increment of shape (batch_size, noise_dim).
        """
        # Validates the 3D
        if sigma.dim() != 3:
            raise ValueError(
                "compute_sigma(t,z, u) must return a tensor of shape"
                "(batch_size, state_dim, noise_dim)"
            )
        noise_dim = sigma.shape[-1] 
        # Delta W = sqrt(h) \xi where \xi \in N(0,I)
        return torch.sqrt(torch.as_tensor(self.h, device=z.device, dtype=z.dtype)
        ) * torch.randn(z.shape[0], noise_dim, device=z.device, dtype=z.dtype)

    def compute_q(self, t, z, u, phi_net):
        """
        Compute the stochastic adjoint diffusion term q(t,z,u,p) = sigma^T p.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            p (torch.Tensor): Costate vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Adjoint diffusion term q of shape (batch_size, noise_dim)
        """

        sigma = self.compute_sigma(t, z, u)
        if sigma.dim() != 3:
            raise ValueError(
            "compute_sigma(t,z,u) must return a tensor of shape"
            "(batch_size, state_dim, noise_dim)"
            )
        z_hess = z.clone().detach().requires_grad_(True)
        # the value function is a tensor
        value_function = phi_net.getPhi(t, z_hess)
        grad_V = torch.autograd.grad(value_function.sum(), z_hess, create_graph=True, retain_graph=True)[0]
        # In order to use jacrev, we need to change grad_V in a function
        # Use of unsqueeze for avoiding dimension mismatch error
        grad_V_fun = lambda z_i: torch.autograd.grad(phi_net.getPhi(t, z_i.unsqueeze(0)).sum(), z_i, create_graph=True)[0]
        hessian_V = torch.func.vmap(torch.func.jacrev(grad_V_fun))(z_hess)
        # q_j = sum_i sigma_{i,j} * d2V/dz2_i, i.e. sigma^T hessian_V.
        q = torch.einsum('bij,bjk->bik', hessian_V, sigma)
        return grad_V, q

    # def compute_diffusion_term(self, t, z, u, phi_net):
    #     """
    #     Compute the stochastic HJB diffusion term 
    #     1/2 Tr[sigma sigma^T V_zz]
    #     The Hessian is evaluated through Hessian-vector products, so the
    #     full state_dim times state_dim Hessian is not explicitly formed.aa
    #     Args:
    #         t: Current time
    #         z: State tensor of shape (batch_size, state_dim).
    #         u: Control input tensor of shape (batch_size, control_dim).
    #         phi_net: Value-function network with getPhi(t,z).
    #     Returns:
    #         Tensor of shape (batch_size).
    #     """
    #     sigma = self.compute_sigma(t, z, u)
    #     if sigma.dim() != 3:
    #         raise ValueError(
    #             "compute_sigma(t,z,u) must return a tensor of shape"
    #             "(batch_size, state_dim, noise_dim)"
    #         )
    #     # Detaches z from existing computation graph and enables gradient tracking
    #     z_hess = z.detach().requires_grad_(True)
    #     # value function
    #     V = phi_net.getPhi(t, z_hess)
    #     # Gradient of value function
    #     grad_V = torch.autograd.grad(
    #         V.sum(),z_hess,create_graph=True,retain_graph=True,
    #     )[0]
    #     """
    #     We keep sigma fixed while taking the Hessian-vector product.
    #     This computes V_zz sigma_j, not a derivative of sigma itself
    #     """
    #     sigma = sigma.detach()
    #     # Initial allocation of zero vector of batch size
    #     diffusion_term = torch.zeros(z.shape[0], device = z.device, dtype = z.dtype)

    #     # Loop over columns of sigma along noise_dim ( then compute gradient treating sigma as constant)
    #     for j in range(sigma.shape[-1]):
    #         sigma_j = sigma[:,:,j]
    #         H_sigma_j = torch.autograd.grad(
    #             (grad_V*sigma_j).sum(), z_hess, create_graph=True, retain_graph=True,)[0]

    #     diffusion_term = diffusion_term + torch.sum(sigma_j * H_sigma_j, dim = 1)

    #     return diffusion_term

    def compute_H(self, t, z, u, p,phi_net, compute_general_H=False):
        """
        Compute the dynamic Hamiltonian H = L + p^T f.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            p (torch.Tensor): Costate vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Hamiltonian values of shape (batch_size,)
        """
        f_val = self.compute_f(t, z, u)
        L_val = self.compute_lagrangian(t, z, u)
        sigma_val = self.compute_sigma(t, z, u)
        _, q = self.compute_q(t, z, u, phi_net)

        inner_product_fp = torch.sum(p * f_val, dim=1)
        # Compute the trace term
        if compute_general_H:
            trace_term = (sigma_val * q).sum(dim=(1, 2))
        else:
            trace_term = 0.5*(sigma_val * q).sum(dim=(1, 2))

        # Compute Hamiltonian
        H_val = -L_val - inner_product_fp - trace_term

        return H_val
    
    # def compute_general_H(self, t, z, u, p):
    #     """
    #     Compute the generalized Hamiltonian H = L + p^T f.
        
    #     Args:
    #         t (torch.Tensor or float): Current time
    #         z (torch.Tensor): State vector of shape (batch_size, state_dim)
    #         u (torch.Tensor): Control input of shape (batch_size, control_dim)
    #         p (torch.Tensor): Costate vector of shape (batch_size, state_dim)
            
    #     Returns:
    #         torch.Tensor: Hamiltonian values of shape (batch_size,)
    #     """
    #     f_val = self.compute_f(t, z, u)
    #     grad_V, q = self.compute_q(t, z, u, u.p_net)
    #     sigma_val = self.compute_sigma(t, z, u)

    #     # Compute Lagrangian
    #     L_val = self.compute_lagrangian(t, z, u)

    #     # Compute inner product p^T f
    #     inner_product_fp = torch.sum(p * f_val, dim=1)
    #     # Compute the trace term
    #     trace_term =(sigma_val * q).sum(dim=(1, 2))

    #     # Compute Hamiltonian
    #     H_val = L_val + inner_product_fp + trace_term

    #     return H_val

    def compute_grad_H_u(self, t, z, u, p, phi_net):
        """
        Compute the gradient of the Hamiltonian with respect to control.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            p (torch.Tensor): Costate vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Gradient of H w.r.t. u of shape (batch_size, control_dim)
        """

        batch_size = z.shape[0]
        
        # Compute gradient of Lagrangian
        grad_term1 = self.compute_grad_lagrangian(t, z, u)
        
        # Compute gradient of dynamics
        grad_f_u_term = self.compute_grad_f_u(t, z, u)
        _ , q = self.compute_q(t, z, u, phi_net)
        # Setup : maximize H in u with (p,q) held fixed. sigma depend on u,
        # so dsigma/du stays in the gradient. q itself is not differentiated.
        q = q.detach()
        grad_sigma_u_term = self.compute_grad_sigma_u(t, z, u)
 
        # Compute gradient of p^T f w.r.t. u
        p = p.unsqueeze(-1)  # Shape: (batch_size, state_dim, 1)
        grad_term2 = torch.bmm(grad_f_u_term, p).view(batch_size, self.control_dim)
        grad_term3 = torch.einsum('bdkm,bdk->bm', grad_sigma_u_term, q).view(batch_size, self.control_dim)

        # Compute total gradient
        # H_u = -L_u - f_u^T p - <q, dsigma/du>
        grad_H_u_val = -grad_term1 - grad_term2 - grad_term3
        
        return grad_H_u_val

    # def compute_grad_H_u_(self, t, z, u, p, grad_f_u_term):
    #     """
    #     Compute the non-batch gradient of the Hamiltonian with respect to control.

    #     Args:
    #         t (torch.Tensor or float): Current time
    #         z (torch.Tensor): State vector of shape (state_dim,)
    #         u (torch.Tensor): Control input of shape (control_dim,)
    #         p (torch.Tensor): Costate vector of shape (state_dim,)
    #         grad_f_u_term: gradient of f with respect to u of shape (control_dim, state_dim)

    #     Returns:
    #         torch.Tensor: Gradient of H w.r.t. u of shape (control_dim,)
    #     """
    #     """ Compute gradient of Lagrangian """
    #     grad_term1 = -1.0*self.compute_grad_lagrangian_(t, z, u)

    #     """ Compute gradient of dynamics """
    #     grad_f_u_term = -1.0*self.compute_grad_f_u_(z, u, grad_f_u_term)

    #     """ Compute gradient of p^T f w.r.t. u """
    #     grad_term2 = torch.matmul(grad_f_u_term, p)
    #     """ Compute gradient of diffusion term w.r.t. u """
    #     grad_sigma_u_term = -1.0*self.compute_grad_sigma_u(t, z, u)
    #     grad_V, q = self.compute_q(t, z, u, u.p_net)
    #     grad_term3 = torch.matmul(grad_sigma_u_term, q)

    #     """ Compute total gradient """
    #     grad_H_u_val = grad_term1 + grad_term2 + grad_term3

    #     return grad_H_u_val

    def compute_grad_H_z(self, t, z, u, p, phi_net):
        """
        Compute the gradient of the Hamiltonian with respect to state.

        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            p (torch.Tensor): Costate vector of shape (batch_size, state_dim)

        Returns:
            torch.Tensor: Gradient of H w.r.t. z of shape (batch_size, state_dim)
        """
        batch_size = z.shape[0]
        # Compute gradient of dynamics w.r.t. state
        grad_L_z_term = self.compute_grad_L_z(t, z, u)
        grad_f_z_term = self.compute_grad_f_z(t, z, u)
        grad_sigma_z_term = self.compute_grad_sigma_z(t, z, u)
        _, q = self.compute_q(t, z, u, phi_net)
        # Compute gradient of p^T f w.r.t. z
        q = q.detach()
        # Set up H_z = -l_z - f_z^T p - <q, dsigma/dz>
        output = -grad_L_z_term - torch.bmm(grad_f_z_term, p.unsqueeze(-1)).view(batch_size, self.state_dim) - torch.einsum('bijk,bij->bk', grad_sigma_z_term, q).view(batch_size, self.state_dim)
        return output
    
    def solve_adjoint_eq(self, z, u, phi_net, dW=None):
        """
        Compute the adjoint equation for the Hamiltonian
        
        Args:
            t (torch.Tensor or float): Current time 
            z (torch.Tensor): State vector of shape (batch_dim, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            pT (torch.Tensor): Terminal costate vector of shape (batch_size, state_dim)

        """
        batch_size = z.shape[0]
        nt = z.shape[2] 
        p = torch.zeros(batch_size, self.state_dim, nt)
        p[:,:,-1] = self.compute_grad_G_z(z[:,:,-1])
        h = (self.t_final - self.t_initial)/nt
        t_i = self.t_final
        for i in range(nt-2, -1, -1):
            # if u is precomputed 
            if torch.is_tensor(u):
                assert nt == u.shape[2]
                u_i = u[:,:,i].view(batch_size, self.control_dim)
            # u is calculated using training.
            elif hasattr(u, 'forward'):
                u_i = u(z[:,:,i],t_i)
            H_z_i = self.compute_grad_H_z(t_i, z[:,:,i], u_i, p[:,:,i+1],phi_net)
            _ , q_i = self.compute_q(t_i, z[:,:,i], u_i, phi_net)
            dW_i = dW[:,:,i]
            q_dW_i = torch.bmm(q_i, dW_i.unsqueeze(-1)).squeeze(-1)
            p[:,:,i] = p[:,:,i+1] - h * H_z_i - q_dW_i
            t_i = t_i - h
        return p

    def compute_loss(self, u, z0, z_t = None, p_t = None, phi_t = None, jac_based=False, dW_t = None):
        """ 
        Z_{k+1} = Z_k + h f_k + sigma_k dW_k, dW_k ~ N(0,hI)
        -V_t - [L+V_z^T f] - 1/2 Tr[sigma sigma^T V_zz].
        Compute the total cost of a trajectory.
        
        Args:
            u (torch.Tensor or callable): Control inputs of shape (batch_size, control_dim, nt)
                                         or a policy function that takes (z, t) and returns control
            z0 (torch.Tensor): Initial states of shape (batch_size, state_dim)
            
        Returns:
            tuple: (total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin)
        """
        batch_size = z0.shape[0]
        # initialization of running cost and penalties enforcing the value fun dynamics
        running_cost = 0.0
        cHJB, cHJBfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        cadj, cadjfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        largest_grad_H_u = -1.0
        avg_grad_H_u = 0.0
        Identity = torch.eye(self.state_dim,device=z0.device,dtype=z0.dtype,).unsqueeze(0).expand(batch_size, -1, -1)
        z = z0
        t_i = 0
        # Integrate system using Euler's method
        if jac_based:
            # verifies tensor dimensions along time axis match discretization steps nt
            # nt control intervals and n1+1 discrete time grid nodes for z_k, p_k and value fun.
            assert self.nt == u.shape[2] and self.nt+1 == z_t.shape[2] \
            and self.nt+1 == p_t.shape[2] and self.nt+1 == phi_t.shape[2]
            # loop over time index i = 0, 1,..., n1-1
            for i in range(self.nt):
                # first extracts the given data u_i, z_{i+1}, p_i, p_{i+1}, Delta W_i
                u_i = u[:, :, i]
                z = z_t[:,:,i+1]
                gradPhi = p_t[:,:,i]
                #p_i = p_t[:,:,i]
                #p_next = p_t[:,:,i+1]
                dW_i = dW_t[:,:,i]
                _ , q_i = self.compute_q(t_i, z, u_i, u.p_net)

                # Accumulates step cost multiplied by h.
                running_cost = running_cost + self.h * self.compute_lagrangian(t_i, z, u_i)
                #cadj = cadj + torch.mean(p_i[:,:self.state_dim] -p_next[:,:self.state_dim]- self.h * self.compute_grad_H_z(t_i, z, u_i, p_i[:,:self.state_dim]) + self.h* torch.bmm(q_i, dW_i.unsqueeze(-1)).squeeze(-1))
                cadj = cadj + torch.mean(gradPhi[:,:self.state_dim] - self.h * self.compute_grad_H_z(t_i, z, u_i, gradPhi[:,:self.state_dim]) - torch.bmm(q_i, dW_i.unsqueeze(-1)).squeeze(-1))
                cHJB = cHJB + self.h*torch.mean(torch.linalg.vector_norm(phi_t[:,:,i].reshape(batch_size) - self.compute_H(t_i, z, u_i, gradPhi[:,:self.state_dim],compute_general_H=False).view(-1,1), ord = 2, dim = 1))

                t_i = t_i + self.h
            # Calculate terminal cost
            temp_final_cost = self.compute_G(z)
            terminal_cost = torch.mean(temp_final_cost)
            gradPhi = p_t[:,:,i+1]
            z_temp = z.view(batch_size*self.num, -1)
            z_target_temp = self.z_target.reshape(batch_size*self.num, -1)
            diff_p = (z_temp[:,:2] - z_target_temp[:,:2]).view(batch_size,-1)
            G = 0.5*torch.norm(diff_p, dim=1)**2            
            cadjfin = cadjfin + torch.mean(gradPhi[:,:self.state_dim] - self.compute_grad_G_z(z) )
            cHJBfin = torch.mean(torch.linalg.vector_norm(phi_t[:,:,i+1] - self.alphaG*temp_final_cost.view(-1, 1), ord=2, dim=1))

        else:
            if torch.is_tensor(u):
                assert self.nt == u.shape[2]

                # time loop 
                for i in range(self.nt):
                    u_i = u[:, :, i].view(batch_size, self.control_dim)
                    f_i = self.compute_f(t_i, z, u_i)
                    sigma_i = self.compute_sigma(t_i, z)
                    dW_i = self.sample_dW(z, sigma_i)
                    z = z + self.h * f_i + torch.bmm(sigma_i, dW_i.unsqueeze(-1)).squeeze(-1)
                    running_cost = running_cost + self.h * self.compute_lagrangian(t_i, z, u_i)
                    t_i = t_i + self.h

                temp_final_cost = self.compute_G(z)
                terminal_cost = torch.mean(temp_final_cost)


            elif hasattr(u, 'forward'):
                # determines if control is generated by parametrized policy network rather than a fixed tensor
                # Check if this is a direct control policy (no HJB computation needed)
                is_direct_control = getattr(u, 'is_direct_control', False)
                # loop in time i = 0, 1,2, ..., nt-1
                for i in range(self.nt):
                    u_i = u(z, t_i, track_all_fp_iters=self.track_all_fp_iters).view(batch_size, self.control_dim)
                    f_i = self.compute_f(t_i, z, u_i)
                    sigma_i = self.compute_sigma(t_i, z, u_i)
                    dW_i = self.sample_dW(z, sigma_i)
                    z = z + self.h * f_i + torch.bmm(sigma_i, dW_i.unsqueeze(-1)).squeeze(-1)
                    running_cost = running_cost + self.h * self.compute_lagrangian(t_i, z, u_i)

                    # Only compute HJB and adjoint for implicit control methods
                    if not is_direct_control:

                        z = z.to(dtype=next(u.p_net.parameters()).dtype)
                        # extract p_i \nabla_z V(t_i, z_i) 
                        gradPhi = u.p_net(t_i, z, full_grad=True)

                        _, q_i = self.compute_q(t_i, z, u_i, u.p_net)
                        cadj = cadj + torch.mean(gradPhi[:,:self.state_dim]- self.h * self.compute_grad_H_z(t_i, z, u_i, gradPhi[:,:self.state_dim], phi_net=u.p_net) - torch.bmm(q_i, dW_i.unsqueeze(-1)).squeeze(-1))
                        if hasattr(u.p_net, "getPhi"):
                            cHJB = cHJB + self.h*torch.mean(torch.linalg.vector_norm(gradPhi[:,-1:] - self.compute_H(t_i, z, u_i, gradPhi[:,:self.state_dim],u.p_net, compute_general_H=False).view(-1,1), ord = 2, dim = 1))
                        
                        grad_H_u = self.compute_grad_H_u(t_i, z, u_i, gradPhi[:,:self.state_dim], u.p_net)
                        max_norm_grad_H_u = torch.max(torch.linalg.vector_norm(grad_H_u, ord=2, dim=1)).item()
                        avg_grad_H_u += torch.mean(torch.linalg.vector_norm(grad_H_u, ord=2, dim=1)).item()
                        if max_norm_grad_H_u > largest_grad_H_u:
                            largest_grad_H_u = max_norm_grad_H_u
                    
                    t_i = t_i + self.h

                temp_final_cost = self.compute_G(z)
                terminal_cost = torch.mean(temp_final_cost) 

            # Only compute terminal HJB and adjoint for implicit control methods
                if not is_direct_control:

                    if self.pen_pos:
                        if self.oc_problem_name in ( "Double Integrator", "Multi Quadcopter"):
                            gradPhi_p = (-gradPhi[:,:self.state_dim].reshape(batch_size * self.num_agents, -1)[:, :3])
                            cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) + self.compute_grad_G_z(z))
                        elif self.oc_problem_name == "Multi Bicycle":
                            gradPhi_p = (-gradPhi[:,:self.state_dim].reshape(batch_size * self.num_agents, -1)[:, :2])
                            cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) + self.compute_grad_G_z(z))
                        elif self.oc_problem_name == "Single Quadcopter":
                            gradPhi_p = (-gradPhi[:,:self.state_dim].reshape(batch_size * self.num_agents, -1)[:, :3])
                            cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) + self.compute_grad_G_z(z))
                    else:
                        cadjfin = cadjfin + torch.mean(-gradPhi[:,:self.state_dim] + self.compute_grad_G_z(z) )

                    if hasattr(u.p_net, "getPhi"):                    
                        cHJBfin = torch.mean(torch.linalg.vector_norm(u.p_net.getPhi(t_i,z) - self.alphaG*temp_final_cost.view(-1, 1), ord=2, dim=1))
        
        # Calculate mean running cost
        running_cost = torch.mean(running_cost)
        
        # Calculate total cost
        total_cost = (self.alphaL * running_cost + self.alphaG * terminal_cost 
                      + self.alphaHJB[0] * cHJB + self.alphaHJB[1] * cHJBfin
                      + self.alphaadj[0] * cadj + self.alphaadj[1] * cadjfin)
        avg_grad_H_u = avg_grad_H_u / self.nt
        return total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, largest_grad_H_u, avg_grad_H_u
    
    def compute_loss_verify(self, u, z0, z_t = None, p_t = None, phi_t = None, dW_t = None, jac_based=False):
        """
        Compute the total cost of a trajectory as well as numerically verify certain 
        theoretical assumptions
        
        Args:
            u (torch.Tensor or callable): Control inputs of shape (batch_size, control_dim, nt)
                                         or a policy function that takes (z, t) and returns control
            z0 (torch.Tensor): Initial states of shape (batch_size, state_dim)
            
        Returns:
            tuple: (total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin)
        """
        batch_size = z0.shape[0]
        running_cost = 0.0
        cHJB, cHJBfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        cadj, cadjfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        smallest_M_sdval = torch.inf # Smallest singular value of M = dT/dtheta over all samples in batch and over all time steps 
        largest_M_sdval = -1.0 # Largest singular value of M over all samples in batch and over all time steps 
        smallest_lambda_min = torch.inf # Batchwise-largest largest eigenvalue of (MM^{T})^{-1} over all time steps 
        largest_lambda_max = -1.0 # Batchwise-smallest smallest eigenvalue of (MM^{T})^{-1} over all time steps
        avg_grad_T_u = 0.0
        largest_grad_T_u_batch = -1.0*torch.ones(batch_size, device=self.device) # Largest norm of grad of T with respect to u, for each sample
        largest_grad_H_u = -1.0
        avg_grad_H_u = 0.0

        z=z0
        t_i = 0.0
        # Integrate system using Euler's method
        if jac_based:
            assert self.nt == u.shape[2] and self.nt+1 == z_t.shape[2] \
            and self.nt+1 == p_t.shape[2] and self.nt+1 == phi_t.shape[2]
            for i in range(self.nt):
                u_i = u[:, :, i]
                z = z_t[:,:,i+1]
                gradPhi = p_t[:, :, i] 
                dW_i = dW_t[:,:,i]
                q_i = self.compute_q(t_i, z, u_i, phi_net=u.p_net)
                
                # Accumulates step cost multiplied by h.
                running_cost = running_cost + self.h * self.compute_lagrangian(t_i, z, u_i)
                cadj = cadj + torch.mean(gradPhi[:,:self.state_dim] - self.h * self.compute_grad_H_z(t_i, z, u_i, gradPhi[:,:self.state_dim], phi_net=u.pnet
                ) -  torch.bmm(q_i, dW_i.unsqueeze(-1)).squeeze(-1))
                cHJB = cHJB + torch.mean(phi_t[:,:,i].reshape(batch_size) - self.h*self.compute_H(t_i, z, u_i, gradPhi[:,:self.state_dim],phi_net=u.p_net,compute_general_H=False).view(-1,1))

                t_i = t_i + self.h

            z = z_t[:, :, -1]
            temp_final_cost = self.compute_G(z)
            terminal_cost = torch.mean(temp_final_cost)
            gradPhi = p_t[:,:,i+1]
            z_temp = z.view(batch_size*self.num, -1)
            z_target_temp = self.z_target.reshape(batch_size*self.num, -1)
            diff_p = (z_temp[:,:2] - z_target_temp[:,:2]).view(batch_size,-1)
            G = 0.5*torch.norm(diff_p, dim=1)**2            
            cadjfin = cadjfin + torch.mean(gradPhi[:,:self.state_dim] - self.compute_grad_G_z(z) )
            cHJBfin = torch.mean(torch.abs(phi_t[:,:,i+1] - temp_final_cost.view(-1, 1)))
        
        else:
            if torch.is_tensor(u):
                assert self.nt == u.shape[2]
                for i in range(self.nt):
                    u_i = u[:, :, i].view(batch_size, self.control_dim)
                    f_i = self.compute_f(t_i, z, u_i)
                    sigma_i = self.compute_sigma(t_i, z)
                    dW_i = self.sample_dW(z, sigma_i)
                    z = z + self.h * f_i + torch.bmm(sigma_i, dW_i.unsqueeze(-1)).squeeze(-1)
                    running_cost = running_cost + self.h * self.compute_lagrangian(t_i, z, u_i)
                    t_i = t_i + self.h
                temp_final_cost = self.compute_G(z)
                terminal_cost = torch.mean(temp_final_cost)

            elif hasattr(u, 'forward'):
                for i in range(self.nt):
                    u_i = u(z, t_i, track_all_fp_iters=self.track_all_fp_iters).view(batch_size, self.control_dim)
                    f_i = self.compute_f(t_i, z, u_i)
                    sigma_i = self.compute_sigma(t_i, z, u_i)
                    dW_i = self.sample_dW(z, sigma_i)
                    z = z + self.h * f_i + torch.bmm(sigma_i, dW_i.unsqueeze(-1)).squeeze(-1)
                    running_cost = running_cost + self.h * self.compute_lagrangian(t_i, z, u_i)                    
                    gradPhi = u.p_net(t_i, z, full_grad=True)

                    cadj = cadj + torch.mean(gradPhi[:,:self.state_dim]- self.h * self.compute_grad_H_z(t_i, z, u_i, gradPhi[:, :self.state_dim], phi_net=u.p_net) - torch.bmm(q_i, dW_i.unsqueeze(-1)).squeeze(-1))

                    if hasattr(u.p_net, "getPhi"):
                        cHJB = cHJB + torch.mean(u.p_net.getPhi(t_i,z) - self.h*self.compute_H(t_i, z, u_i, -gradPhi[:,:self.state_dim],phi_net=u.p_net,compute_general_H=False).view(-1,1))
                        

                    grad_H_u = self.compute_grad_H_u(t_i,z, u_i, gradPhi[:,:self.state_dim])
                    max_norm_grad_H_u = torch.max(torch.norm(grad_H_u, dim=1)).item()
                    avg_grad_H_u += torch.mean(torch.norm(grad_H_u, dim=1)).item()
                    if max_norm_grad_H_u > largest_grad_H_u:
                        largest_grad_H_u = max_norm_grad_H_u

                    # Verify Assumption 2 and Hypothesis of Lemma 1 in End-to-end
                    # training paper
                    M_theta, theta0, metadata = self.compute_grad_T_theta(u, z, t_i)
                    batch_sdvals = torch.linalg.svdvals(M_theta)
                    if torch.min(batch_sdvals[:, -1]).item() < smallest_M_sdval:
                        smallest_M_sdval = torch.min(batch_sdvals[:, -1]).item()
                    if torch.max(batch_sdvals[:, 0]).item() > largest_M_sdval:
                        largest_M_sdval = torch.max(batch_sdvals[:, 0]).item()
                    lambda_min = 1.0 / (batch_sdvals[:, 0] * batch_sdvals[:, 0])
                    if torch.min(lambda_min).item() < smallest_lambda_min:
                        smallest_lambda_min = torch.min(lambda_min).item()
                    lambda_max = 1.0 / (batch_sdvals[:, -1] * batch_sdvals[:, -1])
                    if torch.max(lambda_max).item() > largest_lambda_max:
                        largest_lambda_max = torch.max(lambda_max).item()

                    grad_T_u = self.compute_grad_T_u(u_i, z, t_i, gradPhi[:, :self.state_dim], u.alpha)
                    norm_grad_T_u = torch.linalg.matrix_norm(grad_T_u, ord=2, dim=(1, 2))
                    idx_max = torch.argwhere(norm_grad_T_u > largest_grad_T_u_batch).flatten()
                    largest_grad_T_u_batch[idx_max] = norm_grad_T_u[idx_max]
                    avg_grad_T_u += torch.mean(norm_grad_T_u).item()

                    ti = ti + self.h

                # Calculate terminal cost
                temp_final_cost = self.compute_G(z)
                terminal_cost = torch.mean(temp_final_cost)

                if self.pen_pos:
                    if (self.oc_problem_name == "Double Integrator") or (self.oc_problem_name == "Multi Quadcopter"):
                        gradPhi_p = (gradPhi[:, :self.state_dim].reshape(batch_size * self.num_agents, -1))[:, :3]
                        cadjfin = torch.mean(gradPhi_p.reshape(batch_size, -1) - self.compute_grad_G_z(z))
                    elif self.oc_problem_name == "Multi Bicycle":
                        gradPhi_p = (gradPhi[:, :self.state_dim].reshape(batch_size * self.num_agents, -1))[:, :2]
                        cadjfin = torch.mean(gradPhi_p.reshape(batch_size, -1) - self.compute_grad_G_z(z))
                    elif self.oc_problem_name == "Single Quadcopter":
                        gradPhi_p = (gradPhi[:, :self.state_dim].reshape(batch_size, -1))[:, :3]
                        cadjfin = torch.mean(gradPhi_p.reshape(batch_size, -1) - self.compute_grad_G_z(z))
                else:
                    cadjfin = cadjfin + torch.mean(gradPhi[:, :self.state_dim] - self.compute_grad_G_z(z))

                if hasattr(u.p_net, "getPhi"):
                    cHJBfin = torch.mean(torch.linalg.vector_norm(u.p_net.getPhi(t_i, z) - self.alphaG * temp_final_cost.view(-1, 1), ord=2, dim=1))
    
        # Calculate mean running cost
        running_cost = torch.mean(running_cost)
        
        # Calculate total cost
        total_cost = (self.alphaL * running_cost + self.alphaG * terminal_cost 
                      + self.alphaHJB[0] * cHJB + self.alphaHJB[1] * cHJBfin
                      + self.alphaadj[0] * cadj + self.alphaadj[1] * cadjfin)

        # Verify assumptions
        avg_grad_H_u = avg_grad_H_u / self.nt
        avg_grad_T_u = avg_grad_T_u / self.nt
        sd_grad_T_u = torch.std(largest_grad_T_u_batch).item()
        largest_grad_T_u = torch.max(largest_grad_T_u_batch).item()

        return total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, largest_grad_H_u, avg_grad_H_u, smallest_M_sdval, largest_M_sdval, smallest_lambda_min, largest_lambda_max, largest_grad_T_u, avg_grad_T_u, sd_grad_T_u
    
    
    def compute_grad_T_theta(self, model, z, ti, create_graph=False):
        """
        Compute the Jacobian of the model output w.r.t. all parameters, treating the
        parameters theta as a single flattened tensor.

        Args:
            model (torch.nn.Module):
                Your network. It will be run in training mode so that gradients flow
                through the final differentiable operations.
            z (torch.Tensor):
                Current state
            ti (torch.float):
                Current time
            create_graph (bool, default=False):
                If True, build a graph that allows higher-order derivatives of J.
        Returns:
            J (torch.Tensor):
                Full Jacobian with shape (batch, out_dim, P), where P is the number
                of scalar parameters.
            theta0 (torch.Tensor):
                The flattened parameter vector at which J is evaluated, requires_grad=True.
            meta (dict):
                Metadata to reconstruct parameter structure:
                - names:  list of parameter names (ordered)
                - shapes: list of torch.Size for each parameter
                - idx:    1D tensor of cumulative indices for slicing theta
                Also includes:
                - unflatten(theta): function to map flat theta back to a {name: tensor} dict.
        """
        model.train()
        # Named parameters and buffers (buffers may be empty; kept for generality)
        params_dict = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        names  = list(params_dict.keys())
        shapes = [p.shape for p in params_dict.values()]
        sizes  = [p.numel() for p in params_dict.values()]
        idx = torch.tensor([0] + sizes, device=next(model.parameters()).device).cumsum(0)

        def pack(pdict):
            return torch.cat([p.reshape(-1) for p in pdict.values()])

        def unflatten(theta):
            out = {}
            for i, k in enumerate(names):
                start, end = idx[i].item(), idx[i+1].item()
                out[k] = theta[start:end].view(shapes[i])
            return out

        # Flattened parameter vector theta
        theta0 = pack(params_dict).detach().requires_grad_(True)

        # Wrap the model so theta is the differentiable argument
        def T_of_theta(theta, z, ti):
            pdict = unflatten(theta)
            y = torch.func.functional_call(model, (pdict, buffers), args=(z, ti))
            return y  # shape: (batch, out_dim)

        # jacrev returns J with shape (*y_shape, *theta_shape)
        J = torch.func.jacrev(lambda th: T_of_theta(th, z, ti), has_aux=False)(theta0)

        if create_graph:
            # ensure graph is retained for higher-order derivatives
            J.retain_grad()

        meta = {
            "names": names,
            "shapes": shapes,
            "idx": idx,
            "unflatten": unflatten,
        }
        return J, theta0, meta
    
    def compute_grad_T_u(self, u, z, t, grad_phi, alpha, create_graph=False):
        """
        Compute J(u) = dT_theta/du for a batch of inputs u.

        Parameters
        ----------
        u : torch.Tensor
            Either shape [m] or [B, m]. Will compute a m x m Jacobian per sample.
            u should be floating and on the same device/dtype as used by T_theta.
        z : Any
            Additional input to T_theta (fixed during differentiation).
        t : Any
            Additional input to T_theta (fixed during differentiation).
        grad_phi : Any
            Additional input to T_theta (fixed during differentiation).
        alpha: torch.float
            Additional input to T_theta (fixed during differentiation).
        create_graph : bool
            If True, builds a graph for higher-order derivatives.

        Returns
        -------
        torch.Tensor
            If u is [m], returns [m, m].
            If u is [B, m], returns [B, m, m].

        Notes
        -----
        - Assumes a globally available callable `T_theta(u, z, t)` that
          maps a single u:[m] → [m].
        - Differentiates w.r.t. u only (theta is treated as a constant here).
        """
        if u.ndim == 1:
            # Single sample: shape [p]
            u_single = u.detach().requires_grad_(True)

            def _T_single(u_vec):
                # T_theta should return shape [p]
                return u_vec + alpha*self.compute_grad_H_u(t, z, u_vec, grad_phi)

            # jacrev computes d(T)/d(u) with respect to input u
            J = torch.func.jacrev(_T_single)(u_single)
            if create_graph:
                # The jacrev above already respects create_graph semantics via autograd;
                # but we ensure the requires_grad chain is kept if requested.
                J = J.clone()
            return J  # [p, p]
        elif u.ndim == 2:
            # Batch: shape [B, p]
            B, p = u.shape
            u_batch = u.detach().requires_grad_(True)

            def _T_single(u_vec, z_vec, p_vec, grad_f_u_term):
                return u_vec + alpha*self.compute_grad_H_u_(t, z_vec, u_vec, p_vec, grad_f_u_term)

            # Vectorized jacobian across batch
            # Note: each vmap call needs (control_dim, state_dim) not (B, control_dim, state_dim)
            grad_f_u = torch.zeros(B, p, z.shape[1], device=self.device)
            J_batched = torch.func.vmap(torch.func.jacrev(_T_single), in_dims=(0,0,0,0))(u_batch, z, grad_phi, grad_f_u)  # [B, p, p]
            if create_graph:
                J_batched = J_batched.clone()
            return J_batched

        else:
            raise ValueError(f"`u` must have shape [p] or [B, p], got {tuple(u.shape)}.")

