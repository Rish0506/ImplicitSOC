import torch
import numpy as np
from abc import ABC, abstractmethod
from utils import GradientTester
import matplotlib.pyplot as plt

class ImplicitOC(ABC):
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

        self.oc_problem_name = ""

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
        [NEW] Compute the gradient of the Lagrangian with respect to state.
 
        This is the source term in the stochastic adjoint recursion:
            p_n = dt * dL/dz_n  +  (...)^T * p_{n+1}
 
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
    def compute_sigma(self, t, z):
        """
        Compute the diffusion coefficient/matrix sigma(t, z).

        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)

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
    def compute_grad_sigma_z(self, t, z):
        """
        [NEW] Compute the gradient of sigma with respect to state.
 
        Used in the stochastic adjoint Jacobian:
            (dsigma/dz * dW)_{ij} = sum_k  [d sigma_{ik} / d z_j] * dW_k
 
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
 
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
    
    def compute_general_H(self, t, z, u, p):
        """
        Compute the generalized Hamiltonian H = L + p^T f.
        
        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (batch_size, state_dim)
            u (torch.Tensor): Control input of shape (batch_size, control_dim)
            p (torch.Tensor): Costate vector of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Hamiltonian values of shape (batch_size,)
        """
        f_val = self.compute_f(t, z, u)
        
        # Compute Lagrangian
        L_val = self.compute_lagrangian(t, z, u)
        
        # Compute inner product p^T f
        inner_product = torch.sum(p * f_val, dim=1)
        
        # Compute Hamiltonian
        H_val = L_val + inner_product
        
        return H_val

    def compute_grad_H_u(self, t, z, u, p):
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
        
        # Compute gradient of p^T f w.r.t. u
        p = p.unsqueeze(-1)  # Shape: (batch_size, state_dim, 1)
        grad_term2 = torch.bmm(grad_f_u_term, p).view(batch_size, self.control_dim)
        
        # Compute total gradient
        grad_H_u_val = grad_term1 + grad_term2
        
        return grad_H_u_val

    def compute_grad_H_u_(self, t, z, u, p, grad_f_u_term):
        """
        Compute the non-batch gradient of the Hamiltonian with respect to control.

        Args:
            t (torch.Tensor or float): Current time
            z (torch.Tensor): State vector of shape (state_dim,)
            u (torch.Tensor): Control input of shape (control_dim,)
            p (torch.Tensor): Costate vector of shape (state_dim,)
            grad_f_u_term: gradient of f with respect to u of shape (control_dim, state_dim)

        Returns:
            torch.Tensor: Gradient of H w.r.t. u of shape (control_dim,)
        """
        """ Compute gradient of Lagrangian """
        grad_term1 = -1.0*self.compute_grad_lagrangian_(t, z, u)

        """ Compute gradient of dynamics """
        grad_f_u_term = -1.0*self.compute_grad_f_u_(z, u, grad_f_u_term)

        """ Compute gradient of p^T f w.r.t. u """
        grad_term2 = torch.matmul(grad_f_u_term, p)

        """ Compute total gradient """
        grad_H_u_val = grad_term1 + grad_term2

        return grad_H_u_val

    def compute_grad_H_z(self, t, z, u, p)
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
        grad_H_z_term = self.compute_grad_L_z(t, z, u)
        grad_f_z_term = self.compute_grad_f_z(t, z, u)
        # Compute gradient of p^T f w.r.t. z
        p = p.unsqueeze(-1)  # Shape: (batch_size, state_dim, 1)
        output = grad_H_z_term + torch.bmm(grad_f_z_term, p).view(batch_size, self.state_dim)
        return output

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
                "compute_sigma(t,z) must return a tensor of shape"
                "(batch_size, state_dim, noise_dim)"
            )
        noise_dim = sigma.shape[-1] 
        # Delta W = sqrt(h) \xi where \xi \in N(0,I)
        return torch.sqrt(
            torch.as_tensor(self.h, device=z.device, dtype=z.dtype)
        ) * torch.randn(
            z.shape[0], noise_dim, device=z.device, dtype=z.dtype
        )

    def compute_diffusion_term(self, t, z, phi_net):
        """
        Compute the stochastic HJB diffusion term 
        1/2 Tr[sigma sigma^T V_zz]
        The Hessian is evaluated through Hessian-vector products, so the
        full state_dim \times state_dim Hessian is not explicitly formed.aa
        Args:
            t: Current time
            z: State tensor of shape (batch_size, state_dim).
            phi_net: Value-function network with getPhi(t,z).
        Returns:
            Tensor of shape (batch_size).
        """
        if sigma.dim() != 3:
            raise ValueError(
                "compute_sigma(t,z) must return a tensor of shape"
                "(batch_size, state_dim, noise_dim)"
            )
        # Detaches z from existing computation graph and enables gradient tracking
        z_hess = z.detach().requires_grad_(True)
        # value function
        V = phi_net.getPhi(t, z_hess)
        # Gradient of value function
        grad_V = torch.autograd.grad(
            V.sum(),z_hess,create_graph=True,retain_graph=True,
        )[0]
        """
        We keep sigma fixed while taking the Hessian-vector product.
        This computes V_zz sigma_j, not a derivative of sigma itself
        """
        sigma = sigma.detach()
        # Initial allocation of zero vector of batch size
        diffusion_term = torch.zeros(z.shape[0], device = z.device, dtype = z.dtype)

        # Loop over columns of sigma along noise_dim ( then compute gradient treating sigma as constant)
        for j in range(sigma.shape[-1]):
            sigma_j = sigma[:,:,j]
            H_sigma_j = torch.autograd.grad(
                (grad_V*sigma_j).sum(), z_hess, create_graph=True, retain_graph=True,
            )[0]

        diffusion_term = diffusion_term + torch.sum(
            sigma_j * H_sigma_j, dim = 1
        )

        return 0.5 * diffusion_term

    def solve_adjoint_eq(self, z, u, dW=None, du_dz=None):
        """
        solve p_k = \Delta t L_z +(I+ Delta t f_z + sigma_z dW)^T p_{k+1}
        """
        batch_size = z.shape[0]
        nt = z.shape[2] - 1
        p = torch.zeros(batch_size, self.state_dim, nt+1, device=z.devide, dtype = z.dtype)
        p[:,:,-1] = self.compute_grad_G_z(z[:,:,-1])
        I = torch.eye(self.state_dim, device = z.device, dtype = z.dtype).unsqueeze(0)
        for i in range(nt, -1, -1, -1):
            t_i = self.t_initial + i* self.h
            z_i = z[:,:,i]
            if torch.is_tensor(u):
                current_u = u[:,:,i]
            else:
                current_u = u(z_i, t_i)
            L_z = self.compute_grad_L_z(t_i, z_i, current_u)
            f_z = self.compute_grad_f_z(t_i, z_i, current_u)
            sigma_z = self.compute_grad_sigma_z(t_i, z_i)
            sigma_z_dW = torch.einsum(sigma_z, dW)
            A = I + self.h * f_z + sigma_z_dW
            p[:,:,i] = (self.h*L_z + torch.bmm(A.transpose(1,2),p[:,:,i+1].unsqueeze(-1)).squeeze(-1))

        return p

    def compute_loss(self, u, z0, z_t = None, p_t = None, phi_t = None, jac_based=False,
                    dW_t = None, du_dz_t = None):
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
        running_cost = torch.zeros(batch_size, device=z0.device, dtype=z0.dtype)
        cHJB, cHJBfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        cadj, cadjfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        largest_grad_H_u = -1.0
        avg_grad_H_u = 0.0
        t_i = self.t_initial
        # Integrate system using Euler's method
        if jac_based:
            # verifies tensor dimensions along time axis match discretization steps nt
            # nt control intervals and n1+1 discrete time grid nodes for z_k, p_k and value fun.
            assert self.nt == u.shape[2] and self.nt+1 == z_t.shape[2] \
            and self.nt+1 == p_t.shape[2] and self.nt+1 == phi_t.shape[2]
            # loop over time index i = 0, 1,..., n1-1
            for i in range(self.nt):
                # first extracts the given data u_i, z_{i+1}, p_i, p_{i+1}, Delta W_i
                current_u = u[:, :, i]
                z = z_t[:,:,i+1]
                gradPhi = p_t[:,:,i]
                # cost L(ti, z_{i+1},u_i)
                L = self.compute_lagrangian(t_i, z, current_u)
                # Accumulates step cost multiplied by h.
                running_cost = running_cost + self.h * L_k

                """Use of Ito formula gives us the expression for dV where V(t,z_t)"""
                # Gives sigma(ti, z_k) \Delta W_i 
                sigma_dW_k = torch.bmm(sigma_k, dW_k.unsqueeze(-1)).squeeze(-1)
                # V_i+1 - V_i + h.L_i - p_i^T(sigma_i \Delt w_i)
                value_residual = (phi_t[:,:,i+1].reshape(batch_size)-phi_t[:,:,i].reshape(batch_size)
                                 + self.h * L_k - torch.sum(p_k * sigma_dW_k, dim = 1))
                # Accumulate 
                cHJB = cHJB + self.h * torch.mean(value_residual.square())
                
                # D_z u at step i
                policy_jacobian = policy_jacobian = ( du_dz_t[:, :, :, i]
                    if du_dz_t is not None else None
                )
                
                # recalling terms in adjoint calculation
                ajoint_drift_k, beta_k, details = (
                    self.compute_necessary_adjoint_terms(
                        ti, z_k, current_u, p_k, policy_jacobian=policy_jacobian
                        include_policy_jacobian = du_dz_t is not None, return_details = True,
                    )
                )
                martingale_increment_k = torch.bmm(beta_k, dW_k.unsqueeze(-1)).squeeze(-1)
                # residual for adjoint
                adjoint_residual = (p_next-p_k + self.h * adjoint_drift_k + martingale_increment_k)
                cadj = cadj + self.h * torch.mean(adjoint_residual.square().sum(dim=1))

                # Optimality analysis
                # H_u means nabla_z H(ti,z_i, u_i,p_i)
                H_u_k = details["H_u"]
                # \|nabla_u H\|_2
                grad_norm = torch.linalg.vector_norm(H_u_k, dim=1)
                # this tracks max norm and sum_i E[\|nabla_u H\|_2]
                largest_grad_H_u = max(largest_grad_H_u, torch.max(grad_norm).item())
                avg_grad_H_u += torch.mean(grad_norm).item()
                ti += self.h
                
            # Calculate terminal cost
            
            z = z_t[:,:,-1] # z(T)
            terminal_values = self.compute_G(z)  # G(z_T)
            terminal_cost = torch.mean(terminal_values) # E[G(z_T)]
            grad_G = self.compute_grad_G_z(z) # \nabla_z G(z_T)         
            # Loss for adjoint state E[\|p_T-\nalba_z G(z_T)\|^2_2]
            cadjfin = cadjfin + torch.mean((p_t[:,:,-1] - grad_G).square().sum(dim=1) )
            # Terminal boundary condition loss for value function E[|V(T,z_T)-\alpha_G G(z_T)|^2]
            cHJBfin = torch.mean((phi_t[:,:,-1].reshape(batch_size)-self.alphaG*terminal_values).square())
        
        elif torch.is_tensor(u):
            assert self.nt == u.shape[2]
            z = z0
            # time loop 
            for i in range(self.nt):
                current_u = u[:, :, i].view(batch_size, self.control_dim)
                running_cost = running_cost + self.h * self.compute_lagrangian(ti, z, current_u)
                f_val = self.compute_f(ti, z, current_u)
                sigma_val = self.compute_sigma(ti, z)
                dW = self.sample_dW(z, sigma_val)
                z = z + self.h * f_val + torch.bmm(sigma_val, dW.unsqueeze(-1)).squeeze(-1)
                ti += self.h

            terminal_values = self.compute_G(z)
            terminal_cost = torch.mean(terminal_values)
                    

        elif hasattr(u, 'forward'):
            # determines if control is generated by parametrized policy network rather than a fixed tensor
            # Check if this is a direct control policy (no HJB computation needed)
            is_direct_control = getattr(u, 'is_direct_control', False)
            # loop in time i = 0, 1,2, ..., nt-1
            for i in range(self.nt):
                z_k = z
                t_k = ti
                # computes u_k = pi_{\theta}(z_k,t_k)
                current_u = u(z_k, t_k, track_all_fp_iters=self.track_all_fp_iters).view(batch_size, self.control_dim)
                running_cost = running_cost + self.h * self.compute_lagrangian(t_k, z_k, current_u)
                f_val = self.compute_f(t_k,z_k, current_u)
                sigma_val = self.compute_sigma(t_k,z_k)

                # Only compute HJB and adjoint for implicit control methods
                if not is_direct_control:
                    # extract p_k \nabla_z V(t_k, z_k) 
                    gradPhi_k = u.p_net(t_k, z_k, full_grad=True)
                    p_k = gradPhi_k[:, :self.state_dim]
                    H_val = self.compute_general_H(t_k, z_k, current_u, p_k)
                    diffusion_term = self.compute_diffusion_term(
                        t_k, z_k, u.p_net,sigma=sigma_val,)
                    # V_t - H(t,z,u,V_z) - 1/2Tr(sigma sigma^T V_zz)
                    hjb_residual = ( gradPhi_k[:, -1] - H_val - diffusion_term)
                    cHJB = cHJB + self.h * torch.mean(hjb_residual.square())

                    adjoint_drift_k, beta_k, details = (
                        self.compute_necessary_adjoint_terms(
                            t_k,z_k,current_u,p_k,include_policy_jacobian=True,
                            return_details=True,
                        )
                    )
                    H_u_k = details["H_u"]
                    grad_norm = torch.linalg.vector_norm(H_u_k, dim=1)
                    largest_grad_H_u = max(largest_grad_H_u, torch.max(grad_norm).item(),)
                    avg_grad_H_u += torch.mean(grad_norm).item()

                dW = self.sample_dW(z_k, sigma_val)
                z_next = z_k + self.h * f_val + torch.bmm(sigma_val, dW.unsqueeze(-1)).squeeze(-1)
                t_next = t_k + self.h
                
                if not is_direct_control:
                    p_next = u.p_net(t_next, z_next, full_grad=True)[:, :self.state_dim]
                    martingale_increment = torch.bmm(beta_k, dW.unsqueeze(-1)).squeeze(-1)
                    adjoint_residual = (p_next- p_k+ self.h * adjoint_drift_k+ martingale_increment)
                    cadj = cadj + self.h * torch.mean(adjoint_residual.square().sum(dim=1))

                z = z_next
                ti = t_next

            # Calculate terminal cost
            terminal_values = self.compute_G(z)
            terminal_cost = torch.mean(terminal_values)

            # Only compute terminal HJB and adjoint for implicit control methods
            if not is_direct_control:
                gradPhi_T = u.p_net(ti, z, full_grad=True)
                p_T = gradPhi_T[:, :self.state_dim]
                grad_G = self.compute_grad_G_z(z)
                if self.pen_pos:
                    if self.oc_problem_name in ( "Double Integrator", "Multi Quadcopter"):
                        p_compare = (p_T.reshape(batch_size * self.num_agents, -1)[:, :3].reshape(batch_size, -1)
                        )
                    elif self.oc_problem_name == "Multi Bicycle":
                        p_compare = ( p_T.reshape( batch_size * self.num_agents, -1
                            )[:, :2].reshape(batch_size, -1)
                        )
                    elif self.oc_problem_name == "Single Quadcopter":
                        p_compare = p_T.reshape(batch_size, -1)[:, :3]
                    else:
                        p_compare = p_T
                else:
                    p_compare = p_T

                cadjfin = torch.mean((p_compare - grad_G).square().sum(dim=1))
                V_T = u.p_net.getPhi(ti, z).reshape(batch_size)
                cHJBfin = torch.mean((V_T - self.alphaG * terminal_values).square())
        else:
            raise TypeError("u must be a control tensor or a policy callable.")
        
        # Calculate mean running cost
        running_cost = torch.mean(running_cost)
        
        # Calculate total cost
        total_cost = (self.alphaL * running_cost + self.alphaG * terminal_cost 
                      + self.alphaHJB[0] * cHJB + self.alphaHJB[1] * cHJBfin
                      + self.alphaadj[0] * cadj + self.alphaadj[1] * cadjfin)
        avg_grad_H_u = avg_grad_H_u / self.nt
        return total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, largest_grad_H_u, avg_grad_H_u
    
    def compute_loss_verify(self, u, z0, z_t = None, p_t = None, phi_t = None, dW_t = None, du_dz_t= None, jac_based=False):
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
        
        ti = self.t_initial
        # Integrate system using Euler's method
        if jac_based:
            assert self.nt == u.shape[2] and self.nt+1 == z_t.shape[2] \
            and self.nt+1 == p_t.shape[2] and self.nt+1 == phi_t.shape[2]
            for i in range(self.nt):
                current_u = u[:, :, i]
                z_k = z_t[:,:,i+1]
                p_k = p_t[:, :, i]
                p_next = p_t[:, :, i + 1]
                dW_k = dW_t[:, :, i]
                sigma_k = self.compute_sigma(ti, z_k)
                L_k = self.compute_lagrangian(ti, z_k, current_u)
                running_cost = running_cost + self.h * L_k
                sigma_dW_k = torch.bmm(sigma_k, dW_k.unsqueeze(-1)).squeeze(-1)
                value_residual = (phi_t[:, :, i + 1].reshape(batch_size)
                    - phi_t[:, :, i].reshape(batch_size)+ self.h * L_k
                    - torch.sum(p_k * sigma_dW_k, dim=1))
                cHJB = cHJB + self.h * torch.mean(value_residual.square())

                policy_jacobian = (du_dz_t[:, :, :, i] if du_dz_t is not None else None)
                adjoint_drift_k, beta_k, details = (
                    self.compute_necessary_adjoint_terms(
                        ti,z_k,current_u,p_k,policy_jacobian=policy_jacobian,
                        include_policy_jacobian=du_dz_t is not None,return_details=True,)
                )
                martingale_increment_k = torch.bmm(beta_k, dW_k.unsqueeze(-1)).squeeze(-1)
                adjoint_residual = (p_next- p_k+ self.h * adjoint_drift_k+ martingale_increment_k)
                cadj = cadj + self.h * torch.mean(adjoint_residual.square().sum(dim=1))

                H_u_k = details["H_u"]
                grad_norm = torch.linalg.vector_norm(H_u_k, dim=1)
                largest_grad_H_u = max(largest_grad_H_u, torch.max(grad_norm).item())
                avg_grad_H_u += torch.mean(grad_norm).item()
                ti += self.h

            z = z_t[:, :, -1]
            terminal_values = self.compute_G(z)
            terminal_cost = torch.mean(terminal_values)
            grad_G = self.compute_grad_G_z(z)
            cadjfin = torch.mean((p_t[:, :, -1] - grad_G).square().sum(dim=1))
            cHJBfin = torch.mean((phi_t[:, :, -1].reshape(batch_size)- self.alphaG * terminal_values).square())
        
        else:    
            if torch.is_tensor(u):
                assert self.nt == u.shape[2]
                z=z0
                for i in range(self.nt):
                    current_u = u[:, :, i].view(batch_size, self.control_dim)
                    running_cost = running_cost + self.h * self.compute_lagrangian(ti, z, current_u)
                    f_val = self.compute_f(ti, z, current_u)
                sigma_val = self.compute_sigma(ti, z)
                dW = self.sample_dW(z, sigma_val)
                z = z + self.h * f_val + torch.bmm(sigma_val, dW.unsqueeze(-1)).squeeze(-1)
                ti += self.h
            terminal_values = self.compute_G(z)
            terminal_cost = torch.mean(terminal_values)
    
            elif hasattr(u, 'forward'):
                for i in range(self.nt):
                    z_k = z
                    t_k = ti
                    current_u = u(z_k, t_k, track_all_fp_iters=self.track_all_fp_iters).view(batch_size, self.control_dim)
                    running_cost = running_cost + self.h * self.compute_lagrangian(t_k, z_k, current_u)
                    gradPhi_k = u.p_net(t_k, z_k, full_grad=True)
                    p_k = gradPhi_k[:, :self.state_dim]
                    
                    if hasattr(u.p_net, "getPhi"):
                        # double check sign
                        sigma_val = self.compute_sigma(t_k,z_k)
                        diffusion_term = self.compute_diffusion_term(t_k,z_k,u.p_net, sigma=sigma_val)
                        H_val = self.compute_general_H(t_k, z_k, current_u, p_k)
                        hjb_residual = gradPhi_k[:,-1] - H_val - diffusion_term
                        cHJB = cHJB + torch.mean(hjb_residual.square())
                    
                    grad_H_u = details["H_u"]
                    max_norm_grad_H_u = torch.max(torch.norm(grad_H_u, dim=1)).item()
                    avg_grad_H_u += torch.mean(torch.norm(grad_H_u, dim=1)).item()
                    if max_norm_grad_H_u > largest_grad_H_u:
                        largest_grad_H_u = max_norm_grad_H_u

                    # Verify Assumption 2 and Hypothesis of Lemma 1 in End-to-end
                    # training paper
                    M_theta, theta0, metadata = self.compute_grad_T_theta(u, z, ti)
                    batch_sdvals = torch.linalg.svdvals(M_theta)
                    if torch.min(batch_sdvals[:,-1]).item() < smallest_M_sdval:
                        smallest_M_sdval = torch.min(batch_sdvals[:,-1]).item()
                    if torch.max(batch_sdvals[:,0]).item() > largest_M_sdval:
                        largest_M_sdval = torch.max(batch_sdvals[:,0]).item()
                    lambda_min = 1.0/(batch_sdvals[:,0]*batch_sdvals[:,0])
                    if torch.min(lambda_min).item() < smallest_lambda_min:
                        smallest_lambda_min = torch.min(lambda_min).item()
                    lambda_max = 1.0/(batch_sdvals[:,-1]*batch_sdvals[:,-1])
                    if torch.max(lambda_max).item() > largest_lambda_max:
                        largest_lambda_max = torch.max(lambda_max).item()
                    grad_T_u = self.compute_grad_T_u(current_u, z, ti, gradPhi[:,:self.state_dim], u.alpha)
                    norm_grad_T_u = torch.linalg.matrix_norm(grad_T_u, ord=2, dim=(1,2))
                    idx_max = torch.argwhere(norm_grad_T_u > largest_grad_T_u_batch).flatten()
                    largest_grad_T_u_batch[idx_max] = norm_grad_T_u[idx_max]
                    avg_grad_T_u += torch.mean(norm_grad_T_u).item()

                    ti = ti + self.h

                # Calculate terminal cost
                temp_final_cost = self.compute_G(z)
                terminal_cost = torch.mean(temp_final_cost)

                if self.pen_pos:
                    if (self.oc_problem_name == "Double Integrator") or (self.oc_problem_name == "Multi Quadcopter"):
                        gradPhi_p = (gradPhi[:,:self.state_dim].reshape(batch_size*self.num_agents, -1))[:,:3]
                        cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) - self.compute_grad_G_z(z) )
                    elif self.oc_problem_name == "Multi Bicycle":
                        gradPhi_p = (gradPhi[:,:self.state_dim].reshape(batch_size*self.num_agents, -1))[:,:2]
                        cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) - self.compute_grad_G_z(z) )
                    elif self.oc_problem_name == "Single Quadcopter":
                        gradPhi_p = (gradPhi[:,:self.state_dim].reshape(batch_size, -1))[:,:3]
                        cadjfin = torch.mean(gradPhi_p.reshape(batch_size,-1) - self.compute_grad_G_z(z) )
                else:
                    cadjfin = cadjfin + torch.mean(gradPhi[:,:self.state_dim] - self.compute_grad_G_z(z) )

                if hasattr(u.p_net, "getPhi"):
                    cHJBfin = torch.mean(torch.linalg.vector_norm(u.p_net.getPhi(ti,z) - self.alphaG*temp_final_cost.view(-1, 1),ord=2,dim=1))
        
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

    def compute_loss_consumcheck(self, policy, z0, z_t=None, p_t=None, phi_t=None, jac_based=False):
        
        """
        JFB compute_loss with closed-form adjoint check.
        """
        B = z0.shape[0]
        dt = (self.t_final - self.t_initial) / self.nt
        running_cost = torch.tensor(0.0, device=z0.device)
        cHJB         = torch.tensor(0.0, device=z0.device)
        cadj         = torch.tensor(0.0, device=z0.device)
        max_grad_u_H = torch.tensor(-1.0, device=z0.device)
        z = z0.clone().requires_grad_(True)
        t = self.t_initial

        # Compute Phi0 and its gradient dPhi0 = D_zPhi0
        Phi  = policy.p_net.getPhi(t, z)              # (B,1)
        dPhi = torch.autograd.grad(
                Phi.sum(), z,
                create_graph=True,
                retain_graph=True
            )[0]                                  # (B, D)

        for k in range(self.nt):

            u_k = policy(z, t)                       # (B, m)
            f_k = self.compute_f(t, z, u_k)          # (B, D)
            # print('shapes', u_k.shape, f_k.shape, z.shape, t)
            
            # Forward‐Euler to next state
            z1 = z + dt * f_k
            t1 = t + dt
            z1 = z1.requires_grad_(True)

            # Next‐step Phi1 and gradient dPhi1
            Phi1  = policy.p_net.getPhi(t1, z1)      # (B,1)
            dPhi1 = torch.autograd.grad(
                    Phi1.sum(), z1,
                    create_graph=True,
                    retain_graph=True
                )[0]                             # (B, D)

            # Running cost at (t, z1, u_k)
            running_cost = running_cost + dt * torch.mean(
                self.compute_lagrangian(t, z1, u_k)
            )

            # HJB residual: Phi_k - [Phi_{k+1} - dt*(L + dPhi*f)]
            H_val     = self.compute_general_H(t, z1, u_k, dPhi)
            resid_hjb = Phi.view(B) - (Phi1.view(B) - dt * H_val)
            cHJB      = cHJB + torch.mean(resid_hjb.pow(2))

            # closed-form adjoint check
            #    finite-difference of dPhi
            dp    = (dPhi1 - dPhi) / dt                       # (B, D)
            # print(f"[HJB step {k}, p_prev: {dPhi.shape}, f_val: {f_k.shape}] ")
            #    closed-form RHS:
            #      dot p_x = -r p_x
            rhs_px = -self.r * dPhi[:, 0]                     # (B,)
            #      dot p_h = e^{-delta t} (u-h)^{-gamma} + (dPhi_h @ B)
            h_k    = z[:, 1:1+self.m]                         # (B,m)
            rhs_ph = (
                torch.exp(-self.delta * torch.tensor(t1, device=z0.device, dtype=z0.dtype)) 
                * (u_k - h_k).pow(-self.gamma) #.clamp_min(1e-6)
                + dPhi[:, 1:1+self.m] @ self.B
            )                                                 # (B,m)
            rhs    = torch.cat([rhs_px.unsqueeze(1), rhs_ph], dim=1)  # (B,D)
            residA = dp - rhs                                 # (B,D)
            cadj   = cadj + torch.mean(residA.pow(2).sum(dim=1))
            # print('cadj', cadj)

            # 7) Track max ||D_u H||
            grad_uH = self.compute_grad_H_u(t, z, u_k, dPhi)
            max_norm = grad_uH.norm(dim=1).max()
            max_grad_u_H = torch.maximum(max_grad_u_H, max_norm)

            # 8) Next step
            z, t, Phi, dPhi = z1, t1, Phi1, dPhi1

        # Terminal penalties
        terminal_cost = torch.mean(self.compute_G(z))

        # adjoint terminal: p_T − DG(z_T)
        gradG   = self.compute_grad_G_z(z)                # (B,D)
        cadjfin = torch.mean((dPhi - gradG).pow(2).sum(dim=1))

        # terminal HJB: Phi_T − G(z_T)
        resid_hjb_fin = Phi.view(B) - self.compute_G(z)
        cHJBfin = torch.mean(resid_hjb_fin.pow(2))

        total_cost = (
            self.alphaL * running_cost
        + self.alphaG * terminal_cost
        + self.alphaHJB[0] * cHJB
        + self.alphaadj[1]   * cadjfin
        + self.alphaHJB[1] * cHJBfin
        )

        return (total_cost, running_cost, terminal_cost,
                cHJB, cHJBfin, cadj, cadjfin, max_grad_u_H)
    
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

