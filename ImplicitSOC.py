import torch

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
                 track_all_fp_iters = False): 
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
    def compute_grad_G_z(self,z):
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
        
        # Delta W = sqrt(h) \xi where \xi \in N(0,I)
        return torch.sqrt(torch.as_tensor(self.h, device=z.device, dtype=z.dtype)
        ) * torch.randn(z.shape[0], self.noise_dim, device=z.device, dtype=z.dtype)

    def compute_q(self, t, z, u, phi_net):
        """
        Compute p = Phi_z and q = Phi_zz * sigma.

        Returns:
            grad_V: [batch_size, state_dim]
            q:      [batch_size, state_dim, noise_dim]

        Assumes independent batch samples and that phi_net(t,z)
        returns the state gradient of its scalar Phi.
        """
        sigma = self.compute_sigma(t, z, u)

        if sigma.dim() != 3:
            raise ValueError(
                "compute_sigma(t,z,u) must return a tensor of shape "
                "(batch_size, state_dim, noise_dim)"
            )

        # Keep the state connected to the existing trajectory graph.
        z_hess = z

        # existing Phi.forward expects one Python scalar time.
        t_scalar = torch.as_tensor(t, device=z.device, dtype=z.dtype ).detach().item()

        # Phi.forward already supplies Phi_z.
        # Temporarily give one sample a batch dimension, then remove it.
        grad_V_fun = lambda z_i: phi_net(t_scalar, z_i.unsqueeze(0)).squeeze(0)

        # Evaluate Phi_z for every sample.
        grad_V = torch.func.vmap(grad_V_fun)(z_hess)

        # Differentiate Phi_z to obtain Phi_zz for every sample.
        hessian_V = torch.func.vmap( torch.func.jacrev(grad_V_fun))(z_hess)

        # q[b,i,k] = sum_j Phi_zz[b,i,j] * sigma[b,j,k].
        q = torch.einsum('bij,bjk->bik', hessian_V, sigma)

        return grad_V, q
    

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
        L_val = self.alphaL*self.compute_lagrangian(t, z, u)
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
        grad_term1 = self.alphaL*self.compute_grad_lagrangian(t, z, u)
        
        # Compute gradient of dynamics
        grad_f_u_term = self.compute_grad_f_u(t, z, u)
        _ , q = self.compute_q(t, z, u, phi_net)
        # Setup : maximize H in u with (p,q) held fixed. sigma depend on u,

        grad_sigma_u_term = self.compute_grad_sigma_u(t, z, u)

        # Compute gradient of p^T f w.r.t. u
        p = p.unsqueeze(-1)  # Shape: (batch_size, state_dim, 1)
        grad_term2 = torch.bmm(grad_f_u_term, p).view(batch_size, self.control_dim)
        grad_term3 = torch.einsum('bdkm,bdk->bm', grad_sigma_u_term, q).view(batch_size, self.control_dim)

        # Compute total gradient
        # H_u = -L_u - f_u^T p - <q, dsigma/du>
        grad_H_u_val = -grad_term1 - grad_term2 - grad_term3
        
        return grad_H_u_val

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
        grad_L_z_term = self.alphaL*self.compute_grad_L_z(t, z, u)
        grad_f_z_term = self.compute_grad_f_z(t, z, u)
        grad_sigma_z_term = self.compute_grad_sigma_z(t, z, u)
        _, q = self.compute_q(t, z, u, phi_net)

        output = -grad_L_z_term - torch.bmm(grad_f_z_term, p.unsqueeze(-1)).view(batch_size, self.state_dim) - torch.einsum('bijk,bij->bk', grad_sigma_z_term, q).view(batch_size, self.state_dim)
        return output
    
    def solve_adjoint_eq(self, z, u, phi_net, dW):
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
        p = torch.zeros(batch_size, self.state_dim, nt, device=z.device, dtype = z.dtype)
        p[:,:,-1] = self.alphaG*self.compute_grad_G_z(z[:,:,-1])
        h = (self.t_final - self.t_initial)/(nt-1)

        for i in range(nt-2, -1, -1):
            t_i = self.t_initial + i*h
            u_i = u(z[:,:,i],t_i).view(batch_size, self.control_dim)
            H_z_i = self.compute_grad_H_z(t_i, z[:,:,i], u_i, p[:,:,i+1],phi_net)
            _ , q_i = self.compute_q(t_i, z[:,:,i], u_i, phi_net)
            dW_i = dW[:,:,i]
            q_dW_i = torch.bmm(q_i, dW_i.unsqueeze(-1)).squeeze(-1)
            p[:,:,i] = p[:,:,i+1] - h * H_z_i - q_dW_i

        return p
    
    def compute_loss(self, u, z0, verify = False):
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
        z=z0
        cHJB, cHJBfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        cadj, cadjfin = torch.tensor(0.0, device=z0.device, dtype=z0.dtype), torch.tensor(0.0, device=z0.device, dtype=z0.dtype)
        largest_grad_H_u = -1.0
        avg_grad_H_u = 0.0
        if verify:
            smallest_M_sdval = torch.inf # Smallest singular value of M = dT/dtheta over all samples in batch and over all time steps 
            largest_M_sdval = -1.0 # Largest singular value of M over all samples in batch and over all time steps 
            smallest_lambda_min = torch.inf # Batchwise-largest largest eigenvalue of (MM^{T})^{-1} over all time steps 
            largest_lambda_max = -1.0 # Batchwise-smallest smallest eigenvalue of (MM^{T})^{-1} over all time steps
            avg_grad_T_u = 0.0
            largest_grad_T_u_batch = -1.0*torch.ones(batch_size, device=z0.device, dtype = z0.dtype) # Largest norm of grad of T with respect to u, for each sample

        for i in range(self.nt):
            t_i = self.t_initial + i*self.h
            t_next = self.t_final if i==self.nt-1 else self.t_initial + (i+1)*self.h
            u_i = u(z, t_i, track_all_fp_iters=self.track_all_fp_iters).view(batch_size, self.control_dim)
                # Apply the same checks with verification both off and on.
            f_i = self.compute_f(t_i, z, u_i)
            sigma_i = self.compute_sigma(t_i, z, u_i)
            dW_i = self.sample_dW(z, sigma_i)
            running_cost = running_cost + self.h * self.compute_lagrangian(t_i, z, u_i)     
            z_next = z +self.h*f_i + torch.bmm(sigma_i, dW_i.unsqueeze(-1)).squeeze(-1)
            # Use existing Phi network directly [Phi_z, Phi_t]           
            gradPhi = u.p_net(t_i, z, full_grad=True)
            p_i = gradPhi[:,:self.state_dim]
            p_next = u.p_net(t_next, z_next)
            _, q_i = self.compute_q(t_i, z, u_i, phi_net=u.p_net)
            H_i = self.compute_H(t_i, z, u_i, p_i, u.p_net, compute_general_H=False).view(-1,1)
            HJB_residual = gradPhi[:,-1:] - H_i
            cHJB = cHJB + self.h*torch.mean(torch.linalg.vector_norm(HJB_residual, ord=2, dim=1))
            # Adjoint eqn loss
            H_z_i = self.compute_grad_H_z(t_i, z, u_i, p_i, u.p_net)
            adj_residual = p_next - p_i - self.h*H_z_i - torch.bmm(q_i, dW_i.unsqueeze(-1)).squeeze(-1)
            cadj = cadj + torch.mean(torch.linalg.vector_norm(adj_residual, ord=2, dim=1))

            grad_H_u = self.compute_grad_H_u(t_i,z, u_i, p_i, phi_net=u.p_net)
            max_norm_grad_H_u = torch.max(torch.norm(grad_H_u, dim=1)).item()
            avg_grad_H_u += torch.mean(torch.norm(grad_H_u, dim=1)).item()
            if max_norm_grad_H_u > largest_grad_H_u:
                largest_grad_H_u = max_norm_grad_H_u

            # Verify Assumption 2 and Hypothesis of Lemma 1 in End-to-end
            # training paper
            if verify:
                M_theta, theta0, metadata = self.compute_grad_T_theta(u, u_i, z, t_i)
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

                grad_T_u = self.compute_grad_T_u(u_i, z, t_i, model=u)
                norm_grad_T_u = torch.linalg.matrix_norm(grad_T_u, ord=2, dim=(1, 2))
                idx_max = torch.argwhere(norm_grad_T_u > largest_grad_T_u_batch).flatten()
                largest_grad_T_u_batch[idx_max] = norm_grad_T_u[idx_max]
                avg_grad_T_u += torch.mean(norm_grad_T_u).item()

            z = z_next 

            # Calculate terminal cost
        running_cost = running_cost.mean()
        temp_final_cost = self.compute_G(z)
        terminal_cost = torch.mean(temp_final_cost)

        # Measure the terminal conditions of both adjoint 
        gradPhi = u.p_net(self.t_final, z, full_grad = True)
        cHJBfin = torch.mean(torch.linalg.vector_norm(u.p_net.getPhi(self.t_final, z)-self.alphaG*temp_final_cost.view(-1,1), ord=2, dim=1))
        cadjfin = torch.mean(torch.linalg.vector_norm(gradPhi[:,:self.state_dim]-self.alphaG*self.compute_grad_G_z(z), ord=2, dim=1))
        avg_grad_H_u = avg_grad_H_u / self.nt
        
        # Calculate total cost
        total_cost = (self.alphaL * running_cost + self.alphaG * terminal_cost 
                      + self.alphaHJB[0] * cHJB + self.alphaHJB[1] * cHJBfin
                      + self.alphaadj[0] * cadj + self.alphaadj[1] * cadjfin)

        results = (total_cost, running_cost, terminal_cost, cHJB, cHJBfin, cadj, cadjfin, largest_grad_H_u, avg_grad_H_u)

        # Verify assumptions
        if verify:
            avg_grad_T_u = avg_grad_T_u / self.nt
            sd_grad_T_u = torch.std(largest_grad_T_u_batch, unbiased=False).item()
            largest_grad_T_u = torch.max(largest_grad_T_u_batch).item()
            return results + (smallest_M_sdval, largest_M_sdval, smallest_lambda_min, largest_lambda_max, largest_grad_T_u, avg_grad_T_u, sd_grad_T_u)
        return results
    
    def compute_loss_verify(self, u, z0):
        # Old trainer name only; all calculations are inside compute_loss.
        return self.compute_loss(u, z0, verify=True)
    
    # Singular value of T_{theta} measure parameter sensitivity and rank. 
    def compute_grad_T_theta(self, model,u,  z, ti, create_graph=False):
        """
            Compute the parameter Jacobian of one fixed-point update,
            holding the supplied control u and state z fixed.

            The model's training/evaluation mode is unchanged.

            Returns:
                J: Jacobian with shape [batch_size, control_dim, P].
                theta0: Detached snapshot of the trainable parameter values.
                metadata: Information for reconstructing parameter shapes.

            If create_graph=True, higher derivatives connect to the model
            parameters, not to the theta0 snapshot.

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
        # Collect the trainable parameters in a consistent order
        params_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
        parameters = list(params_dict.values())
        if not parameters:
            raise ValueError("The model must have trainable parameters")
        # model.train()
        # # Named parameters and buffers (buffers may be empty; kept for generality)
        # params_dict = dict(model.named_parameters())
        # buffers = dict(model.named_buffers())

        # Record how to reconstruct each parameter
        names  = list(params_dict.keys())
        shapes = [p.shape for p in parameters]
        sizes  = [p.numel() for p in parameters]
        # record where each parameter starts and ends in the flattened vector
        idx = torch.tensor([0] + sizes, device=parameters[0].device).cumsum(0)

        theta0 = torch.cat([p.detach().reshape(-1) for p in parameters])

        # # Join all parameters into theta
        # def pack(pdict):
        #     return torch.cat([p.reshape(-1) for p in pdict.values()])

        # Reconstruct the parameter dictionary from theta
        def unflatten(theta):
            out = {}
            for i, k in enumerate(names):
                start, end = idx[i].item(), idx[i+1].item()
                out[k] = theta[start:end].view(shapes[i])
            return out
        with torch.enable_grad():
            T_value = model.T(u.detach(), z.detach(), ti)
            batch_rows =[]
            for b in range(u.shape[0]):
                rows=[]
                for j in range(u.shape[1]):
                     # Derivatives of one output component against all weights.
                    gradients = torch.autograd.grad(T_value[b,j], parameters, create_graph=create_graph, retain_graph=True, allow_unused=True)
                    pieces =[]
                    for p, g in zip(parameters, gradients):
                        if g is None:
                            g = torch.zeros_like(p)
                        pieces.append(g.reshape(-1))
                    # One row containing derivatives against all P parameters.    
                    rows.append(torch.cat(pieces))
                # One [control_dim, P] matrix for this sample
                batch_rows.append(torch.stack(rows))
            # Full batch: [batch_size, control_dim, P]
            J = torch.stack(batch_rows)


        metadata = {
            "names": names,
            "shapes": shapes,
            "idx": idx,
            "unflatten": unflatten,
        }
        return J, theta0, metadata

    # We calculate to to measure local sensitivity to the control input
    def compute_grad_T_u(self, u, z, t, model, create_graph=False):
        """
        Compute the Jacobian of one fixed-point update with respect to u.

        Args:
            u:
                Current control tensor, shape [batch_size, control_dim].
            z:
                Current state tensor, shape [batch_size, state_dim].
            t:
                Current time.
            model:
                Implicit policy containing T(u,z,t).
            create_graph:
                Whether the Jacobian calculation should remain differentiable.

        Returns:
            J:
                Control Jacobian, shape
                [batch_size, control_dim, control_dim].

        Assumes that different batch samples do not interact.
        """
        with torch.enable_grad():
            u_batch = u.detach().requires_grad_(True)
            z_fixed = z.detach()
            T_value = model.T(u_batch, z_fixed, t)
            rows = []
            for j in range(u_batch.shape[1]):
                # Differentiate output component j against all control components
                row = torch.autograd.grad(T_value[:,j].sum(), u_batch, create_graph=create_graph, retain_graph=True)[0]
                rows.append(row)

            J = torch.stack(rows, dim=1)

        return J