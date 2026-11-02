"""
Working Jacobian-based trainer that extends your existing ImplicitNetSOC.
This follows the exact pattern from your utils_JFB.py but adapted for optimal control.
"""

import torch
import torch.nn as nn
import time
# Imports operating-system utilities for directories, file paths,
import os
#use it to convert training history into a table and save a CSV file.
import pandas as pd
from matplotlib import pyplot as plt
from Stochastic2D import Stochastic2D
from CVXPolicy import CVXPolicy_S2D
from ImplicitNets import ImplicitNetSOC
from ImplicitNets import Phi
# Imports process-monitoring utilities. We use this package to inspect CPU memory consumption.
import psutil
import copy

class LRScheduler:
    """
    Custom LR scheduler class that is similar to PyTorch's
    ReduceLROnPlateau LR scheduler, but reduces the learning rate
    by some factor after a fixed number of consecutive epochs during 
    which there is no decrease in the loss function. ReduceLROnPlateau 
    reduces the learing rate only after a fixed number of epochs in which 
    the loss function is less than the best loss achieved up to that point
    """

    # Initializer : initial rate, minimum rate, reduction factor, patience (number of epochs to wait before reducing the learning rate)
    def __init__(self, optim, init_lr, min_lr, fact, pat):
        # Store the optimizer and parameters
        self.optimizer = optim
        self.initial_lr = init_lr
        # stores the minimum learning rate, reduction factor and patience 
        self.min_lr = min_lr
        self.factor = fact
        self.patience = pat
        self.current_lr = init_lr
        # initialize the counter of consecutive losses that did not decrease 
        self.num_no_decr = 0
        # set the previous loss to infinity. 
        self.prev_loss = float('inf')
        # initialize the current epoch at 0
        self.current_epoch = 0

    # Defines a method for retrieving the initial learning rate 
    def get_initial_lr(self):
        return self.initial_lr

    # Defines a method for retrieving the current learning rate 
    def get_current_lr(self):
        return self.current_lr

    # Defines a method for retrieving the current epoch
    def get_current_epoch(self):
        return self.current_epoch

    # Defines a method for retrieving the previous loss value 
    def get_prev_loss(self):
        return self.prev_loss

    # Defines one scheduler update. The caller supplies a new loss value 
    def step(self, new_loss):
        # increase the current epoch count by 1
        self.current_epoch += 1
        # checks whether new loss is strictly smaller than previous loss
        if new_loss < self.prev_loss:
            # if the loss decreased, reset the counter of consecutive losses that did not decrease
            self.num_no_decr = 0
        else:
            # add one more on the counter of consecutive losses that did not decrease
            self.num_no_decr += 1
        # sets the previous to new loss for the next step 
        self.prev_loss = new_loss

        # The counter must exceed the patience and this much not be the first step. 
        if (self.num_no_decr > self.patience) and (self.current_epoch != 1):
            # reduce the learning rate by the factor and update the optimizer's learning rate 
            self.current_lr *= self.factor

            # Ensuring that the current leanring rate does not go below minimum learning rate 
            if self.current_lr < self.min_lr:
                self.current_lr = self.min_lr

            # update the optimizer's learning rate for all parameter groups 
            for g in self.optimizer.param_groups:
                g['lr'] = self.current_lr
                
            self.num_no_decr = 0

# Next, we define the trainer class that will handle the training loop for the optimal control problem. 
# This class will manage the policy network, the stochastic control problem, and the optimizer. 
# It will also handle the training process, including logging and saving the best model.
class OptimalControlTrainer:
    # initialize the trainer with policy_net, soc_problem, optmizer, scheduler, device, and verification
    def __init__(self, policy_net, soc_problem, optimizer, scheduler=None, ver=False, device='cpu'):
        # store the policy network, SOC problem, optimizer, scheduler, device and verification 
        self.policy = policy_net
        self.soc_problem = soc_problem
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mode = 'standard' # Default mode
        self.verify = ver # Whether or not to do additional computations to verify theoretical assumptions, default is False
        # Gradient clipping for direct control methods
        self.enable_grad_clip = False
        self.grad_clip_value = 1.0
        # === CHANGE: Restored full history tracking ===
        # Supplies name for loss, cost, and many other things. 
        self.history = {k: [] for k in [
            'epoch', 'loss', 'running_cost', 'terminal_cost', 
            'cHJB', 'cHJBfin', 'cadj', 'cadjfin',
            'time_per_epoch', 'grad_norm', 'lr', 'max_fp_itrs',
            'max_fp_res_norm', 'memory_MB', 'max_memory_MB',
            'gpu_memory_MB', 'gpu_max_memory_MB', 'work_units',
            'max_grad_H', 'avg_grad_H', 'smallest_M_sdval', 
            'largest_M_sdval', 'smallest_lambda_min', 'largest_lambda_max',
            'max_grad_T_u', 'avg_grad_T_u', 'sd_grad_T_u', 'angle'
        ]}

    # Sets the mode of the trainer to standard or cvx. 
    def set_mode(self, mode='standard'):
        if mode not in ['standard', 'cvx']:
            raise ValueError("Mode must be 'standard' or 'cvx'")
        self.mode = mode
        print(f"Trainer mode set to '{self.mode}'")

    # Defines one training update using the supplies batch of initial states z0
    def standard_step(self, z0):
        # switches the policy to training mode
        self.policy.train()
        # Initialize the fixed point iteration and residual norm to zero
        max_fp_itrs = 0.0
        max_fp_res_norm = 0.0
        
        # if self.mode == 'standard':
        #     convergence_stats = self.policy.get_convergence_stats()
        #     max_fp_itrs = convergence_stats['fp_depth']
        #     max_fp_res_norm = convergence_stats['max_res_norm']

        # clears previously accumulated gradient for the optimizer's parameter. 
        self.optimizer.zero_grad()
        # === CHANGE: Unpack all 7 values from compute_loss ===
        total_cost, run_cost, term_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H = self.soc_problem.compute_loss(self.policy, z0)
        # differentiates the scalar loss through the recorded computation graph and accumulates parameter gradients
        total_cost.backward()
        # check whether the mode is standard
        if self.mode == 'standard':
            # retrieves the policy'r currently stored statistics 
            convergence_stats = self.policy.get_convergence_stats()
            # read the dictionary entry for max no. of iteration and max residual norm 
            max_fp_itrs = convergence_stats['fp_depth']
            max_fp_res_norm = convergence_stats['max_res_norm']
        # check whether gradient clipping is enabled 
        if self.enable_grad_clip:
            #If enabled, rescales parameter gradients when their combined norm exceeds the threshold.
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_value)
        # Updates the parameters using their stored gradients and the optimizer’s rule. This is where the network weights actually change.
        self.optimizer.step()
        # === CHANGE: Return all cost components ===
        return {
            'loss': total_cost.item(), 
            'running_cost': run_cost.item(), 
            'terminal_cost': term_cost.item(),
            'cHJB': cHJB.item(),
            'cHJBfin': cHJBfin.item(),
            'cadj': cadj.item(),
            'cadjfin': cadjfin.item(),
            'max_fp_itrs': max_fp_itrs,
            'max_fp_res_norm': max_fp_res_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'max_grad_H': max_grad_H,
            'avg_grad_H': avg_grad_H
        }
    # Now the verifcation variant. Its repeaded training statemnets still perform the same individual operations 
    # Define a training step that also requests additional diagnostic calculations
    def standard_step_verify(self, z0):
        # set to train
        self.policy.train()
        # initialiing max iter with res norm and angle
        max_fp_itrs = 0.0
        max_fp_res_norm = 0.0
        angle = 0.0
        
        # if self.mode == 'standard':
        #     convergence_stats = self.policy.get_convergence_stats()
        #     max_fp_itrs = convergence_stats['fp_depth']
        #     max_fp_res_norm = convergence_stats['max_res_norm']

        # make the optimizer forget previously accumulated gradients for the parameters 
        self.optimizer.zero_grad()

        # Copy weights to compute gradient using full AD as well as JFB for angle computation
        #policy_AD = copy.deepcopy(self.policy) 
        # === CHANGE: Unpack all values from compute_loss ===
        total_cost, run_cost, term_cost, cHJB, cHJBfin, cadj, cadjfin, max_grad_H, avg_grad_H, smallest_M_sdval, largest_M_sdval, smallest_lambda_min, largest_lambda_max, max_grad_T_u, avg_grad_T_u, sd_grad_T_u = self.soc_problem.compute_loss_verify(self.policy, z0)
        #og_full_AD = self.oc_problem.track_all_fp_iters
        #self.oc_problem.track_all_fp_iters = True
        #total_cost_ad, run_cost_ad, term_cost_ad, cHJB_ad, cHJBfin_ad, cadj_ad, cadjfin_ad, max_grad_H_ad, avg_grad_H_ad = self.oc_problem.compute_loss(policy_AD, z0)
        #self.oc_problem.track_all_fp_iters = og_full_AD

        # Calculates parameter gradients 
        total_cost.backward()
        # same thing as before,set to standard, get convergence stats, iterations and residual norm
        if self.mode == 'standard':
            convergence_stats = self.policy.get_convergence_stats()
            max_fp_itrs = convergence_stats['fp_depth']
            max_fp_res_norm = convergence_stats['max_res_norm']
        # check for clipping
        if self.enable_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_value)
        #total_cost_ad.backward()
        """
        nabla_theta_J_ = []
        for p_ad in policy_AD.parameters():
            if p_ad.grad is not None:
                nabla_theta_J_.append(p_ad.grad.view(-1))
        nabla_theta_J = torch.cat(nabla_theta_J_)
        d_JFB_ = []
        for p in self.policy.parameters():
            if p.grad is not None:
                d_JFB_.append(p.grad.view(-1))
        d_JFB = torch.cat(d_JFB_)
        angle = torch.acos(torch.dot(nabla_theta_J, d_JFB)/(torch.linalg.norm(nabla_theta_J, ord=2)*torch.linalg.norm(d_JFB, ord=2))).item()
        print(f"angle between true gradient and JFB approximation: {angle:.4e}")
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        """
        # update the parametes using the gradients calculated above 
        self.optimizer.step()
        # === CHANGE: Return all cost components ===
        return {
            'loss': total_cost.item(), 
            'running_cost': run_cost.item(), 
            'terminal_cost': term_cost.item(),
            'cHJB': cHJB.item(),
            'cHJBfin': cHJBfin.item(),
            'cadj': cadj.item(),
            'cadjfin': cadjfin.item(),
            'max_fp_itrs': max_fp_itrs,
            'max_fp_res_norm': max_fp_res_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'max_grad_H': max_grad_H,
            'avg_grad_H': avg_grad_H,
            'smallest_M_sdval': smallest_M_sdval,
            'largest_M_sdval': largest_M_sdval,
            'smallest_lambda_min': smallest_lambda_min,
            'largest_lambda_max': largest_lambda_max,
            'max_grad_T_u': max_grad_T_u,
            'avg_grad_T_u': avg_grad_T_u,
            'sd_grad_T_u': sd_grad_T_u,
            'angle': angle
        }

    def cvx_step(self, z0):
        # The logic is identical for CVX mode, but we keep it separate for clarity
        return self.standard_step(z0)

    # Defines the method that chooses and runs one training update.
    def train_epoch(self, z0):
        if self.mode == 'cvx':
            return self.cvx_step(z0)
        elif self.mode == 'standard' and self.verify: # verify assumptions
            return self.standard_step_verify(z0)
        else: # standard mode
            return self.standard_step(z0)

    # z0, number of epoches, plot after every 25 epoches and best policy is kind of check point.
    def train(self, z0, num_epochs, verbose=True, plot_frequency=25, save_model_name='best_policy'):
        # get the problem type and read its name to create a directory for saving results. 
        soc_name = type(self.soc_problem).__name__
        # Builds a output-directory path using the problem name and mode. (example: results_Stochastic2D/standard_mode)
        save_dir = f'results_{soc_name}/{self.mode}_mode'
        # Check whether the directory is missing and create it if necessary
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{save_model_name}.pth')
        # === Save CSV after each epoch ===
        # Builds the CSV ihstory filename 
        history_path = os.path.join(save_dir, f'history_{save_model_name}.csv')
        # Print the selected training mode and num of epochs
        print(f"Starting training in '{self.mode}' mode for {num_epochs} epochs. Results in '{save_dir}'")
        print("-" * 60)
        # Initialize the best loss to infinity.
        best_loss = float('inf')

        # Get the current process ID with os.getpid(), then creates a psutil object for inspecting that process
        # In simple words: This line retrieves the current process ID using os.getpid() and then creates a psutil.Process object for that process.
        #  This object allows you to monitor and retrieve information about the current process, such as memory usage, CPU usage, and other system-related metrics.
        process = psutil.Process(os.getpid())
        # Loops for epoch 1 through total no of epochs. 
        for epoch in range(1, num_epochs + 1):
            # GPU Memory usage (in MB)
            # initialize GPU memory usage 
            gpu_memory_MB = 0.0
            gpu_max_memory_MB = 0.0
            max_memory_MB = 0.0
            # records the current time before the training start
            epoch_start_time = time.time()
            # perfomrs on training update and recives it dictionary of metrices. The same supplies z0 is reused each epoch
            step_info = self.train_epoch(z0)
            # Supplies the reported loss to scheduler to update the learning rate if needed.
            self.scheduler.step(step_info['loss'])

            # Reads the CPU memory in bytes and divides by 1024 twice to convert into megabytes(MB). 
            memory_MB = process.memory_info().rss / 1024 / 1024
            if memory_MB > max_memory_MB:
                max_memory_MB = memory_MB
            if torch.cuda.is_available():
                gpu_memory_MB = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_max_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024
            time_per_epoch = time.time() - epoch_start_time
            # Loops over parameters with gradients, computes each gradient tensor's norm, square those norms, sum them 
            # and then takes the square root of the sum to get the overall gradient norm. (p -parameter tensor)
            # This is a common way to measure the magnitude of the gradients across all parameters in a neural network.
            grad_norm = sum(p.grad.norm().item()**2 for p in self.policy.parameters() if p.grad is not None)**0.5

            # Reads the number of tracked iterations from the policy. If the policy does not have this attribute, it defaults to 1
            tracked = getattr(self.policy, 'tracked_iters', 1)
            # Calculates the total work units for the epoch: batch size * no of time steps* tracked iteration count
            work_units = self.soc_problem.batch_size*self.soc_problem.nt*tracked
            # If the reported fixed point count is smaller than the tracked iteration count or if full iteration tracking is enabled,
            # then the work units are recalculated using the reported fixed point count instead of the tracked iteration count.
            if (step_info['max_fp_itrs'] < tracked) or self.soc_problem.track_all_fp_iters:
                work_units = self.soc_problem.batch_size*self.soc_problem.nt*step_info['max_fp_itrs']
            

            # This loop correctly populates the full history dict
            for key in self.history:
                if key == 'memory_MB':
                    self.history[key].append(memory_MB)
                elif key == 'max_memory_MB':
                    self.history[key].append(max_memory_MB)
                elif key == 'gpu_memory_MB':
                    self.history[key].append(gpu_memory_MB)
                elif key == 'gpu_max_memory_MB':
                    self.history[key].append(gpu_max_memory_MB)
                elif key == 'work_units':
                    self.history[key].append(work_units)
                else:
                    self.history[key].append(locals().get(key, step_info.get(key, 0)))

            # === CHANGE: Updated print statement to show HJB/ADJ costs and memory ===
            if verbose:
                print(f"Epoch {epoch:03d} | Loss: {step_info['loss']:.3e} | L: {step_info['running_cost']:.3e} | G: {step_info['terminal_cost']:.3e} | "
                        f"HJB: {step_info.get('cHJB', 0):.3e} | HJB fin: {step_info.get('cHJBfin',0):.3e} |Adj: {step_info.get('cadj', 0):.2e} | "
                      f"Grad: {grad_norm:.2e} | Time: {time_per_epoch:.2f}s | "
                      f"CPU Mem: {memory_MB:.1f}MB | Max CPU: {max_memory_MB:.1f}MB | "
                      f"GPU Mem: {gpu_memory_MB:.1f}MB | Max GPU: {gpu_max_memory_MB:.1f}MB | lr: {step_info['lr']:.3e} | "
                      f"max_fp_itrs: {step_info['max_fp_itrs']} | res_norm: {step_info['max_fp_res_norm']:.3e} | max_grad_H: {step_info['max_grad_H']:.3e} | avg_grad_H: {step_info['avg_grad_H']:.3e} ")

            # assign the current loss to prev_loss for the next epoch
            prev_loss = step_info['loss']

            # Checks whether the reported training loss is lower than the best reported loss seen during this call to train
            if step_info['loss'] < best_loss:
                # updates the best loss record
                best_loss = step_info['loss']
                # save the policy's parameters and registered buffers. 
                torch.save(self.policy.state_dict(), save_path)
                # Prints a message indicating that a new best model has been saved, along with the loss value.
                if verbose: print(f"    -> New best model saved to {save_path} with loss {best_loss:.4e}")

            # Plot: trajectories every plot_frequency epochs, saved into save_dir with the epoch in the name
            if plot_frequency and epoch % plot_frequency == 0:
                self.policy.eval()
                with torch.no_grad():
                    z_traj = self.soc_problem.generate_trajectory(self.policy, z0,self.soc_problem.nt, return_full_trajectory=True)
                self.soc_problem.plot_trajectories(z_traj.detach(), filename=os.path.join(save_dir, f'trajectories_{save_model_name}_epoch{epoch:04d}.png'))

                # restores training mode after plotting branches
                self.policy.train()

            # === Save CSV after each epoch ===
            #Converts the dictionary of history lists into a table and writes it to CSV.
            # This rewrites the accumulated table each epoch 
            pd.DataFrame(self.history).to_csv(history_path, index=False)
        # Calls the loss-plotting method after the epoch loop.
        self.plot_loss_curve(save_dir, save_model_name)
        # Returns the trainer’s history dictionary to the caller after training finishes.
        return self.history
    
    def plot_loss_curve(self, save_dir, save_model_name):
        plt.figure(figsize=(10, 6)); plt.yscale('log'); plt.grid(True)
        # epochs horizontally and loss vertically.
        plt.plot(self.history['epoch'], self.history['loss'], label='Total Loss')
        plt.title(f'Training Loss (Mode: {self.mode})'); plt.legend()
        plt.savefig(os.path.join(save_dir, f'loss_curve_{save_model_name}.png')); plt.close()

# Runs the intended block whenfile is executed directly, importing normally skips this block. 
if __name__ == '__main__':

    # --- Shared Configuration ---
    config_OC = {'batch_size': 10, 'nt': 2, 't_final': 1.0, }
    config_train = { 'lr': 1e-3, 'epochs': 2}
    device = 'cpu'

    # --- 1. Train with Standard Mode ---
    print("\n\n--- Testing Standard Mode ---")
    # Construct the problem
    S2D_problem_std = Stochastic2D(device=device, **config_OC)
    # Construct the scalar network
    phi_std = Phi(3, 10, S2D_problem_std.state_dim, dev=device)
    # Begins constructing the implicit policy with the dimensions of the problem.
    implicit_net = ImplicitNetSOC(S2D_problem_std.state_dim, S2D_problem_std.control_dim,
        soc_problem=S2D_problem_std, p_net=phi_std, dev=device).to(device)
    # Creats an Adam optimizer for the implicit policy network with specified learning rate.
    optimizer_std = torch.optim.Adam(implicit_net.parameters(), lr=config_train['lr'])
    # Creates Pytorch scheduler that reduces the learning rate when a metric has stopped improving. 
    # Basically after every 15 bad epoches, learning rate is multiplied by 0.8. We can change it also.
    scheduler_std = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_std, mode='min', factor=0.8, patience=15)
    # Creates the trainer for implicit_net, SOC problem, optimizer and scheduler and sets the device.
    trainer_std = OptimalControlTrainer(implicit_net, S2D_problem_std, optimizer_std, scheduler_std, device=device)
    # sets the training mode to standard
    trainer_std.set_mode('standard')
    # define initial condition 
    z0_std = S2D_problem_std.sample_initial_condition()
    # Runs two training updates. It saves the final loss curve, 
    trainer_std.train(z0_std, num_epochs=config_train['epochs'], save_model_name='best_standard_policy')
    
    # --- 2. Train with CVXPY Mode ---
    # Everything is same just for CVX.
    print("\n--- Testing CVXPY Mode ---")
    from ImplicitNets import Phi
    S2D_problem = Stochastic2D(device=device, **config_OC)
    Phi = Phi(3, 10, S2D_problem.state_dim, dev=device)
    cvx_policy = CVXPolicy_S2D(S2D_problem.state_dim, S2D_problem.control_dim, p_net=Phi).to(device)
    optimizer = torch.optim.Adam(cvx_policy.parameters(), lr=config_train['lr'])
    scheduler_cvx = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15)
    trainer_cvx = OptimalControlTrainer(cvx_policy, S2D_problem, optimizer, scheduler_cvx, device=device)
    trainer_cvx.set_mode('cvx')
    z0 = S2D_problem.sample_initial_condition()
    trainer_cvx.train(z0, num_epochs=config_train['epochs'], save_model_name='best_cvx_policy')


    # check grad_u H of the trained model
    print("\n--- Checking grad_u H of the trained model ---")
    
    
