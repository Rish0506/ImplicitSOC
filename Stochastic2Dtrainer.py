import torch
import numpy as np
# reads options such as epochs
import argparse
from Stochastic2D import Stochastic2D
from ImplicitNets import ImplicitNetSOC, Phi
from OptimalControlTrainer import OptimalControlTrainer
#from CVXPolicy import CVXPolicy_S2D
import time
import sys, os

# This defines a class: a small object with its own data and methods.
class Logger(object):
    # runs when a new logger is created 
    def __init__(self, filename = "results_Stochastic2D/stochastic2d_run.log"):
        # saves the terminal output stream before the script replaces it. The logger can then continue showing output in the terminal
        # stdout : Python’s built-in file object responsible for handling standard output.
        self.terminal = sys.stdout
        # extracts the directory part of the file name. (ex: runs/output.log then logdir is run)
        logdir = os.path.dirname(filename)
        # this checks that there is a directory path and that it does not really exist. If it's missing, os.makedirs creates it.
        # exist_ok = True prevens an error if the directory already exists by the time command runs.
        if logdir and not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok = True)
        # This opens the log file in append mode("a"), so new output goes at the end instead of erasing earlier runs
        # append mode is a fike opening and writing setting that adds new data to the end of an existing file without deleting or converting what is already there.
        self.log = open(filename, "a")

    #def write(self, message): defines an instance method named write inside a class that takes the current object instance (self) and a parameter (message).
    # print() writes text to sys.stdout. The logger's write method sends that test to both the originial terminal and the file.
    # That's why logger sometimes is called a tee: it duplicated the output.
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    # def flush(self) is a method used to force the write buffer to empty its contents directly into the stream or file.
    # or flush() asks both streams to push any buffered text out immediately
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# This replaces Python’s usual standard-output stream with the logger. After this line, print(...) writes to both the terminal and that file.
# this happens as son as Python loads the file         
sys.stdout = Logger("results_Stochastic2D/stochastic2d_run.log")

def run_s2d_jfb(config_oc, config_train, full_AD_mode=False, device="cpu", plot_frequency=None):
    """
    Solves Stochastic2D optimal control problem with INN + JFB
    """
    print()
    print("#####################################################################")
    print("##############                                      #################")
    print("##############        Stochastic2D with INN+JFB     #################")
    print("##############                                      #################")
    print("#####################################################################")
    print()
    print(f"Full AD Mode: {full_AD_mode}")
    print()

    stochastic2D_std = Stochastic2D(device = device, **config_oc)
    print(f"State dimension: {stochastic2D_std.state_dim}, control dimension: {stochastic2D_std.control_dim}")
    print(f"batch_size={config_oc['batch_size']}  nt={config_oc['nt']}  " f"t_final={config_oc['t_final']}")
    phi_net = Phi(3, 96, stochastic2D_std.state_dim)
    tracked_iters = 1
    stochastic2D_std.track_all_fp_iters = full_AD_mode
    implicit_net = ImplicitNetSOC(stochastic2D_std.state_dim, stochastic2D_std.control_dim, max_iters=int(50-tracked_iters), alpha=1e-1, tol=1e-1, tracked_iters=tracked_iters, soc_problem=stochastic2D_std, p_net=phi_net, dev=device).to(device)
    optimizer_std = torch.optim.Adam(implicit_net.parameters(), lr=config_train['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_std, mode="min", factor=0.5, patience=40, min_lr=1e-4)
    trainer_std = OptimalControlTrainer(implicit_net, stochastic2D_std, optimizer_std, scheduler=scheduler, device=device)
    trainer_std.set_mode("standard")

    save_model_name = f"best_policy_JFB_BatchSize_{config_oc['batch_size']}__nt_{config_oc['nt']}_{time.ctime().replace(' ','_').replace(':','_')}"
    if full_AD_mode:
        save_model_name = f"best_policy_FullAD_BatchSize_{config_oc['batch_size']}_nt_{config_oc['nt']}_{time.ctime().replace(' ','_').replace(':','_')}"

    z0_std = stochastic2D_std.sample_initial_condition()
    trainer_std.train(z0_std, num_epochs=config_train['epochs'], plot_frequency=None, save_model_name=save_model_name)

# def run_s2d_jbb(config_oc,config_train, device="cpu", plot_frequency=None):
#     """
#     Solving problem using JBB
#     """
#     print()
#     print("##########################################################################")
#     print("#################                                           ##############")
#     print("#################     Stochastic_2D with INN + CVX (JBB)    ##############")
#     print("#################                                           ##############")
#     print("##########################################################################")
#     print()

#     stochastic_cvx = Stochastic2D(device=device, **config_oc)
#     phi_net = Phi(3,60, stochastic_cvx.state_dim)
#     cvx_policy = CVXPolicy_S2D(config_oc['nt'], p_net=phi_net, tol=2e-3, dev=device).to(device)
#     optimizer_cvx = torch.optim.Adam(cvx_policy.parameters(),lr = config_train['lr'])
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cvx, mode='min', factor=0.5, patience=10)
#     trainer_cvx = OptimalControlTrainer(cvx_policy, stochastic_cvx, optimizer_cvx, scheduler=scheduler, device=device)
#     trainer_cvx.set_mode('cvx')
#     save_model_name = f"best_policy_JBB_BatchSize_{config_oc['batch_size']}_nt_{config_oc['nt']}_{time.ctime().replace(' ','_').replace(':','_')}"
#     z0_cvx = stochastic_cvx.sample_initial_condition()
#     trainer_cvx.train(z0_cvx, num_epochs=config_train['epochs'], plot_frequency=None, save_model_name=save_model_name)

def main():
    parser = argparse.ArgumentParser(
        description="Train Stochastic2D optimal control with implicit networks"
    )

    parser.add_argument('--train_jfb',action='store_true',default=True,
        help='Train with Jacobian-free backpropagation (default: True)',
    )
    parser.add_argument('--no_train_jfb',action='store_false',dest='train_jfb',help='Skip the JFB run',)

    parser.add_argument('--train_full_ad',action='store_true',default=False,
        help='Train with full automatic differentiation',)
    
    # parser.add_argument('--train_jbb',action='store_true',default=False,
    #     help='Train the CVXPY / Jacobian-based policy',)

    parser.add_argument("--batch_size", type=int, default=32, help="Trajectories per step (default: 32)")
    parser.add_argument("--nt", type=int, default=20, help="Time steps on [0, t_final] (default: 20)")
    parser.add_argument("--t_final", type=float, default=1.0, help="Horizon (default: 1.0)")
    parser.add_argument("--epochs", type=int, default=300, help="Epochs per trial (default: 500)")
    parser.add_argument("--lr", type=float, default=1.0e-2, help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--n_trials", type=int, default=1, help="Repeated runs. The seed is set once (default: 1)")
    parser.add_argument("--hidden", type=int, default=64, help="Phi hidden width (default: 64)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or cuda:N (default: cpu)")
    parser.add_argument("--seed", type=int, default=1015, help="Random seed (default: 1015)")

    args = parser.parse_args()

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            args.device = "cpu"
        elif ":" in args.device:
            device_id = int(args.device.split(":")[1])
            if device_id >= torch.cuda.device_count():
                print(
                    f"Warning: CUDA device {device_id} not available "
                    f"(only {torch.cuda.device_count()} device(s)), using cuda:0"
                )
                args.device = "cuda:0"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_oc = {
        "batch_size": args.batch_size,
        "nt": args.nt,
        "t_final": args.t_final,
    }
    config_train = {
        "lr": args.lr,
        "epochs": args.epochs,
    }
    #plot_frequency = args.plot_every if args.plot_every > 0 else None

    print(f"device={args.device}  seed={args.seed}  trials={args.n_trials}")
    print(f"JFB={args.train_jfb}  full_AD={args.train_full_ad}")
    #print(f"JFB={args.train_jfb}  full_AD={args.train_full_ad}  JBB={args.train_jbb}")

    for trial in range(args.n_trials):
        print()
        print(f"==================== trial {trial + 1} / {args.n_trials} ====================")
        if args.train_jfb:
            run_s2d_jfb(config_oc,config_train,full_AD_mode=False,device=args.device,)
        if args.train_full_ad:
            run_s2d_jfb(config_oc,config_train,full_AD_mode=True,device=args.device,)
        # if args.train_jbb:
        #     run_s2d_jbb(config_oc,config_train,device=args.device,)


if __name__ == "__main__":
    main()


    





