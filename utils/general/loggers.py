import time
import wandb

#from loguru import logger

class LossTracker:
    """
    For simple tracking of loss values during training phase.
    """
    def __init__(self):
        self.total_loss = 0.0
        self.num_steps = 0

    def update(self, loss):
        self.total_loss += loss
        self.num_steps += 1

    def average_loss(self):
        if self.num_steps == 0:
            return 0
        return self.total_loss / self.num_steps

    def reset(self):
        self.total_loss = 0.0
        self.num_steps = 0
        
        
class TimeTracker:
    """
    For tracking of time comsumption during training phase.
    """
    def __init__(self):
        """
            total_time: total time for each operation
            start_time: latest start time for each operation
            times: total times for each operation
        """
        self.total_time = {}
        self.start_time = {}
        self.times = {}

    def start(self, name):
        self.start_time[name] = time.time()

    def stop(self, name):
        if name not in self.times:
            self.times[name] = 0
            self.total_time[name] = 0.
        if name in self.start_time:
            self.times[name] += 1
            self.total_time[name] += time.time() - self.start_time[name]
            del self.start_time[name]

    def reset(self):
        self.total_time = {}
        self.start_time = {}
        self.times = {}

    def report(self):
        """
        Report the average time comsumption for each operation.
        """
        report = "\n".join([f"{name}: {(self.total_time[name] / self.times[name]):.4f} seconds" for name in self.times.keys()])
        return report

class WandbLogger:
    def __init__(self, wandb_cfg, cfg_dir=None):
        """
        Args:
            wandb_cfg (easydict): configuration file
            log_dir (str): directory to save logs
        """
        self.cfg = wandb_cfg
        wandb.require("core")
        wandb.login()
        wandb.init(
            project=wandb_cfg.PROJECT,
            group=wandb_cfg.GROUP,
            job_type=wandb_cfg.JOB_TYPE,
            name=wandb_cfg.NAME,
            config=wandb_cfg
        )
        
        if cfg_dir is not None:
            wandb.save(cfg_dir, policy="now")

    def log(self, log_dict, step=None):
        if step is None:
            wandb.log(log_dict)
        else:
            wandb.log(log_dict, step=step)

    def finish(self):
        wandb.finish()