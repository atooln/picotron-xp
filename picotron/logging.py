import time
import os
import psutil
from typing import Dict, Any, Optional
import wandb
import picotron.process_group_manager as pgm
from picotron.utils import to_readable_format

class ExperimentLogger:
    """
    A Pythonic, unified logger for distributed training.
    Handles WandB, Console printing, and System Monitoring (M1/Mac).
    """
    def __init__(self, config: Dict[str, Any], group_name: str = "picotron"):
        self.config = config
        self.step = 0
        self.start_time = time.time()
        
        # Determine if this process is responsible for logging
        # (Rank 0 of the last pipeline stage)
        self.is_main_rank = (
            pgm.process_group_manager.tp_rank == 0 and 
            pgm.process_group_manager.dp_rank == 0 and 
            pgm.process_group_manager.cp_rank == 0 and 
            pgm.process_group_manager.pp_is_last_stage
        )

        # Initialize system metrics state
        self.last_system_metrics = self._get_system_metrics() if self.is_main_rank else {}

        if self.is_main_rank and config["logging"]["use_wandb"]:
            wandb.init(
                project=group_name,
                name=f"{config['logging']['run_name']}",
                config=config,
                # Resume allowed if run_id provided, otherwise new run
                resume="allow", 
                id=config["logging"].get("run_id", None) 
            )

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        The unified entry point for all logging.
        """
        if not self.is_main_rank:
            return

        if step is not None:
            self.step = step
        else:
            self.step += 1

        # 1. Enrich with M1 System Metrics (every 10 steps to save overhead)
        if self.step % 10 == 0:
            self.last_system_metrics = self._get_system_metrics()
        
        metrics.update(self.last_system_metrics)

        # 2. Log to WandB
        if self.config["logging"]["use_wandb"]:
            wandb.log(metrics, step=self.step)

        # 3. Print to Console
        self._print_console(metrics)

    def _get_system_metrics(self) -> Dict[str, float]:
        """Captures hardware state (Swap/CPU) for performance debugging."""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "sys/ram_pct": vm.percent,
            "sys/swap_gb": swap.used / 1e9,
            "sys/cpu_pct": psutil.cpu_percent(),
        }

    def _print_console(self, metrics: Dict[str, float]):
        """Formats the output beautifully for the terminal."""
        # Only print keys we care about in the terminal to keep it clean
        loss = metrics.get("train/loss", 0.0)
        tps = metrics.get("perf/tokens_per_sec", 0.0)
        mfu = metrics.get("perf/mfu", 0.0)
        swap = metrics.get("sys/swap_gb", 0.0)
        gpu_mem = metrics.get("perf/memory_usage_gb", 0.0)
        ram_pct = metrics.get("sys/ram_pct", 0.0)
        
        # Use simple f-strings for readability
        print(
            f"[Step {self.step:<5}] "
            f"Loss: {loss:6.4f} | "
            f"T/s: {to_readable_format(tps):>7} | "
            f"MFU: {mfu:5.2f}% | "
            f"Mem: {gpu_mem:5.2f}GB | "
            f"Pressure: {ram_pct:3.0f}%"
            f"{f' | Swap: {swap:.1f}GB' if swap > 0.1 else ''}" # Alert if swapping
        )

    def finish(self):
        if self.is_main_rank and self.config["logging"]["use_wandb"]:
            wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
