import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class DistributedConfig:
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    pp_engine: str = "1f1b"
    backend: str = "nccl"
    use_cpu: bool = False
    
    # Computed fields
    tensor_parallel_size: Optional[int] = None
    context_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    data_parallel_size: Optional[int] = None
    
    # Runtime fields (populated from environment)
    local_rank: Optional[int] = None
    global_rank: Optional[int] = None
    world_size: Optional[int] = None

@dataclass
class ModelConfig:
    name: str
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    dtype: str = "bfloat16"
    use_flash_attention: bool = True
    use_fused_adam: bool = True

@dataclass
class TrainingConfig:
    seed: int = 42
    learning_rate: float = 3e-4
    total_train_steps: int = 200
    seq_length: int = 1024
    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_samples: int = 400000
    max_tokens: Optional[int] = None
    
    # Computed fields
    global_batch_size: Optional[int] = None

@dataclass
class DatasetConfig:
    name: str
    subset_name: Optional[str] = None
    num_workers: int = 0
    num_proc: int = 1
    split: str = "train"

@dataclass
class CheckpointConfig:
    save_dir: str = "ckpt"
    save_frequency: int = 300
    load_path: str = ""

@dataclass
class LoggingConfig:
    use_wandb: bool = False
    project_name: str = "picotron"
    run_name: Optional[str] = None
    run_id: Optional[str] = None

@dataclass
class EnvironmentConfig:
    OMP_NUM_THREADS: str = "1"
    TOKENIZERS_PARALLELISM: str = "false"
    FLASH_ATTEN: str = "1"
    HF_TOKEN: Optional[str] = None

@dataclass
class PicotronConfig:
    distributed: DistributedConfig
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    environment: EnvironmentConfig

    @classmethod
    def load(cls, path: str) -> "PicotronConfig":
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(
            distributed=DistributedConfig(**data.get("distributed", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            environment=EnvironmentConfig(**data.get("environment", {}))
        )
    
    def to_dict(self):
        return asdict(self)
    
    def setup_environment(self):
        """Sets up environment variables based on the configuration."""
        os.environ["OMP_NUM_THREADS"] = self.environment.OMP_NUM_THREADS
        os.environ["TOKENIZERS_PARALLELISM"] = self.environment.TOKENIZERS_PARALLELISM
        os.environ["FLASH_ATTEN"] = self.environment.FLASH_ATTEN
        
        if self.environment.HF_TOKEN is None:
            if "HF_TOKEN" not in os.environ: 
                raise ValueError("HF_TOKEN is neither set in the config file nor in the environment")
        else:
            if "HF_TOKEN" not in os.environ:
                os.environ["HF_TOKEN"] = self.environment.HF_TOKEN
            else:
                print("Warning: HF_TOKEN is set in the environment and the config file. Using the environment variable.")
                
    def setup_distributed_env(self):
        """Reads distributed environment variables and populates the config."""
        self.distributed.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.distributed.global_rank = int(os.environ.get("RANK", 0))
        self.distributed.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Determine backend
        if self.distributed.use_cpu:
            self.distributed.backend = "gloo"
        # If not cpu, we keep the default from config (usually nccl or mps default)
