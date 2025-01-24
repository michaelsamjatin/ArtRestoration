import os
from pathlib import Path
from datetime import datetime
import json
import torch
from utils.train_gan import run_training


if __name__ == '__main__':
    job_id = str(os.environ.get("SLURM_JOB_ID", "_no_id"))
    n_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    task_id = int(os.environ.get('SLURM_PROCID', 0))

    pprint = lambda x: print(f"[{task_id}] {x}")
    pprint(f"Current task {task_id}. Total tasks {n_tasks}.")

    node_list = os.environ.get('SLURM_JOB_NODELIST', None)
    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', None)
    node_id = os.environ.get('SLURM_NODEID', None)
    node_name = os.environ.get('SLURMD_NODENAME', None)
    pprint(f"Node list: {node_list}, Num nodes: {num_nodes}, Node ID: {node_id}, Node name: {node_name}")

    is_cuda = torch.cuda.is_available()
    pprint(f"Is CUDA available: {is_cuda}")
    if is_cuda:
        pprint(f"{torch.cuda.current_device()=}\n{torch.cuda.device_count()=}\n{torch.cuda.get_device_name()=}\n")

    parameter_root = Path("parameters")
    n_params = len(list(parameter_root.glob("*.params")))
    if not n_params == n_tasks:
        raise ValueError(f"Number of tasks ({n_tasks}) does not match number of parameters ({n_params}).")

    with open(parameter_root / f"{task_id}.params", "r") as file:
        params = json.load(file)

    params["device"] = f"cuda:{task_id}"
    params["model_name"] += f".{task_id}"
 
    pprint(f"Running params: {params}")
    run_training(params)
