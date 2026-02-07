"""
Job Shop Scheduling Problem (JSSP) Environment.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.envs.generators import JSSPGenerator


class JSSPEnv(RL4COEnvBase):
    """
    Job Shop Scheduling Problem (JSSP) Environment.

    The goal is to schedule J jobs on M machines.
    Each job consists of M operations in a fixed sequence.
    Each operation requires a specific machine and has a duration.

    State:
    - Tracks progress of each job (which operation is next).
    - Tracks global availability time of each machine.

    Action:
    - At each step, select a Job 'j' to schedule its NEXT operation.
    - The operation is scheduled on its required machine 'm' at the earliest possible time
      (max of machine 'm' availability and job 'j' availability).
    """

    name: str = "jssp"

    def __init__(
        self,
        generator: Optional[JSSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """Initialize JSSPEnv."""
        generator_params = generator_params or kwargs
        if generator is None:
            generator = JSSPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize JSSP state."""
        device = td.device
        bs = td.batch_size

        # td contains: "proc_time" (B, J, M), "machine_order" (B, J, M)
        num_jobs = td["proc_time"].shape[1]
        num_machines = td["proc_time"].shape[2]

        # Track next operation index for each job (0 to M-1)
        td["next_op_idx"] = torch.zeros(*bs, num_jobs, dtype=torch.long, device=device)

        # Track availability time of each job (when previous op finishes)
        td["job_avail_time"] = torch.zeros(*bs, num_jobs, dtype=torch.float, device=device)

        # Track availability time of each machine
        td["machine_avail_time"] = torch.zeros(*bs, num_machines, dtype=torch.float, device=device)

        # Track visited/finished jobs
        td["finished_jobs"] = torch.zeros(*bs, num_jobs, dtype=torch.bool, device=device)

        # Track current makespan (max completion time so far)
        td["makespan"] = torch.zeros(*bs, 1, dtype=torch.float, device=device)

        # Keep track of count of steps to detect done
        td["steps_taken"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)

        # Precompute total operations for done check
        td["total_ops"] = torch.tensor(num_jobs * num_machines, device=device).expand(*bs, 1)

        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action (schedule next op of selected job)."""
        # Action is job index j in [0, J-1]
        action = td["action"]  # (B,)

        # Gather info for the selected job
        # 1. Get next op index
        job_idx = action
        next_op_idx = td["next_op_idx"].gather(1, job_idx.unsqueeze(-1)).squeeze(-1)  # (B,)

        # 2. Get required machine for this op
        # machine_order: (B, J, M)
        # We need machine_order[b, job_idx, next_op_idx]
        # This double gathering is tricky in pure pytorch without explicit indices for batch
        # Use gather on flattened or specific dim

        # (B, J, M) -> select job -> (B, 1, M) -> select op -> (B, 1, 1)
        selected_job_machines = td["machine_order"].gather(
            1, job_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, td["machine_order"].shape[-1])
        )
        required_machine = (
            selected_job_machines.gather(2, next_op_idx.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        )  # (B,)

        # 3. Get duration
        selected_job_times = td["proc_time"].gather(
            1, job_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, td["proc_time"].shape[-1])
        )
        duration = selected_job_times.gather(2, next_op_idx.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (B,)

        # 4. Calculate start time
        # Start = max(job available time, machine available time)
        job_avail = td["job_avail_time"].gather(1, job_idx.unsqueeze(-1)).squeeze(-1)  # (B,)
        machine_avail = td["machine_avail_time"].gather(1, required_machine.unsqueeze(-1)).squeeze(-1)  # (B,)

        start_time = torch.max(job_avail, machine_avail)
        end_time = start_time + duration

        # 5. Update state

        # Update Machine Availability
        # Scatter back to machine_avail_time
        td["machine_avail_time"] = td["machine_avail_time"].scatter(
            1, required_machine.unsqueeze(-1), end_time.unsqueeze(-1)
        )

        # Update Job Availability
        td["job_avail_time"] = td["job_avail_time"].scatter(1, job_idx.unsqueeze(-1), end_time.unsqueeze(-1))

        # Update Next Op Index
        td["next_op_idx"] = td["next_op_idx"].scatter_add(
            1, job_idx.unsqueeze(-1), torch.ones_like(job_idx).unsqueeze(-1)
        )

        # Update Finished Status
        # If next_op_idx equals num_machines, job is done
        num_machines = td["proc_time"].shape[2]
        # We just incremented next_op_idx. Check if it equals num_machines
        # Re-gather updated next_op_idx
        updated_op_idx = td["next_op_idx"].gather(1, job_idx.unsqueeze(-1)).squeeze(-1)
        job_finished = updated_op_idx >= num_machines

        # Only update finished_jobs for the selected job if it is actually finished
        # We act on the boolean mask
        current_finished = td["finished_jobs"].gather(1, job_idx.unsqueeze(-1)).squeeze(-1)
        new_finished = current_finished | job_finished
        td["finished_jobs"] = td["finished_jobs"].scatter(1, job_idx.unsqueeze(-1), new_finished.unsqueeze(-1))

        # Update global makespan
        td["makespan"] = torch.max(td["makespan"], end_time.unsqueeze(-1))

        # Update steps
        td["steps_taken"] += 1

        return td

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Valid actions are any unfinished jobs.
        """
        # finished_jobs is True if done
        # Mask needs to be True for VALID actions (unfinished)
        return ~td["finished_jobs"]

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reward is negative makespan."""
        return -td["makespan"]

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when all operations scheduled."""
        # Either check steps == total_ops or all jobs finished
        return td["finished_jobs"].all(dim=-1)
