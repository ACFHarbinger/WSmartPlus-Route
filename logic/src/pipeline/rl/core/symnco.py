"""
SymNCO algorithm implementation.
Based on Kim et al. (2022) and Adapted from RL4CO.
"""
from __future__ import annotations

from tensordict import TensorDict

from logic.src.pipeline.rl.core.pomo import POMO
from logic.src.utils.losses import invariance_loss, problem_symmetricity_loss, solution_symmetricity_loss


class SymNCO(POMO):
    """
    SymNCO algorithm: REINFORCE with problem/solution symmetricity and invariance losses.

    Includes support for:
    - Data Augmentation (Dihedral/Symmetric)
    - Multi-start decoding
    - Shared baseline across starts AND augmentations
    - Consistency losses
    """

    def __init__(
        self,
        alpha: float = 0.2,  # weight for invariance loss
        beta: float = 1.0,  # weight for solution symmetricity loss
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> dict:
        """
        SymNCO shared step with symmetricity losses.
        """
        td = self.env.reset(batch)
        bs = td.batch_size[0]

        # Determine number of starts
        n_start = self.num_starts
        if n_start is None:
            if hasattr(self.env, "get_num_starts"):
                n_start = self.env.get_num_starts(td)
            else:
                n_start = td["locs"].shape[1]

        # Augmentation
        n_aug = self.num_augment
        if self.augmentation is not None:
            td = self.augmentation(td)

        # Run policy (must return init_embeds or proj_embeddings)
        out = self.policy(td, self.env, decode_type="sampling" if phase == "train" else "greedy", num_starts=n_start)

        # reward: [batch, n_aug, n_start]
        reward = out["reward"].view(bs, n_aug, n_start)

        if phase == "train":
            # log_likelihood: [batch, n_aug, n_start]
            ll = out["log_likelihood"].view(bs, n_aug, n_start)

            # 1. Problem symmetricity loss (consistency across augmentations)
            # Baseline is mean across augmentations for each start
            loss_ps = problem_symmetricity_loss(reward, ll, dim=1)

            # 2. Solution symmetricity loss (consistency across starts)
            # Baseline is mean across starts for each augmentation
            loss_ss = solution_symmetricity_loss(reward, ll, dim=-1)

            # 3. Invariance loss (invariant representation across augmentations)
            if "proj_embeddings" in out:
                loss_inv = invariance_loss(out["proj_embeddings"], n_aug)
            else:
                loss_inv = 0.0

            loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv

            if self.entropy_weight > 0 and "entropy" in out:
                loss = loss - self.entropy_weight * out["entropy"].mean()

            out["loss"] = loss

            # Update metrics
            best_reward, _ = reward.view(bs, -1).max(dim=-1)
            out["reward"] = best_reward
        else:
            # During val/test, we take the best across starts and augments
            max_reward_per_aug, _ = reward.max(dim=-1)
            best_reward, _ = max_reward_per_aug.max(dim=-1)
            out["reward"] = best_reward

        # Log metrics
        self.log(f"{phase}/reward", out["reward"].mean(), prog_bar=True, sync_dist=True)
        if phase == "train":
            self.log("train/loss_ps", loss_ps, sync_dist=True)
            self.log("train/loss_ss", loss_ss, sync_dist=True)
            if "proj_embeddings" in out:
                self.log("train/loss_inv", loss_inv, sync_dist=True)

        return out
