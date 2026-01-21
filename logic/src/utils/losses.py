"""
Consistency and symmetry losses for NCO.
"""
import torch
import torch.nn.functional as F


def problem_symmetricity_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Consistency loss across problem augmentations."""
    if reward.shape[dim] < 2:
        return torch.tensor(0.0, device=reward.device)
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -(advantage.detach() * log_likelihood)
    return loss.mean()


def solution_symmetricity_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Consistency loss across solution starts."""
    if reward.shape[dim] < 2:
        return torch.tensor(0.0, device=reward.device)
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -(advantage.detach() * log_likelihood)
    return loss.mean()


def invariance_loss(proj_embed: torch.Tensor, num_augment: int) -> torch.Tensor:
    """Loss for invariant representation across augmentations."""
    # proj_embed: [batch * num_augment, graph_size, d] or [batch * num_augment, d]
    # Handle both cases by flattening if needed
    if proj_embed.dim() == 3:
        # [batch * num_augment, graph_size, d] -> [batch * num_augment, graph_size * d]
        proj_embed = proj_embed.view(proj_embed.shape[0], -1)

    bs = proj_embed.shape[0] // num_augment
    pe = proj_embed.view(bs, num_augment, -1)

    # Cosine similarity between first augmentation and others
    similarity = 0
    ref = pe[:, 0]
    for i in range(1, num_augment):
        similarity += F.cosine_similarity(ref, pe[:, i], dim=-1)

    # We want to maximize similarity, so minimize negative similarity
    return -(similarity / (num_augment - 1)).mean()
