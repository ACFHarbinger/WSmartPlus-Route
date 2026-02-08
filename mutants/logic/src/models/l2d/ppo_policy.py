from .l2d_policy import L2DPolicy


class L2DPolicy4PPO(L2DPolicy):
    """
    L2DPolicy variant for PPO.
    For now, it's identical to L2DPolicy as the base class supports step-wise decoding.
    """

    pass
