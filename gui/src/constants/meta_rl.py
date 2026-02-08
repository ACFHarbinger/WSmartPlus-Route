"""
Meta-Reinforcement Learning constants for the GUI.
"""

# Meta-Reinforcement Learning
MRL_METHODS = {
    "": None,
    "Temporal Difference Learning (TDL)": "tdl",
    "Reward Weight Adjustment (RWA)": "rwa",
    "Contextual Bandits (CB)": "cb",
    "Multi-Objective Reinforcement Learning (MORL)": "morl",
}

CB_EXPLORATION_METHODS = {
    "Upper Confidence Bound (UCB)": "ucb",
    "Thompson Sampling": "thompson_sampling",
    "Epsilon-Greedy": "epsilon_greedy",
}

RWA_MODELS = {"Recurrent Neural Network (RNN)": "rnn"}

RWA_OPTIMIZERS = {
    "Adam with Infinity Norm (Adamax)": "adamax",
    "Adaptive Moment Estimation (Adam)": "adam",
    "Adam with Weight Decay (AdamW)": "adamw",
    "Rectified Adam (RAdam)": "radam",
    "Nesterov-accelerated Adam (NAdam)": "nadam",
    "Root Mean Square Propagation (RMSProp)": "rmsprop",
}
