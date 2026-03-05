import contextlib
from abc import abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from .base import RLAgent


class ContextualBanditAgent(RLAgent):
    """
    Base class for Contextual Multi-Armed Bandit (CMAB) agents.

    Unlike traditional bandits, contextual bandits use external features (context)
    to inform arm selection. This class maintains basic environmental metadata
    and handles random state management.

    Attributes:
        n_arms: Number of available actions.
        feature_dim: Dimension of the context feature vector.
        t: Global trial counter.
        rng: Random number generator.
        reward_history: Dictionary mapping arm indices to deques of recent rewards.
    """

    def __init__(self, n_arms: int, feature_dim: int, seed: Optional[int] = None, history_size: int = 50):
        """
        Initialize the contextual bandit agent.

        Args:
            n_arms: Number of available actions.
            feature_dim: Dimension of the context feature vector.
            seed: Optional seed for the random number generator.
            history_size: Size of the reward tracking buffer.
        """
        self.n_arms = n_arms
        self.feature_dim = feature_dim
        self.t = 0
        self.rng = np.random.default_rng(seed)

        self.reward_history = {i: deque(maxlen=history_size) for i in range(n_arms)}

    def decay_epsilon(self) -> None:
        """Optional exploration rate decay (if applicable)."""
        pass

    @abstractmethod
    def select_action(self, context: np.ndarray, rng: np.random.Generator) -> int:
        pass

    @abstractmethod
    def get_weights(self) -> Any:
        """Return the current weights/parameters of the agent."""
        pass


class LinUCBAgent(ContextualBanditAgent):
    """
    Linear Upper Confidence Bound (LinUCB) agent.

    Maintains a linear model per arm for expected rewards given context.
    Algorithm: LinUCB with Disjoint Linear Models.

    Reference: Li et al., "A Contextual-Bandit Approach to Personalized
    News Article Recommendation", WWW 2010.
    """

    def __init__(
        self,
        n_arms: int,
        feature_dim: int,
        alpha: float = 1.0,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the LinUCB agent.

        Args:
            n_arms: Number of available arms.
            feature_dim: Dimension of context feature vector (x).
            alpha: Exploration parameter (controls width of UCB bonus).
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, feature_dim, seed, history_size)
        self.alpha = alpha

        # A: dxd design matrix for each arm (feature outer products)
        # Initialized as Identity matrix for Ridge Regression regularization
        self.A = [np.identity(feature_dim) for _ in range(n_arms)]

        # b: dx1 response vector for each arm (feature * reward)
        self.b = [np.zeros(feature_dim) for _ in range(n_arms)]

    def select_action(self, context: np.ndarray, rng: np.random.Generator) -> int:
        """
        Select an arm using the LinUCB policy.

        Args:
            context: Current context feature vector [d].
            rng: Random number generator for tie-breaking.

        Returns:
            The selected arm index.
        """
        self.t += 1
        ucb_values = []

        for arm in range(self.n_arms):
            # Compute ridge regression weights: theta_hat = A^(-1) * b
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]

            # Linear prediction: p = theta_hat^T * context
            expected_reward = theta @ context

            # Confidence bonus: alpha * sqrt(context^T * A^(-1) * context)
            confidence_bonus = self.alpha * np.sqrt(context @ A_inv @ context)

            # Combined score for UCB selection
            p = expected_reward + confidence_bonus
            ucb_values.append(p)

        # Select arm with the highest UCB value
        max_p = np.max(ucb_values)
        # Tie-breaking with small epsilon buffer
        best_arms = np.where(np.abs(ucb_values - max_p) < 1e-9)[0]

        return int(rng.choice(best_arms))

    def update(
        self, context: np.ndarray, action: int, reward: float, next_context: Any = None, done: bool = False
    ) -> None:
        """
        Update the linear model for the selected arm with observed reward.

        Args:
            context: Context vector associated with the choice.
            action: Index of the selected arm.
            reward: Observed reward.
            next_context: Unused in bandits.
            done: Unused in bandits.
        """
        # Standard tracking
        self.reward_history[action].append(reward)

        # Update design matrix: A = A + x * x^T
        self.A[action] += np.outer(context, context)

        # Update response vector: b = b + r * x
        self.b[action] += reward * context

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic info for the LinUCB agent.

        Returns:
            Dictionary containing trial count, parameters, and recent performance.
        """
        stats = {
            "t": self.t,
            "n_arms": self.n_arms,
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "avg_history_rewards": {i: float(np.mean(h)) if h else 0.0 for i, h in self.reward_history.items()},
        }
        return stats

    def get_weights(self) -> List[np.ndarray]:
        """
        Calculate current theta estimates (weights) for all arms.

        Returns:
            List of NumPy arrays, one per arm.
        """
        weights = []
        for arm in range(self.n_arms):
            # weights = A^(-1) * b
            A_inv = np.linalg.inv(self.A[arm])
            weights.append(A_inv @ self.b[arm])
        return weights


class ContextualThompsonSamplingAgent(ContextualBanditAgent):
    """
    Contextual Thompson Sampling agent with linear Gaussian model.

    Bayesian approach to contextual bandits. Maintains a posterior distribution
    over the weights of a linear reward model. Selects arms by sampling weights
    from the posterior and playing the greedy action for the sample.

    Reference: Agrawal & Goyal, "Thompson Sampling for Contextual Bandits
    with Linear Payoffs", ICML 2013.
    """

    def __init__(
        self,
        n_arms: int,
        feature_dim: int,
        lambda_prior: float = 1.0,
        noise_variance: float = 0.1,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Contextual Thompson Sampling agent.

        Args:
            n_arms: Number of available actions.
            feature_dim: Dimension of context feature vector (d).
            lambda_prior: Regularization parameter (initial precision).
            noise_variance: Assumed variance of the reward noise.
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, feature_dim, seed, history_size)
        self.lambda_prior = lambda_prior
        self.noise_variance = noise_variance

        # B: Precision matrix (inverse covariance) for each arm.
        # Initialized to lambda * Identity (Ridge prior).
        self.B = [lambda_prior * np.identity(feature_dim) for _ in range(n_arms)]

        # mu: Current mean weight vector estimate for each arm.
        self.mu = [np.zeros(feature_dim) for _ in range(n_arms)]

        # f: Accumulator for mu calculation (B^-1 * f = mu).
        # tracks (1/v) * sum(reward_i * context_i).
        self.f = [np.zeros(feature_dim) for _ in range(n_arms)]

    def select_action(self, context: np.ndarray, rng: np.random.Generator) -> int:
        """
        Select an arm using Thompson sampling from the linear posterior.

        Args:
            context: Current context feature vector [d].
            rng: Random number generator.

        Returns:
            The selected arm index.
        """
        self.t += 1
        sampled_rewards = []

        for arm in range(self.n_arms):
            try:
                # Calculate current covariance matrix (B^-1)
                B_inv = np.linalg.inv(self.B[arm])

                # Sample weight vector 'theta' from the posterior Normal distribution
                # Posterior is N(mu, B^-1)
                theta_sample = rng.multivariate_normal(self.mu[arm], B_inv)
            except np.linalg.LinAlgError:
                # Fallback to mean if matrix inversion fails
                theta_sample = self.mu[arm]

            # Predicted reward for this sample
            sampled_rewards.append(theta_sample @ context)

        # Select arm that maximizes the sampled reward
        max_r = np.max(sampled_rewards)
        best_arms = np.where(np.abs(sampled_rewards - max_r) < 1e-9)[0]

        return int(rng.choice(best_arms))

    def update(
        self, context: np.ndarray, action: int, reward: float, next_context: Any = None, done: bool = False
    ) -> None:
        """
        Update the posterior distribution for the selected arm index.

        Args:
            context: Context vector associated with the choice.
            action: Selected arm index.
            reward: Observed reward.
            next_context: Unused.
            done: Unused.
        """
        # Track history
        self.reward_history[action].append(reward)

        # Update precision matrix: B = B + (1/v) * x * x^T
        inv_v = 1.0 / max(self.noise_variance, 1e-8)
        self.B[action] += inv_v * np.outer(context, context)

        # Update accumulator: f = f + (1/v) * r * x
        self.f[action] += inv_v * reward * context

        # Re-calculate posterior mean: mu = B^(-1) * f
        with contextlib.suppress(np.linalg.LinAlgError):
            self.mu[action] = np.linalg.inv(self.B[action]) @ self.f[action]

    def reset(self) -> None:
        """Reset the posterior to the initial prior for all arms."""
        super().reset()
        self.t = 0
        self.B = [self.lambda_prior * np.identity(self.feature_dim) for _ in range(self.n_arms)]
        self.mu = [np.zeros(self.feature_dim) for _ in range(self.n_arms)]
        self.f = [np.zeros(self.feature_dim) for _ in range(self.n_arms)]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic info for the Thompson Sampling agent.

        Returns:
            Dictionary containing trials, prior/noise parameters, and recent rewards.
        """
        return {
            "t": self.t,
            "n_arms": self.n_arms,
            "feature_dim": self.feature_dim,
            "noise_variance": self.noise_variance,
            "lambda_prior": self.lambda_prior,
            "avg_history_rewards": {i: float(np.mean(h)) if h else 0.0 for i, h in self.reward_history.items()},
        }

    def get_weights(self) -> List[np.ndarray]:
        """Return the current posterior means for each arm."""
        return [m.copy() for m in self.mu]


class GPCMABAgent(ContextualBanditAgent):
    """
    Gaussian Process Combinatorial Multi-Armed Bandit (GP-CMAB) agent.

    Models the reward function as a Gaussian Process (GP) over the joint
    (context, action) space. Uses the GP-UCB acquisition function for
    arm selection, balancing exploration (high posterior variance) with
    exploitation (high posterior mean). Supports combinatorial "super-arm"
    selection where multiple base arms are chosen simultaneously.

    The GP prior uses a Radial Basis Function (RBF) kernel:

        k(x, x') = signal_variance * exp(-||x - x'||^2 / (2 * length_scale^2))

    Posterior updates follow standard GP regression equations:

        mu(x*) = k(x*, X) * [K(X, X) + sigma^2 * I]^(-1) * y
        sigma^2(x*) = k(x*, x*) - k(x*, X) * [K(X, X) + sigma^2 * I]^(-1) * k(X, x*)

    For combinatorial selection (super-arms), the agent greedily constructs a
    subset of ``k`` arms by iteratively selecting the arm with the highest GP-UCB
    score, making it suitable for problems where multiple operators are activated
    simultaneously (e.g., selecting a portfolio of destroy + repair operators).

    Reference:
        Srinivas et al., "Gaussian Process Optimization in the Bandit Setting:
        No Regret and Experimental Design", ICML 2010.
        Chen et al., "Combinatorial Multi-Armed Bandit: General Framework and
        Applications", ICML 2013.

    Attributes:
        n_arms: Number of base arms (individual operators).
        feature_dim: Dimension of the context feature vector.
        beta: Exploration parameter for the GP-UCB acquisition function.
        length_scale: Length scale of the RBF kernel.
        signal_variance: Output variance of the RBF kernel.
        noise_variance: Observation noise variance (sigma^2).
        max_history: Maximum number of observations to retain.
        super_arm_size: Number of arms to select per round (1 = standard bandit).
        X_history: List of observed (context, action_one_hot) input vectors.
        y_history: List of observed reward values.
        K_inv: Cached inverse of the kernel matrix (for incremental updates).
        alpha_vec: Cached K_inv @ y vector (for fast posterior mean computation).
    """

    def __init__(
        self,
        n_arms: int,
        feature_dim: int,
        beta: float = 2.0,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 0.1,
        max_history: int = 500,
        super_arm_size: int = 1,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the GP-CMAB agent.

        Args:
            n_arms: Number of individual base arms.
            feature_dim: Dimension of the context feature vector.
            beta: Exploration weight for GP-UCB (controls optimism).
                Higher values encourage more exploration. The theoretical
                recommendation is ``beta_t = 2 * log(|D| * t^2 * pi^2 / 6 / delta)``
                for a fixed confidence ``1-delta``, but a constant value works
                well in practice.
            length_scale: RBF kernel length scale. Controls how quickly the
                correlation between points decays with distance.
            signal_variance: RBF kernel output variance. Scales the prior
                amplitude of the GP.
            noise_variance: Observation noise variance (sigma^2). Assumed i.i.d.
                Gaussian noise on reward observations.
            max_history: Maximum number of data points to retain. When exceeded,
                the oldest observations are pruned to maintain
                computational tractability (O(n^3) inversion).
            super_arm_size: Number of arms to select per round. If > 1, the
                agent performs combinatorial (greedy) selection.
            seed: Optional seed for deterministic reproducibility.
            history_size: Size of per-arm reward history buffer for diagnostics.
        """
        super().__init__(n_arms, feature_dim, seed, history_size)
        self.beta = beta
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.max_history = max_history
        self.super_arm_size = min(super_arm_size, n_arms)

        # The GP input dimension is context + one-hot action encoding
        self.input_dim = feature_dim + n_arms

        # Observation history
        self.X_history: List[np.ndarray] = []
        self.y_history: List[float] = []

        # Cached kernel matrix inverse and alpha vector for O(1) posterior queries.
        # These are lazily initialized after the first observation.
        self.K_inv: Optional[np.ndarray] = None
        self.alpha_vec: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_input(self, context: np.ndarray, action: int) -> np.ndarray:
        """
        Encode a (context, action) pair into a single GP input vector.

        The action is represented as a one-hot vector appended to the context,
        creating a joint feature space that allows the GP to learn action-specific
        reward functions conditioned on context.

        Args:
            context: Context feature vector of shape [feature_dim].
            action: Index of the selected arm.

        Returns:
            Concatenated input vector of shape [feature_dim + n_arms].
        """
        one_hot = np.zeros(self.n_arms)
        one_hot[action] = 1.0
        return np.concatenate([context, one_hot])

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the Radial Basis Function (RBF / Squared Exponential) kernel.

        Formula:
            k(x1, x2) = sigma_f^2 * exp(-||x1 - x2||^2 / (2 * l^2))

        where ``sigma_f^2`` is the signal variance and ``l`` is the length scale.

        Args:
            x1: First input vector.
            x2: Second input vector.

        Returns:
            Scalar kernel value.
        """
        sq_dist = np.sum((x1 - x2) ** 2)
        return float(self.signal_variance * np.exp(-sq_dist / (2.0 * self.length_scale**2)))

    def _kernel_vector(self, x_star: np.ndarray) -> np.ndarray:
        """
        Compute kernel values between a query point and all training points.

        Args:
            x_star: Query input vector of shape [input_dim].

        Returns:
            Vector of kernel values k(x*, X) of shape [n_observations].
        """
        if not self.X_history:
            return np.array([])
        return np.array([self._rbf_kernel(x_star, x_i) for x_i in self.X_history])

    def _predict(self, context: np.ndarray, action: int) -> tuple:
        """
        Compute GP posterior mean and variance for a (context, action) pair.

        Uses cached ``K_inv`` and ``alpha_vec`` for efficient computation:

            mu(x*) = k(x*, X)^T @ alpha_vec
            sigma^2(x*) = k(x*, x*) - k(x*, X)^T @ K_inv @ k(x*, X)

        Args:
            context: Context feature vector.
            action: Arm index.

        Returns:
            Tuple of (posterior_mean, posterior_variance).
        """
        x_star = self._encode_input(context, action)

        # Prior prediction (no data yet)
        if self.K_inv is None or len(self.X_history) == 0:
            return 0.0, self.signal_variance

        # Step 1: Compute kernel vector between query and all training points
        k_star = self._kernel_vector(x_star)

        # Step 2: Posterior mean = k^T @ alpha  where alpha = K_inv @ y
        mu = float(k_star @ self.alpha_vec)

        # Step 3: Posterior variance = k(x*, x*) - k^T @ K_inv @ k
        k_star_star = self._rbf_kernel(x_star, x_star)
        v = self.K_inv @ k_star
        var = k_star_star - float(k_star @ v)

        # Clamp variance to avoid numerical issues (must remain non-negative)
        var = max(var, 1e-10)

        return mu, var

    def _recompute_kernel_inverse(self) -> None:
        """
        Fully recompute the kernel matrix and its inverse from scratch.

        Constructs the full kernel matrix ``K(X, X) + sigma^2 * I`` and computes
        the inverse. This is O(n^3) and is only called when incremental
        updates are not possible (e.g., after history pruning).
        """
        n = len(self.X_history)
        if n == 0:
            self.K_inv = None
            self.alpha_vec = None
            return

        # Build symmetric kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k_val = self._rbf_kernel(self.X_history[i], self.X_history[j])
                K[i, j] = k_val
                K[j, i] = k_val

        # Add observation noise to diagonal
        K += self.noise_variance * np.eye(n)

        # Compute inverse with jitter fallback for numerical stability
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))

        y_arr = np.array(self.y_history)
        self.alpha_vec = self.K_inv @ y_arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, context: np.ndarray, rng: np.random.Generator) -> int:
        """
        Select an arm (or the primary arm of a super-arm) using GP-UCB.

        GP-UCB score for arm ``a`` given context ``x``:

            score(a) = mu(x, a) + sqrt(beta) * sigma(x, a)

        For combinatorial selection (``super_arm_size > 1``), arms are selected
        greedily: the arm with the highest GP-UCB score is chosen first,
        then the next best among remaining arms, and so on. The first arm
        in the selected set is returned as the primary action.

        Args:
            context: Current context feature vector [feature_dim].
            rng: Random number generator for tie-breaking.

        Returns:
            The selected arm index (primary arm if combinatorial).
        """
        self.t += 1
        ucb_scores = np.zeros(self.n_arms)

        # Compute GP-UCB acquisition score for each arm
        for arm in range(self.n_arms):
            mu, var = self._predict(context, arm)
            # GP-UCB: mu + sqrt(beta) * sigma
            ucb_scores[arm] = mu + np.sqrt(self.beta) * np.sqrt(var)

        if self.super_arm_size == 1:
            # Standard (non-combinatorial) selection
            max_score = np.max(ucb_scores)
            best_arms = np.where(np.abs(ucb_scores - max_score) < 1e-9)[0]
            return int(rng.choice(best_arms))

        # Combinatorial: greedily select top-k arms, return the first
        remaining_scores = ucb_scores.copy()
        for _ in range(self.super_arm_size):
            max_score = np.max(remaining_scores)
            best = np.where(np.abs(remaining_scores - max_score) < 1e-9)[0]
            chosen = int(rng.choice(best))
            remaining_scores[chosen] = -np.inf
            # Return the very first (highest-scoring) arm immediately
            return chosen

        # Fallback (should never reach here)
        return int(rng.integers(0, self.n_arms))  # pragma: no cover

    def select_super_arm(self, context: np.ndarray, rng: np.random.Generator) -> List[int]:
        """
        Select a combinatorial super-arm (ordered subset of arms).

        Unlike ``select_action`` which returns only the primary arm, this
        method returns the complete ordered list of selected arms, suitable
        for problems that require activating multiple operators per round.

        Args:
            context: Current context feature vector [feature_dim].
            rng: Random number generator for tie-breaking.

        Returns:
            Ordered list of selected arm indices (length = super_arm_size).
        """
        self.t += 1
        ucb_scores = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            mu, var = self._predict(context, arm)
            ucb_scores[arm] = mu + np.sqrt(self.beta) * np.sqrt(var)

        selected_arms: List[int] = []
        remaining_scores = ucb_scores.copy()
        for _ in range(self.super_arm_size):
            max_score = np.max(remaining_scores)
            best = np.where(np.abs(remaining_scores - max_score) < 1e-9)[0]
            chosen = int(rng.choice(best))
            selected_arms.append(chosen)
            remaining_scores[chosen] = -np.inf

        return selected_arms

    def update(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
        next_context: Any = None,
        done: bool = False,
    ) -> None:
        """
        Update the GP model with a new observation (context, action, reward).

        Uses the Woodbury identity for efficient rank-1 incremental update
        of the kernel matrix inverse, avoiding full O(n^3) re-inversion:

        Let ``k = k(X, x_new)``, ``kappa = k(x_new, x_new) + sigma^2``::

            K_inv_new = [[K_inv + (K_inv @ k @ k^T @ K_inv) / s, -K_inv @ k / s],
                         [-k^T @ K_inv / s,                       1 / s         ]]

        where ``s = kappa - k^T @ K_inv @ k`` (Schur complement).

        Args:
            context: Context vector associated with the observation.
            action: Index of the arm that was played.
            reward: Observed reward.
            next_context: Unused (bandit setting).
            done: Unused (bandit setting).
        """
        # Track per-arm reward history for diagnostics
        self.reward_history[action].append(reward)

        # Encode the observation into the GP input space
        x_new = self._encode_input(context, action)

        # Prune oldest observations if history exceeds maximum
        if len(self.X_history) >= self.max_history:
            self.X_history.pop(0)
            self.y_history.pop(0)
            # Force full recomputation since history shifted
            self.K_inv = None
            self.alpha_vec = None

        # Append new observation
        self.X_history.append(x_new)
        self.y_history.append(reward)

        n = len(self.X_history)

        if n == 1:
            # First observation: trivial 1x1 kernel matrix
            kappa = self._rbf_kernel(x_new, x_new) + self.noise_variance
            self.K_inv = np.array([[1.0 / kappa]])
            self.alpha_vec = np.array([reward / kappa])
            return

        if self.K_inv is not None and self.K_inv.shape[0] == n - 1:
            # --- Rank-1 incremental update via Woodbury identity ---

            # Step 1: Kernel vector between new point and existing points
            k = np.array([self._rbf_kernel(x_new, x_i) for x_i in self.X_history[:-1]])

            # Step 2: Self-kernel + noise
            kappa = self._rbf_kernel(x_new, x_new) + self.noise_variance

            # Step 3: Schur complement s = kappa - k^T @ K_inv_old @ k
            K_inv_k = self.K_inv @ k
            s = kappa - float(k @ K_inv_k)
            s = max(s, 1e-10)  # Numerical stability clamp

            # Step 4: Block-wise expansion of K_inv
            K_inv_new = np.zeros((n, n))
            K_inv_new[: n - 1, : n - 1] = self.K_inv + np.outer(K_inv_k, K_inv_k) / s
            K_inv_new[: n - 1, n - 1] = -K_inv_k / s
            K_inv_new[n - 1, : n - 1] = -K_inv_k / s
            K_inv_new[n - 1, n - 1] = 1.0 / s

            self.K_inv = K_inv_new
        else:
            # Full recomputation (after pruning or initialization mismatch)
            self._recompute_kernel_inverse()

        # Update alpha: alpha = K_inv @ y
        y_arr = np.array(self.y_history)
        self.alpha_vec = self.K_inv @ y_arr

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic statistics about the GP-CMAB agent.

        Returns:
            Dictionary containing trial count, kernel parameters, observation
            count, and per-arm average rewards.
        """
        return {
            "t": self.t,
            "n_arms": self.n_arms,
            "feature_dim": self.feature_dim,
            "beta": self.beta,
            "length_scale": self.length_scale,
            "signal_variance": self.signal_variance,
            "noise_variance": self.noise_variance,
            "n_observations": len(self.X_history),
            "max_history": self.max_history,
            "super_arm_size": self.super_arm_size,
            "avg_history_rewards": {i: float(np.mean(h)) if h else 0.0 for i, h in self.reward_history.items()},
        }

    def get_weights(self) -> Optional[np.ndarray]:
        """
        Return the current GP alpha vector (K_inv @ y).

        Unlike linear models, a GP does not have a single weight vector.
        The alpha vector serves as the closest analogue, encoding the
        influence of each training observation on the posterior mean.

        Returns:
            The alpha vector of shape [n_observations], or None if no data.
        """
        if self.alpha_vec is not None:
            return self.alpha_vec.copy()
        return None

    def save(self, path: str) -> None:
        """
        Serialize the GP-CMAB agent state to disk.

        Args:
            path: Absolute path to the destination file.
        """
        import pickle

        state = {
            "n_arms": self.n_arms,
            "feature_dim": self.feature_dim,
            "beta": self.beta,
            "length_scale": self.length_scale,
            "signal_variance": self.signal_variance,
            "noise_variance": self.noise_variance,
            "max_history": self.max_history,
            "super_arm_size": self.super_arm_size,
            "t": self.t,
            "X_history": self.X_history,
            "y_history": self.y_history,
            "K_inv": self.K_inv,
            "alpha_vec": self.alpha_vec,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """
        Deserialize the GP-CMAB agent state from disk.

        Args:
            path: Absolute path to the source file.
        """
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.n_arms = state["n_arms"]
        self.feature_dim = state["feature_dim"]
        self.beta = state["beta"]
        self.length_scale = state["length_scale"]
        self.signal_variance = state["signal_variance"]
        self.noise_variance = state["noise_variance"]
        self.max_history = state["max_history"]
        self.super_arm_size = state["super_arm_size"]
        self.t = state["t"]
        self.X_history = state["X_history"]
        self.y_history = state["y_history"]
        self.K_inv = state["K_inv"]
        self.alpha_vec = state["alpha_vec"]
        self.input_dim = self.feature_dim + self.n_arms

    def reset(self) -> None:
        """Reset the GP model to its prior (discard all observations)."""
        self.t = 0
        self.X_history = []
        self.y_history = []
        self.K_inv = None
        self.alpha_vec = None
        self.reward_history = {i: deque(maxlen=50) for i in range(self.n_arms)}
