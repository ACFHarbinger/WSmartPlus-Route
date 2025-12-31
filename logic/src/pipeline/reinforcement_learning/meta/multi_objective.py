import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
from sklearn.cluster import KMeans
from logic.src.pipeline.reinforcement_learning.meta.weight_strategy import WeightAdjustmentStrategy


class ParetoSolution:
    """
    Represents a solution on the Pareto front, storing policy parameters,
    weights, and objective values.
    """
    def __init__(self, weights, objectives, reward, policy_params=None, model_id=None):
        """
        Args:
            weights: Dictionary of weight values
            objectives: Dictionary of objective values (waste_efficiency, overflow_rate)
            reward: Scalar reward achieved
            policy_params: Optional saved model parameters
            model_id: Identifier for this solution
        """
        self.weights = copy.deepcopy(weights)
        self.objectives = copy.deepcopy(objectives)
        self.reward = reward
        self.policy_params = policy_params  # Can be None or actual parameters
        self.model_id = model_id
        self.dominated = False
        
    def dominates(self, other):
        """
        Check if this solution dominates another solution.
        A solution dominates another if it's at least as good in all objectives
        and strictly better in at least one objective.
        
        Args:
            other: Another ParetoSolution
            
        Returns:
            True if this solution dominates the other
        """
        # For waste efficiency, higher is better
        waste_better = self.objectives['waste_efficiency'] >= other.objectives['waste_efficiency']
        # For overflow rate, lower is better
        overflow_better = self.objectives['overflow_rate'] <= other.objectives['overflow_rate']
        
        # Must be better in at least one objective
        strictly_better = (
            (self.objectives['waste_efficiency'] > other.objectives['waste_efficiency']) or
            (self.objectives['overflow_rate'] < other.objectives['overflow_rate'])
        )
        
        return waste_better and overflow_better and strictly_better


class ParetoFront:
    """
    Maintains a set of non-dominated solutions forming a Pareto front.
    """
    def __init__(self, max_size=50):
        """
        Args:
            max_size: Maximum number of solutions to keep on the front
        """
        self.solutions = []
        self.max_size = max_size
        
    def add_solution(self, solution):
        """
        Add a solution to the Pareto front, removing dominated solutions.
        
        Args:
            solution: ParetoSolution to add
            
        Returns:
            True if solution was added (is non-dominated)
        """
        # Check if this solution is dominated by any existing solution
        for existing in self.solutions:
            if existing.dominates(solution):
                return False
        
        # Remove any solutions that are dominated by this new solution
        self.solutions = [s for s in self.solutions if not solution.dominates(s)]
        
        # Add the new solution
        self.solutions.append(solution)
        
        # If we exceed max size, use clustering to select representative solutions
        if len(self.solutions) > self.max_size:
            self._prune_with_clustering()
            
        return True
    
    def _prune_with_clustering(self):
        """
        Reduce the number of solutions using K-means clustering,
        keeping solutions closest to cluster centers.
        """
        if len(self.solutions) <= self.max_size:
            return
        
        # Extract objectives for clustering
        objectives = np.array([
            [s.objectives['waste_efficiency'], s.objectives['overflow_rate']]
            for s in self.solutions
        ])
        
        # Normalize the data
        min_vals = objectives.min(axis=0)
        max_vals = objectives.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        normalized = (objectives - min_vals) / range_vals
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.max_size, random_state=42)
        clusters = kmeans.fit_predict(normalized)
        
        # Keep solutions closest to their cluster centers
        centers = kmeans.cluster_centers_
        
        # Initialize a list to hold the selected solutions
        selected_solutions = []
        
        # For each cluster
        for i in range(self.max_size):
            # Get indices of solutions in this cluster
            cluster_indices = np.where(clusters == i)[0]
            
            if len(cluster_indices) > 0:
                # Calculate distances to cluster center
                cluster_points = normalized[cluster_indices]
                distances = np.sqrt(((cluster_points - centers[i]) ** 2).sum(axis=1))
                
                # Find the solution closest to the center
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_solutions.append(self.solutions[closest_idx])
        
        # Update solutions
        self.solutions = selected_solutions
    
    def get_reference_points(self, num_points=5):
        """
        Generate evenly spaced reference points along the Pareto front.
        
        Args:
            num_points: Number of reference points to generate
            
        Returns:
            List of ParetoSolution objects at reference points
        """
        if len(self.solutions) == 0:
            return []
        
        # Sort solutions by waste efficiency
        sorted_solutions = sorted(
            self.solutions, 
            key=lambda s: s.objectives['waste_efficiency']
        )
        
        # If we have fewer solutions than requested points, return all solutions
        if len(sorted_solutions) <= num_points:
            return sorted_solutions
        
        # Otherwise, select evenly spaced solutions
        indices = np.linspace(0, len(sorted_solutions) - 1, num_points, dtype=int)
        return [sorted_solutions[i] for i in indices]
    
    def plot_front(self, save_path=None):
        """
        Plot the Pareto front.
        
        Args:
            save_path: Optional path to save the plot
        """
        if len(self.solutions) == 0:
            return
        
        # Extract objectives
        waste_eff = [s.objectives['waste_efficiency'] for s in self.solutions]
        overflow_rate = [s.objectives['overflow_rate'] for s in self.solutions]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(waste_eff, overflow_rate, s=50, alpha=0.7)
        plt.xlabel('Waste Collection Efficiency')
        plt.ylabel('Overflow Rate')
        plt.title('Pareto Front of Waste Collection Solutions')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels to some points
        for i, sol in enumerate(self.solutions):
            if i % max(1, len(self.solutions) // 5) == 0:  # Label only some points to avoid crowding
                plt.annotate(f"w_waste={sol.weights['w_waste']:.1f}\nw_over={sol.weights['w_over']:.1f}",
                            (waste_eff[i], overflow_rate[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


class MORLWeightOptimizer(WeightAdjustmentStrategy):
    """
    Multi-Objective RL optimizer that adjusts weights using Pareto front exploration
    and adaptive weight selection based on performance metrics.
    """
    def __init__(self, 
                initial_weights,
                weight_names=['w_waste', 'w_over', 'w_len'],
                objective_names=['waste_efficiency', 'overflow_rate'],
                weight_ranges=[0.01, 5.0],
                history_window=20,
                exploration_factor=0.2,
                adaptation_rate=0.1,
                device='cuda'):
        """
        Args:
            initial_weights: Dictionary of initial weights
            weight_names: Names of the weights to optimize
            objective_names: Names of objectives to track
            weight_ranges (list): Range for weight components [min, max]
            history_window: Window size for tracking performance history
            exploration_factor: How much to explore new weight combinations
            adaptation_rate: How quickly to adapt weights
            device: Computation device
        """
        self.device = device
        self.weight_names = weight_names
        self.objective_names = objective_names
        
        # Initialize current weights
        self.current_weights = {}
        for name in weight_names:
            if name in initial_weights:
                self.current_weights[name] = initial_weights[name]
            else:
                self.current_weights[name] = 1.0  # Default
        
        # Set constraints
        # Set constraints
        if isinstance(weight_ranges[0], (int, float)):
            self.min_weights = {name: float(weight_ranges[0]) for name in weight_names}
        else:
            self.min_weights = weight_ranges[0] or {name: 0.01 for name in weight_names}

        if isinstance(weight_ranges[1], (int, float)):
            self.max_weights = {name: float(weight_ranges[1]) for name in weight_names}
        else:
            self.max_weights = weight_ranges[1] or {name: 5.0 for name in weight_names}
        
        # Performance tracking
        self.history_window = history_window
        self.performance_history = deque(maxlen=history_window)
        self.reward_history = deque(maxlen=history_window)
        
        # Weight adaptation parameters
        self.exploration_factor = exploration_factor
        self.adaptation_rate = adaptation_rate
        
        # Pareto front tracking
        self.pareto_front = ParetoFront()
        
        # Track current state
        self.current_objective_values = {obj: 0.0 for obj in objective_names}
        self.current_reward = 0.0
        self.day = 0
        self.step = 0
        
        # Track direction trends
        self.objective_trends = {obj: 0.0 for obj in objective_names}
    
    def propose_weights(self, context=None):
        """
        Implementation of Strategy interface.
        """
        # Internal update or just current
        return self.get_current_weights()
    
    def feedback(self, reward, metrics, day=None, step=None):
        """
        Implementation of Strategy interface.
        """
        # Metrics handling. MORL expects dictionary with keys like 'waste_collected'.
        # If metrics is not a dictionary with keys, we likely fail here.
        # This implies standard interface usage should stick to dicts if possible.
        self.update_weights(metrics=metrics, reward=reward, day=day, step=step)

    def get_current_weights(self):
        """
        Get current weight values as a dictionary
        
        Returns:
            Dictionary mapping weight names to values
        """
        return self.current_weights
        
    def update_performance_history(self, metrics, reward):
        """
        Update performance history with latest metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            reward: Scalar reward achieved
        """
        # Calculate objective values from metrics
        objective_values = self._calculate_objectives(metrics)
        
        # Update current objective values
        self.current_objective_values = objective_values
        self.current_reward = reward
        
        # Add to history
        self.performance_history.append(objective_values)
        self.reward_history.append(reward)
        
        # Update Pareto front with current solution
        solution = ParetoSolution(
            weights=self.current_weights,
            objectives=objective_values,
            reward=reward,
            model_id=f"day_{self.day}_step_{self.step}"
        )
        self.pareto_front.add_solution(solution)
        
        # Update direction trends if we have enough history
        if len(self.performance_history) > 1:
            for obj in self.objective_names:
                prev = self.performance_history[-2][obj]
                curr = objective_values[obj]
                change = curr - prev
                
                # Exponential moving average of trends
                alpha = 0.3  # Smoothing factor
                self.objective_trends[obj] = alpha * change + (1 - alpha) * self.objective_trends[obj]
    
    def _calculate_objectives(self, metrics):
        """
        Calculate objective values from raw metrics.
        
        Args:
            metrics: Dictionary with raw metrics
            
        Returns:
            Dictionary of objective values
        """
        objectives = {}
        
        # Calculate waste collection efficiency (higher is better)
        # This combines waste collected per distance traveled
        if 'waste_collected' in metrics and 'tour_length' in metrics:
            waste = metrics['waste_collected']
            length = max(0.1, metrics['tour_length'])  # Avoid division by zero
            objectives['waste_efficiency'] = waste / length
        else:
            objectives['waste_efficiency'] = 0.0
            
        # Calculate overflow rate (lower is better)
        if 'num_overflows' in metrics and 'total_bins' in metrics:
            overflows = metrics['num_overflows']
            total = max(1, metrics['total_bins'])  # Avoid division by zero
            objectives['overflow_rate'] = overflows / total
        else:
            objectives['overflow_rate'] = 1.0  # Worst case
            
        return objectives
    
    def _need_to_adjust_weights(self):
        """
        Determine if weights need adjustment based on performance trends.
        
        Returns:
            Boolean indicating if adjustment is needed
        """
        # Not enough history yet
        if len(self.performance_history) < self.history_window // 2:
            return False
            
        # Check for performance degradation
        recent_rewards = list(self.reward_history)[-5:]
        if len(recent_rewards) >= 5:
            is_decreasing = all(recent_rewards[i] <= recent_rewards[i-1] for i in range(1, len(recent_rewards)))
            if is_decreasing:
                return True
                
        # Check for stagnation in objectives
        if len(self.performance_history) >= 5:
            recent_waste = [p['waste_efficiency'] for p in list(self.performance_history)[-5:]]
            recent_overflow = [p['overflow_rate'] for p in list(self.performance_history)[-5:]]
            
            waste_stagnant = max(recent_waste) - min(recent_waste) < 0.01 * max(recent_waste)
            overflow_stagnant = max(recent_overflow) - min(recent_overflow) < 0.01 * max(recent_overflow)
            
            if waste_stagnant and overflow_stagnant:
                return True
        
        # Otherwise, adjust periodically
        return self.step % max(1, int(self.history_window * 0.5)) == 0
    
    def update_weights(self, metrics=None, reward=None, day=None, step=None):
        """
        Update weights based on performance metrics and exploration.
        
        Args:
            metrics: Latest performance metrics (optional)
            reward: Latest reward (optional)
            day: Current day (optional)
            step: Current step (optional)
            
        Returns:
            Dictionary of updated weights
        """
        # Update tracking variables
        if day is not None:
            self.day = day
        if step is not None:
            self.step = step
            
        # Update history if metrics provided
        if metrics is not None and reward is not None:
            self.update_performance_history(metrics, reward)
            
        # Determine if adjustment is needed
        if not self._need_to_adjust_weights():
            return self.current_weights
            
        # Decide adjustment strategy based on state
        if np.random.random() < self.exploration_factor:
            # Exploration: randomly select from Pareto front or explore new weights
            self._explore_new_weights()
        else:
            # Exploitation: adapt weights based on current trends
            self._adapt_weights_by_trends()
            
        # Ensure weights are within bounds
        self._constrain_weights()
            
        return self.current_weights
    
    def _explore_new_weights(self):
        """
        Explore new weights, either from Pareto front or by random perturbation.
        """
        # 50% chance to select from Pareto front if enough solutions exist
        if len(self.pareto_front.solutions) > 3 and np.random.random() < 0.5:
            reference_points = self.pareto_front.get_reference_points(5)
            
            # Select one reference point randomly
            if reference_points:
                selected = np.random.choice(reference_points)
                for name in self.weight_names:
                    if name in selected.weights:
                        self.current_weights[name] = selected.weights[name]
        else:
            # Random exploration around current weights
            for name in self.weight_names:
                # Calculate exploration range
                current = self.current_weights[name]
                min_val = self.min_weights.get(name, current * 0.1)
                max_val = self.max_weights.get(name, current * 10.0)
                
                # Apply random perturbation within range
                range_size = max_val - min_val
                perturbation = (np.random.random() - 0.5) * range_size * self.exploration_factor
                self.current_weights[name] = current + perturbation
    
    def _adapt_weights_by_trends(self):
        """
        Adapt weights based on performance trends to improve specific objectives.
        """
        # Determine which objective needs more focus based on trends
        waste_trend = self.objective_trends.get('waste_efficiency', 0)
        overflow_trend = -self.objective_trends.get('overflow_rate', 0)  # Negate because lower is better
        
        # Check current performance levels
        overflow_rate = self.current_objective_values.get('overflow_rate', 0)
        
        # If overflow rate is high, prioritize reducing overflows
        if overflow_rate > 0.1:  # More than 10% of bins overflowing
            self.current_weights['w_over'] *= (1 + self.adaptation_rate)
            self.current_weights['w_waste'] *= (1 - self.adaptation_rate * 0.5)
        # If waste efficiency trend is negative, increase waste weight
        elif waste_trend < 0:
            self.current_weights['w_waste'] *= (1 + self.adaptation_rate)
            self.current_weights['w_len'] *= (1 - self.adaptation_rate * 0.5)
        # If overflow trend is negative (overflows increasing), increase overflow weight
        elif overflow_trend < 0:
            self.current_weights['w_over'] *= (1 + self.adaptation_rate)
        # Otherwise, balance objectives
        else:
            # Slightly increase waste weight and reduce length weight for efficiency
            self.current_weights['w_waste'] *= (1 + self.adaptation_rate * 0.2)
            self.current_weights['w_len'] *= (1 - self.adaptation_rate * 0.1)
    
    def _constrain_weights(self):
        """
        Ensure weights stay within defined bounds.
        """
        for name in self.weight_names:
            min_val = self.min_weights.get(name, 0.1)
            max_val = self.max_weights.get(name, 100.0)
            self.current_weights[name] = max(min_val, min(max_val, self.current_weights[name]))
    
    def get_weight_history_dataframe(self):
        """
        Get weight and objective history as a pandas DataFrame.
        
        Returns:
            DataFrame with weights and objectives
        """
        # Convert Pareto front solutions to DataFrame rows
        data = []
        for sol in self.pareto_front.solutions:
            row = {}
            row.update(sol.weights)
            row.update({f"obj_{k}": v for k, v in sol.objectives.items()})
            row['reward'] = sol.reward
            row['model_id'] = sol.model_id
            data.append(row)
        return pd.DataFrame(data)
