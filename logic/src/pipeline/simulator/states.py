"""
State Pattern Implementation for Simulation Lifecycle Management.

This module implements the State Pattern to manage the three phases of
simulation execution: Initialization, Running, and Finishing. Each phase
is encapsulated in a separate state class with well-defined transitions.

Architecture:
    - SimulationContext: Maintains simulation state and manages transitions
    - SimState: Abstract base class for states
    - InitializingState: Setup (data loading, model loading, checkpoints)
    - RunningState: Day-by-day execution loop
    - FinishingState: Result aggregation and persistence

State Transitions:
    InitializingState → RunningState → FinishingState → None (end)

The State Pattern provides:
    - Clear separation of lifecycle phases
    - Explicit state transitions
    - Centralized state management
    - Resumption from checkpoints (re-enter RunningState)

Classes:
    SimulationContext: Context object holding simulation state
    SimState: Abstract state interface
    InitializingState: Initialization phase
    RunningState: Execution phase
    FinishingState: Finalization phase
"""

import os
import time
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from logic.src.utils.definitions import ROOT_DIR, TQDM_COLOURS, SIM_METRICS, DAY_METRICS
from logic.src.utils.log_utils import log_to_json
from logic.src.utils.setup_utils import setup_model, setup_env, setup_hrl_manager
from logic.src.utils.config_loader import load_config

from .bins import Bins
from .checkpoints import checkpoint_manager, SimulationCheckpoint, CheckpointError
from .day import run_day
from .processor import process_data, process_model_data, setup_basedata, setup_dist_path_tup, save_matrix_to_excel


class SimulationContext:
    """
    Context object for the Simulation State Machine.

    Manages the full lifecycle of a single simulation run, coordinating
    transitions between initialization, execution, and finalization states.

    The context holds all simulation state:
        - Configuration (opts, device, policy)
        - Data (bins, coordinates, distance matrices)
        - Models (neural networks, solvers)
        - Runtime (checkpoints, progress bars, locks)
        - Results (logs, execution times)

    State transitions are triggered by calling transition_to(new_state),
    which updates current_state and delegates control to the new state's
    handle() method.

    Attributes:
        opts: Simulation configuration dictionary
        device: torch.device for neural models
        policy: Policy identifier string
        bins: Bins state manager
        model_env: Loaded neural model or solver
        checkpoint: SimulationCheckpoint for persistence
        current_state: Active state object
        result: Final result dictionary (populated on completion)
    """

    def __init__(self, opts, device, indices, sample_id, pol_id, model_weights_path, variables_dict):
        """
        Initializes the simulation context with configuration and shared variables.

        Args:
            opts: Dictionary of simulation options.
            device: torch.device for tensor computations.
            indices: List of bin indices for the current sample.
            sample_id: Identifier for the current data sample.
            pol_id: Index of the current policy in opts['policies'].
            model_weights_path: Path to neural model weights.
            variables_dict: Shared variables (locks, counters, progress bars).
        """
        self.opts = opts
        self.device = device
        self.indices = indices
        self.sample_id = sample_id
        self.pol_id = pol_id
        self.model_weights_path = model_weights_path
        
        # Shared Variables (Global Lock/Counter passed from outside)
        self.lock = variables_dict.get('lock')
        self.counter = variables_dict.get('counter')
        self.overall_progress = variables_dict.get('overall_progress')

        # State Variables
        self.current_state = None
        self.result = None
        
        # Simulation Data
        self.data_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator")
        self.results_dir = os.path.join(ROOT_DIR, "assets", opts['output_dir'], str(opts['days']) + "_days", str(opts['area']) + '_' + str(opts['size']))
        self.policy = opts['policies'][pol_id]
        self.pol_strip, self.data_dist = self.policy.rsplit("_", 1)
        
        # Runtime Data
        self.start_day = 1
        self.checkpoint = None
        self.bins = None
        self.new_data = None
        self.coords = None
        self.dist_tup = None
        self.model_tup = None
        self.model_env = None
        self.hrl_manager = None
        self.cached = None
        self.run_time = 0
        self.overflows = 0
        self.current_collection_day = 0
        self.daily_log = None
        self.attention_dict = {}
        self.execution_time = 0
        self.tic = 0
        self.config = None
        
        # Progress Bar items
        self.tqdm_pos = variables_dict.get('tqdm_pos', 0)
        self.colour = TQDM_COLOURS[pol_id % len(TQDM_COLOURS)]
        
        self.transition_to(InitializingState())

    def transition_to(self, state):
        """
        Transitions the context to a new state.

        Args:
            state: The new SimState object to transition to.
        """
        self.current_state = state
        if self.current_state is not None:
            self.current_state.context = self
        
    def run(self):
        """
        Main execution loop that runs the state machine until completion.

        Returns:
            The simulation result dictionary.
        """
        while self.current_state is not None:
            self.current_state.handle()
        return self.result

    def get_current_state_tuple(self):
        """
        Retrieves a tuple of current simulation variables for checkpointing.

        Returns:
            A tuple containing simulation state data.
        """
        # Helper for checkpoints
        return (
            self.new_data, self.coords, self.dist_tup, None, # adj_matrix not strictly stored in context unless added
            self.bins, self.model_tup, self.cached, self.overflows, 
            self.current_collection_day, self.daily_log, self.execution_time
        )


class SimState(ABC):
    context: SimulationContext

    @abstractmethod
    def handle(self):
        """
        Executes the logic associated with the current simulation state.
        This method is responsible for moving the context to the next state.
        """
        pass


class InitializingState(SimState):
    def handle(self):
        """
        Handles the initialization phase of the simulation.
        
        Loads data, setups models, and initializes indices and distributions.
        Transitions to RunningState upon completion.
        """
        opts = self.context.opts
        ctx = self.context
        
        # Ensure results directory exists
        if not os.path.exists(ctx.results_dir):
            os.makedirs(ctx.results_dir, exist_ok=True)

        # Load Configuration
        # config_path is now expected to be a dict {key: path} or None
        ctx.config = {}
        config_paths = opts.get('config_path')
        
        if config_paths:
            if isinstance(config_paths, dict):
                for key, path in config_paths.items():
                    try:
                        loaded = load_config(path)
                        # Store in ctx.config under the key, OR merge if we want a unified config?
                        # Strategy: Store all under 'configs' and also try to resolve 'active' config?
                        # For simplicity, let's merge them into ctx.config? 
                        # But conflicts might arise.
                        # Better strategy: ctx.config will be the dictionary of {key: loaded_content}
                        # And run_day will pick the right one?
                        # Or checking strict match?
                        
                        # Let's merge everything into ctx.config so lookup is easy (e.g. ctx.config['lookahead']['hgs'])
                        # But wait, if we load two lookahead files, they might clash.
                        # If key is unique (e.g. 'hgs', 'sans'), we can store them as ctx.config[key] = loaded
                        
                        # If loaded has a root key (e.g. 'lookahead'), we might want to merge deep.
                        # For now, let's just store the full loaded dict under the cli key if the cli key is explicit.
                        # User said "similar to model_path" where keys are policies.
                        # So ctx.config[key] = load_config(path) is reasonable.
                        
                        ctx.config[key] = loaded
                        print(f"Loaded configuration for '{key}' from {path}")
                    except Exception as e:
                        print(f"Warning: Failed to load config file {path}: {e}")
            else:
                # Legacy fall back if it's a string (though arg_parser should prevent this)
                try:
                    ctx.config = load_config(config_paths) # Assuming single path
                    print(f"Loaded configuration from {config_paths}")
                except Exception:
                    pass
            
        # Setup Data
        data, bins_coordinates, depot = setup_basedata(opts['size'], ctx.data_dir, opts['area'], opts['waste_type'])
        ctx.checkpoint = SimulationCheckpoint(ctx.results_dir, opts['checkpoint_dir'], ctx.policy, ctx.sample_id)

        # Resume Logic
        saved_state, last_day = (None, 0)
        if opts['resume']:
            saved_state, last_day = ctx.checkpoint.load_state()
            
            # If we resumed and have an overall progress bar, we need to account for skipped days
            # logic/src/pipeline/simulator/simulation.py:289 -> overall_progress.update(completed_days)
            if saved_state is not None and ctx.overall_progress:
                ctx.overall_progress.update(last_day)

        # Model Setup
        configs = None
        if 'am' in ctx.pol_strip or "transgcn" in ctx.pol_strip:
            ctx.model_env, configs = setup_model(ctx.policy, ctx.model_weights_path, opts['model_path'], ctx.device, ctx.lock, opts['temperature'], opts['decode_type'])
            ctx.hrl_manager = setup_hrl_manager(opts, ctx.device, configs, policy=ctx.policy, base_path=ctx.model_weights_path, worker_model=ctx.model_env)
        elif "vrpp" in ctx.pol_strip:
            ctx.model_env = setup_env(ctx.policy, opts['server_run'], opts['gplic_file'], opts['symkey_name'], opts['env_file'])
            ctx.model_tup = (None, None)
        else:
            ctx.model_tup = (None, None)

        # Restore or Init State
        if opts['resume'] and saved_state is not None:
            (ctx.new_data, ctx.coords, ctx.dist_tup, adj_matrix, 
             ctx.bins, ctx.model_tup, ctx.cached, ctx.overflows, 
             ctx.current_collection_day, ctx.daily_log, ctx.run_time) = saved_state
            ctx.start_day = last_day + 1
        else:
            ctx.new_data, ctx.coords = process_data(data, bins_coordinates, depot, ctx.indices)
            
            # Recalculate Distances
            ctx.dist_tup, adj_matrix = setup_dist_path_tup(
                ctx.coords, opts['size'], opts['distance_method'], opts['dm_filepath'], 
                opts['env_file'], opts['gapik_file'], opts['symkey_name'], ctx.device, 
                opts['edge_threshold'], opts['edge_method'], ctx.indices
            )

            if 'am' in ctx.pol_strip or "transgcn" in ctx.pol_strip:
                ctx.model_tup = process_model_data(ctx.coords, ctx.dist_tup[2], ctx.device, opts['vertex_method'], 
                                            configs, opts['edge_threshold'], opts['edge_method'], 
                                            opts['area'], opts['waste_type'], adj_matrix)
            
            # Setup Bins
            if "gamma" in ctx.data_dist:
                ctx.bins = Bins(opts['size'], ctx.data_dir, ctx.data_dist[:-1], area=opts['area'], waste_type=opts['waste_type'], 
                                waste_file=opts['waste_filepath'], noise_mean=opts.get('noise_mean', 0.0), noise_variance=opts.get('noise_variance', 0.0))
                gamma_option = int(ctx.policy[-1]) - 1
                ctx.bins.setGammaDistribution(option=gamma_option)
            else:
                ctx.bins = Bins(opts['size'], ctx.data_dir, ctx.data_dist, area=opts['area'], waste_type=opts['waste_type'], 
                                waste_file=opts['waste_filepath'], noise_mean=opts.get('noise_mean', 0.0), noise_variance=opts.get('noise_variance', 0.0))

            ctx.cached = [] if opts['cache_regular'] else None
            if opts['stats_filepath'] is not None:
                ctx.bins.set_statistics(opts['stats_filepath'])
            if opts['waste_filepath'] is not None:
                ctx.bins.set_sample_waste(ctx.sample_id)
            
            ctx.bins.set_indices(ctx.indices)
            ctx.daily_log = {key: [] for key in DAY_METRICS}

        self.context.transition_to(RunningState())


class RunningState(SimState):
    def handle(self):
        """
        Handles the day-by-day simulation execution.
        
        Runs the daily simulation loop, manages checkpoints, and updates progress.
        Transitions to FinishingState after all days are processed.
        """
        ctx = self.context
        opts = ctx.opts
        
        desc = f"{ctx.policy} #{ctx.sample_id}"
        realtime_log_path = os.path.join(ctx.results_dir, f"log_realtime_{opts['data_distribution']}_{opts['n_samples']}N.jsonl")
        
        ctx.tic = time.process_time() + ctx.run_time
        
        try:
            with checkpoint_manager(ctx.checkpoint, opts['checkpoint_days'], ctx.get_current_state_tuple) as hook:
                hook.set_timer(ctx.tic)
                
                iterator = range(ctx.start_day, opts['days']+1)
                if not opts['no_progress_bar']:
                    iterator = tqdm(iterator, desc=desc, position=ctx.tqdm_pos+1, 
                                    dynamic_ncols=True, leave=False, colour=ctx.colour)
                
                for day in iterator:
                    hook.before_day(day)
                    
                    # Select specific config for this policy if available
                    # Strategy: Check if any key in ctx.config is a substring of ctx.policy
                    current_policy_config = {}
                    
                    # 1. Base config (if any unkeyed config exists or common sections)
                    # For now ctx.config is {key: dict}.
                    
                    # 2. Merge matching configs
                    # If I have keys 'lookahead' and 'hgs' and policy is 'policy_look_ahead_hgs', merge both.
                    for key, cfg in ctx.config.items():
                         if key in ctx.policy:
                             # Merge cfg into current_policy_config
                             # Simple update for now (shallow merge of top keys)
                             # Deep merge would be better but keeping it simple.
                             current_policy_config.update(cfg)
                             
                    data_ls, output_ls, ctx.cached = run_day(
                        opts['size'], ctx.policy, ctx.bins, ctx.new_data, ctx.coords, opts['run_tsp'], ctx.sample_id,
                        ctx.overflows, day, ctx.model_env, ctx.model_tup, opts['n_vehicles'], opts['area'], realtime_log_path, 
                        opts['waste_type'], ctx.dist_tup, ctx.current_collection_day, ctx.cached, ctx.device, 
                        ctx.lock, ctx.hrl_manager, opts['gate_prob_threshold'], opts['mask_prob_threshold'], opts['two_opt_max_iter'],
                        config=current_policy_config
                    )
                    
                    ctx.execution_time = time.process_time() - ctx.tic
                    ctx.new_data, ctx.coords, ctx.bins = data_ls
                    ctx.overflows, dlog, output_dict = output_ls
                    
                    if ctx.counter:
                         with ctx.counter.get_lock(): ctx.counter.value += 1
                    
                    if 'am' in ctx.pol_strip or "transgcn" in ctx.pol_strip:
                        if ctx.pol_strip not in ctx.attention_dict:
                            ctx.attention_dict[ctx.pol_strip] = []
                        ctx.attention_dict[ctx.pol_strip].append(output_dict)
                        
                    for key, val in dlog.items():
                        ctx.daily_log[key].append(val)
                        
                    hook.after_day(ctx.execution_time)
                    
                    if ctx.overall_progress:
                        ctx.overall_progress.update(1)

            self.context.transition_to(FinishingState())
            
        except CheckpointError as e:
            ctx.result = e.error_result
            self.context.transition_to(None) # End
        except Exception as e:
            raise e

class FinishingState(SimState):
    def handle(self):
        """
        Handles the finalization phase of the simulation.
        
        Aggregates results, saves logs to disk, and performs cleanup.
        Sets the final result in the context and ends the state machine.
        """
        ctx = self.context
        opts = ctx.opts
        
        ctx.execution_time = time.process_time() - ctx.tic
        
        lg = [
            np.sum(ctx.bins.inoverflow), np.sum(ctx.bins.collected), np.sum(ctx.bins.ncollections), 
            np.sum(ctx.bins.lost), ctx.bins.travel, np.nan_to_num(np.sum(ctx.bins.collected)/ctx.bins.travel, 0), 
            np.sum(ctx.bins.inoverflow)-np.sum(ctx.bins.collected)+ctx.bins.travel, ctx.bins.profit, ctx.bins.ndays, ctx.execution_time
        ]
        
        daily_log_path = os.path.join(ctx.results_dir, f"daily_{opts['data_distribution']}_{opts['n_samples']}N.json")
        
        if opts['n_samples'] > 1:
            log_path = os.path.join(ctx.results_dir, f"log_full_{opts['n_samples']}N.json")
            log_to_json(log_path, SIM_METRICS, {ctx.policy: lg}, sample_id=ctx.sample_id, lock=ctx.lock)
            log_to_json(daily_log_path, DAY_METRICS, {f"{ctx.pol_strip} #{ctx.sample_id}": ctx.daily_log.values()}, lock=ctx.lock)
        else:
            log_path = os.path.join(ctx.results_dir, f"log_mean_{opts['n_samples']}N.json")
            log_to_json(log_path, SIM_METRICS, {ctx.policy: lg}, lock=ctx.lock)
            log_to_json(daily_log_path, DAY_METRICS, {ctx.pol_strip: ctx.daily_log.values()}, lock=ctx.lock)
        
        # Save fill history
        save_matrix_to_excel(ctx.bins.get_fill_history(), ctx.results_dir, opts['seed'], 
                             opts['data_distribution'], ctx.policy, ctx.sample_id)
                                  
        if ctx.checkpoint:
             ctx.checkpoint.clear()
        
        ctx.result = {ctx.policy: lg, 'success': True}
        self.context.transition_to(None) # End
