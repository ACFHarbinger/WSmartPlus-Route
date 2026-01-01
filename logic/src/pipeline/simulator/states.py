import os
import time
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from logic.src.utils.definitions import ROOT_DIR, TQDM_COLOURS, SIM_METRICS, DAY_METRICS
from logic.src.utils.log_utils import log_to_json
from logic.src.utils.setup_utils import setup_model, setup_env, setup_hrl_manager

from .bins import Bins
from .checkpoints import checkpoint_manager, SimulationCheckpoint, CheckpointError
from .day import run_day
from .processor import process_data, process_model_data, setup_basedata, setup_dist_path_tup, save_matrix_to_excel


class SimulationContext:
    """
    Context for the Simulation State Machine.
    Holds shared state and manages transitions.
    """
    def __init__(self, opts, device, indices, sample_id, pol_id, model_weights_path, variables_dict):
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
        
        # Progress Bar items
        self.tqdm_pos = variables_dict.get('tqdm_pos', 0)
        self.colour = TQDM_COLOURS[pol_id % len(TQDM_COLOURS)]
        
        self.transition_to(InitializingState())

    def transition_to(self, state):
        self.current_state = state
        if self.current_state is not None:
            self.current_state.context = self
        
    def run(self):
        while self.current_state is not None:
            self.current_state.handle()
        return self.result

    def get_current_state_tuple(self):
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
        pass


class InitializingState(SimState):
    def handle(self):
        opts = self.context.opts
        ctx = self.context
        
        # Ensure results directory exists
        if not os.path.exists(ctx.results_dir):
            os.makedirs(ctx.results_dir, exist_ok=True)
            
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
                ctx.bins = Bins(opts['size'], ctx.data_dir, ctx.data_dist[:-1], area=opts['area'], waste_type=opts['waste_type'], waste_file=opts['waste_filepath'])
                gamma_option = int(ctx.policy[-1]) - 1
                ctx.bins.setGammaDistribution(option=gamma_option)
            else:
                ctx.bins = Bins(opts['size'], ctx.data_dir, ctx.data_dist, area=opts['area'], waste_type=opts['waste_type'], waste_file=opts['waste_filepath'])

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
                    
                    data_ls, output_ls, ctx.cached = run_day(
                        opts['size'], ctx.policy, ctx.bins, ctx.new_data, ctx.coords, opts['run_tsp'], ctx.sample_id,
                        ctx.overflows, day, ctx.model_env, ctx.model_tup, opts['n_vehicles'], opts['area'], realtime_log_path, 
                        opts['waste_type'], ctx.dist_tup, ctx.current_collection_day, ctx.cached, ctx.device, 
                        ctx.lock, ctx.hrl_manager, opts['gate_prob_threshold'], opts['mask_prob_threshold'], opts['two_opt_max_iter']
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
