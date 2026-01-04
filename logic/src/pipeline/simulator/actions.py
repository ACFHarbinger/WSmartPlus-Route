from typing import Dict, Any
from abc import ABC, abstractmethod
from logic.src.utils.log_utils import send_daily_output_to_gui
from logic.src.policies.adapters import PolicyFactory
from logic.src.pipeline.simulator.day import get_daily_results


class SimulationAction(ABC):
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        """
        Executes the action and updates the context.
        """
        pass

class FillAction(SimulationAction):
    def execute(self, context: Dict[str, Any]) -> None:
        bins = context['bins']
        day = context['day']
        
        if bins.is_stochastic():
            new_overflows, fill, total_fill, sum_lost = bins.stochasticFilling()
        else:
            new_overflows, fill, total_fill, sum_lost = bins.loadFilling(day)
            
        context['new_overflows'] = new_overflows
        context['fill'] = fill
        context['total_fill'] = total_fill
        context['sum_lost'] = sum_lost
        
        # Accumulate overflows in context (if needed by subsequent steps or final return)
        if 'overflows' in context:
            context['overflows'] += new_overflows

class PolicyExecutionAction(SimulationAction):
    def execute(self, context: Dict[str, Any]) -> None:
        policy_name = context['policy_name']
        adapter = PolicyFactory.get_adapter(policy_name)
        
        # NeuralPolicyAdapter expects 'fill' in context, which FillAction just put there.
        
        # Extract config
        config = context.get('config', {})
        
        tour, cost, extra_output = adapter.execute(**context)
        
        context['tour'] = tour
        context['cost'] = cost
        context['extra_output'] = extra_output
        
        # Handle specific extra outputs updates
        if 'policy_regular' in policy_name:
            context['cached'] = extra_output
        elif policy_name[:2] == 'am' or policy_name[:4] == 'ddam' or "transgcn" in policy_name:
            context['output_dict'] = extra_output

class CollectAction(SimulationAction):
    def execute(self, context: Dict[str, Any]) -> None:
        bins = context['bins']
        tour = context['tour']
        cost = context['cost']
        
        collected, total_collected, ncol, profit = bins.collect(tour, cost)
        
        context['collected'] = collected
        context['total_collected'] = total_collected
        context['ncol'] = ncol
        context['profit'] = profit

class LogAction(SimulationAction):
    def execute(self, context: Dict[str, Any]) -> None:              
        tour = context['tour']
        cost = context['cost']
        total_collected = context['total_collected']
        ncol = context['ncol']
        profit = context['profit']
        new_overflows = context['new_overflows']
        coords = context['coords']
        day = context['day']
        sum_lost = context['sum_lost']
        
        dlog = get_daily_results(total_collected, ncol, cost, tour, day, new_overflows, sum_lost, coords, profit)
            
        context['daily_log'] = dlog
        
        send_daily_output_to_gui(
            dlog, context['policy_name'], context['sample_id'], context['day'], 
            context['total_fill'], context['collected'], context['bins'].c, 
            context['realtime_log_path'], tour, coords, context['lock']
        )
