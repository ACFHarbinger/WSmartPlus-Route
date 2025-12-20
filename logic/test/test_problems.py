
import torch
import pytest
import numpy as np

from unittest.mock import MagicMock, patch
from logic.src.problems.vrpp import problem_vrpp
from logic.src.problems.vrpp.state_vrpp import StateVRPP
from logic.src.problems.vrpp.problem_vrpp import VRPP, VRPPDataset, generate_instance as gen_vrpp, CVRPP
from logic.src.problems.vrpp.state_cvrpp import StateCVRPP
from logic.src.problems.wcvrp import problem_wcvrp
from logic.src.problems.wcvrp.state_wcvrp import StateWCVRP
from logic.src.problems.wcvrp.problem_wcvrp import WCVRP, WCVRPDataset, generate_instance as gen_wcvrp


class TestVRPP:
    """Tests for the VRPP problem class."""

    @pytest.mark.unit
    def test_get_costs_basic(self):
        """Test cost calculation for a simple VRPP route."""
        # Inject globals
        problem_vrpp.COST_KM = 1.0
        problem_vrpp.REVENUE_KG = 0.1
        problem_vrpp.BIN_CAPACITY = 100.0

        dataset = {
            'depot': torch.tensor([[0.0, 0.0]]), # (1, 2) OK
            'loc': torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]), # (1, 2, 2)
            'waste': torch.tensor([[10.0, 20.0]]), # (1, 2) OK
            'max_waste': torch.tensor([100.0]) # (1) OK
        }
        pi = torch.tensor([[1, 2]]) # (1, 2)
        
        neg_profit, c_dict, _ = VRPP.get_costs(dataset, pi, cw_dict=None)
        
        # Length calculation:
        # 0->1: sqrt(1^2) = 1.0
        # 1->2: sqrt(1^2) = 1.0
        # 2->0: sqrt(2^2) = 2.0
        # Total Length = 4.0
        assert torch.isclose(c_dict['length'], torch.tensor([4.0]))
        
        # Waste: 10 + 20 = 30.0
        assert torch.isclose(c_dict['waste'], torch.tensor([30.0]))
        
        # Profit (COST_KM=1, REVENUE=0.1)
        # negative_profit = Length * 1 - Waste * 0.1 = 4.0 - 3.0 = 1.0
        assert torch.isclose(neg_profit, torch.tensor([1.0]))

    @pytest.mark.unit
    def test_make_state_defaults(self):
        """Test proper initialization of defaults in make_state."""
        problem_vrpp.COST_KM = 1.0
        problem_vrpp.REVENUE_KG = 0.1
        problem_vrpp.BIN_CAPACITY = 100.0

        with patch('logic.src.problems.vrpp.problem_vrpp.StateVRPP.initialize') as mock_init:
            VRPP.make_state(input='mock_input')
            
            args, kwargs = mock_init.call_args
            assert 'profit_vars' in kwargs
            assert kwargs['profit_vars']['cost_km'] is not None

    @pytest.mark.unit
    def test_dataset_generation(self, mocker):
        """Test dataset generation calls."""
        mocker.patch('logic.src.problems.vrpp.problem_vrpp.load_area_and_waste_type_params',
                     return_value=(100, 0.1, 10, 1.0, 1.0))
        mocker.patch('logic.src.problems.vrpp.problem_vrpp.generate_waste_prize', 
                     return_value=np.zeros((1, 10)))

        dataset = VRPPDataset(size=10, num_samples=2, distribution='unif')
        assert len(dataset) == 2
        assert dataset[0]['waste'].shape == (1, 10)

class TestStateVRPP:
    """Tests for StateVRPP logic."""

    @pytest.mark.unit
    def test_initialize(self):
        """Test initialization of StateVRPP."""
        # Input mock
        batch_size = 2
        n_loc = 3
        input_data = {
            'depot': torch.zeros(batch_size, 2),
            'loc': torch.rand(batch_size, n_loc, 2),
            'waste': torch.rand(batch_size, n_loc),
            'max_waste': torch.ones(batch_size) * 10.0
        }
        
        state = StateVRPP.initialize(input_data, edges=None)
        
        assert state.ids.shape == (batch_size, 1)
        assert state.visited_.shape == (batch_size, 1, n_loc + 1)
        # waste should be padded with 0 (depot) -> n_loc + 1
        assert state.waste.shape == (batch_size, n_loc + 1)
        assert state.waste[:, 0].eq(0).all()
        assert state.cur_total_waste.shape == (batch_size, 1)

    @pytest.mark.unit
    def test_update_and_mask(self):
        """Test state update and mask generation."""
        batch_size = 1
        n_loc = 2
        input_data = {
            'depot': torch.zeros(batch_size, 2),
            'loc': torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]), # (1, 2, 2)
            'waste': torch.tensor([[5.0, 6.0]]),
            'max_waste': torch.tensor([10.0])
        }
        state = StateVRPP.initialize(input_data, edges=None)
        
        # Test mask at start (depot is 0, others unvisited)
        # 0=feasible, 1=infeasible
        # Start: i=0. Mask forbids visiting depot if not all visited? 
        # get_mask: mask = visited_ | visited_[:, :, 0:1]
        # visited_ is all 0. mask is 0.
        # mask[:, :, 0] = 0 (depot always feasible according to code comment)
        mask = state.get_mask()
        assert mask[:, :, 0].eq(0).all() # Depot allowed
        
        # Action: Go to node 1 (index 1)
        selected = torch.tensor([1]) # (batch_size,)
        state = state.update(selected)
        
        # Check visited
        # visited_ has shape (batch, 1, n_loc+1)
        assert state.visited_[:, 0, 1].item() == 1
        
        # Check collected waste
        # Node 1 waste = 5.0. total = 5.0
        assert state.cur_total_waste.item() == 5.0
        
        # Check lengths
        # 0->1 dist is 1.0.
        assert state.lengths.item() == 1.0
        
        # Check mask again
        mask = state.get_mask()
        assert mask[:, :, 1].item() == 1 # Node 1 visited, now infeasible
        assert mask[:, :, 2].item() == 0 # Node 2 unvisited

class TestWCVRP:
    """Tests for the WCVRP problem class."""

    @pytest.mark.unit
    def test_get_costs_overflow(self):
        """Test cost calculation with overflows for WCVRP."""
        problem_wcvrp.COST_KM = 1.0
        problem_wcvrp.REVENUE_KG = 0.1
        
        dataset = {
            'depot': torch.tensor([[0.0, 0.0]]),
            'loc': torch.tensor([[[1.0, 0.0]]]), # (1, 1, 2)
            'waste': torch.tensor([[120.0]]), # (1, 1)
            'max_waste': torch.tensor([100.0]) # (1)
        }
        
        pi = torch.tensor([[1, 0]]) # (1, 2) - Visit 1 then padded 0
        
        cost, c_dict, _ = WCVRP.get_costs(dataset, pi, cw_dict=None)
        
        # Overflows: 120 >= 100 but visited -> 0 remaining overflows
        assert c_dict['overflows'] == 0.0
        
        # Length: 0->1 (1.0) + 1->0 (1.0) = 2.0
        assert c_dict['length'] == 2.0
        
        # Cost check: overflows + length - waste ??
        # Or cost logic in WCVRP?
        # cost = overflows + length - waste
        # 1.0 + 2.0 - 100.0 (waste clamped to max) = -97.0
        
        # Wait, waste calculation in WCVRP:
        # w = waste.gather... clamp(max)
        # So collected waste is 100.
        
        assert torch.isclose(cost, torch.tensor([-98.0]))

    @pytest.mark.unit
    def test_make_state_removes_profit_vars(self):
        """Test that profit_vars is removed from kwargs for WCVRP."""
        with patch('logic.src.problems.wcvrp.problem_wcvrp.StateWCVRP.initialize') as mock_init:
            kwargs = {'profit_vars': {'a': 1}, 'other': 2}
            WCVRP.make_state(input='mock', **kwargs)
            
            args, called_kwargs = mock_init.call_args
            assert 'profit_vars' not in called_kwargs
            assert called_kwargs['other'] == 2

    @pytest.mark.unit
    def test_dataset_generation(self, mocker):
        """Test dataset generation calls for WCVRP."""
        # Mock dependencies
        mocker.patch('logic.src.problems.wcvrp.problem_wcvrp.generate_waste_prize', 
                     return_value=np.zeros((1, 10)))

        dataset = WCVRPDataset(size=10, num_samples=2, distribution='unif')
        assert len(dataset) == 2
        assert dataset[0]['waste'].shape == (1, 10)

class TestStateWCVRP:
    """Tests for StateWCVRP logic."""

    @pytest.mark.unit
    def test_initialize(self):
        """Test initialization of StateWCVRP."""
        # Input mock
        batch_size = 1
        n_loc = 2
        input_data = {
            'depot': torch.zeros(batch_size, 2),
            'loc': torch.rand(batch_size, n_loc, 2),
            'waste': torch.tensor([[12.0, 5.0]]), # Node 1 overflows (12 > 10)
            'max_waste': torch.tensor([10.0])
        }
        
        state = StateWCVRP.initialize(input_data, edges=None)
        
        # Check initial overflows
        # Node 1 has 12, max 10 -> Overflow. Node 2 has 5 -> No overflow.
        # cur_overflows is a count of overflowing bins? 
        # Code: cur_overflows=torch.sum((input['waste'] >= max_waste[:, None]), dim=-1)
        # So it counts how many bins are overflowing initially? Yes.
        assert state.cur_overflows.item() == 1.0

    @pytest.mark.unit
    def test_update_waste_clamping(self):
        """Test that collected waste is clamped to max_waste."""
        batch_size = 1
        n_loc = 1
        input_data = {
            'depot': torch.zeros(batch_size, 2),
            'loc': torch.tensor([[[1.0, 0.0]]]),
            'waste': torch.tensor([[15.0]]), # 15 > 10
            'max_waste': torch.tensor([10.0])
        }
        state = StateWCVRP.initialize(input_data, edges=None)
        
        # Initial checks
        assert state.cur_overflows.item() == 1.0
        
        # Update: Visit node 1
        selected = torch.tensor([1])
        state = state.update(selected)
        
        assert state.cur_total_waste.item() == 10.0

class TestStateCVRPP:
    """Tests for StateCVRPP logic."""

    @pytest.mark.unit
    def test_capacity_mask(self):
        """Test that nodes exceeding remaining capacity are masked."""
        batch_size = 1
        n_loc = 2
        # Setup: Bin Capacity 10.
        # Node 1: Waste 6.
        # Node 2: Waste 6.
        # We can visit 1 OR 2, but not both (6+6 > 10).
        input_data = {
            'depot': torch.zeros(batch_size, 2),
            'loc': torch.rand(batch_size, n_loc, 2),
            'waste': torch.tensor([[6.0, 6.0]]),
            'max_waste': torch.ones(batch_size) * 10.0 # Clamp limit
        }
        
        # Manually invoke initialize with explicit profit_vars to set capacity
        profit_vars = {'cost_km': 1.0, 'revenue_kg': 1.0, 'bin_capacity': 10.0, 'vehicle_capacity': 10.0}
        state = StateCVRPP.initialize(input_data, edges=None, profit_vars=profit_vars)
        
        # Visit Node 1
        selected = torch.tensor([1])
        state = state.update(selected)
        
        # Check used_capacity
        assert state.used_capacity.item() == 6.0
        
        # Check mask
        mask = state.get_mask()
        # Node 2 (idx 2) should be masked because 6 (curr) + 6 (potential) = 12 > 10
        assert mask[:, :, 2].item() == 1
        # Depot (idx 0) should be unmasked (allowed to return)
        assert mask[:, :, 0].item() == 0

    @pytest.mark.unit
    def test_depot_reset(self):
        """Test that visiting depot resets used_capacity."""
        batch_size = 1
        n_loc = 2
        input_data = {
            'depot': torch.zeros(batch_size, 2),
            'loc': torch.rand(batch_size, n_loc, 2),
            'waste': torch.tensor([[6.0, 6.0]]),
            'max_waste': torch.ones(batch_size, 1) * 10.0
        }
        profit_vars = {'cost_km': 1.0, 'revenue_kg': 1.0, 'bin_capacity': 10.0, 'vehicle_capacity': 10.0}
        state = StateCVRPP.initialize(input_data, edges=None, profit_vars=profit_vars)
        
        # 0 -> 1 (Load 6)
        state = state.update(torch.tensor([1]))
        assert state.used_capacity.item() == 6.0
        
        # 1 -> 0 (Depot) -> Should reset capacity
        state = state.update(torch.tensor([0]))
        assert state.used_capacity.item() == 0.0
        
        # 0 -> 2 (Load 6) -> Should be allowed now
        state = state.update(torch.tensor([2]))
        assert state.used_capacity.item() == 6.0

class TestCVRPP:
    """Tests for the CVRPP problem class."""

    @pytest.mark.unit
    def test_make_state_uses_cvrpp_state(self):
        """Test that CVRPP uses StateCVRPP."""
        problem_vrpp.COST_KM = 1.0
        problem_vrpp.REVENUE_KG = 0.1
        problem_vrpp.BIN_CAPACITY = 100.0
        
        with patch('logic.src.problems.vrpp.state_cvrpp.StateCVRPP.initialize') as mock_init:
            CVRPP.make_state(input='mock')
            mock_init.assert_called_once()
            
    @pytest.mark.unit
    def test_get_costs_multitrip(self):
        """Test CVRPP.get_costs with multi-trip capacity logic."""
        # Setup: Vehicle Capacity 100.
        problem_vrpp.BIN_CAPACITY = 100.0
        problem_vrpp.VEHICLE_CAPACITY = 100.0
        problem_vrpp.COST_KM = 1.0
        problem_vrpp.REVENUE_KG = 1.0
        
        batch_size = 1
        n_loc = 2
        dataset = {
            'depot': torch.zeros(batch_size, 2),
            'loc': torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]), # 1 and 2 away
            'waste': torch.tensor([[60.0, 60.0]]), # Each 60. Total 120 > 100.
            'max_waste': torch.tensor([100.0])
        }
        
        # Scenario 1: Visit 1 -> 2 directly (Total 120) -> Should Fail Assertion
        pi_fail = torch.tensor([[1, 2]])
        with pytest.raises(AssertionError, match="Used more than capacity"):
            CVRPP.get_costs(dataset, pi_fail, cw_dict=None)
            
        # Scenario 2: Visit 1 -> 0 (Depot) -> 2 (Total 60 reset 60) -> Should Pass
        pi_pass = torch.tensor([[1, 0, 2]])
        cost, c_dict, _ = CVRPP.get_costs(dataset, pi_pass, cw_dict=None)
        
        # Check Profit
        # Length: 
        # 0->1 (1.0)
        # 1->0 (1.0)
        # 0->2 (2.0)
        # 2->0 (2.0)
        # Total Length = 6.0
        # collected waste is waste at 1 and 2. Depot implies reset capacity not "collection of negative waste".
        # get_costs logic sums gathered waste. gather([1, 0, 2]) -> [60, 0, 60]. Sum = 120.
        # Profit = 120 - 6 = 114. Negative Profit = -114.
        assert torch.isclose(cost, torch.tensor([-114.0]))