import torch
import pytest
from unittest.mock import MagicMock
from logic.src.pipeline.reinforcement_learning.core.sapo import SAPOTrainer


class TestSAPO:
    """Tests for SAPO Trainer implementation and loss logic."""

    @pytest.fixture
    def mock_deps(self):
        # Mock dependencies
        model = MagicMock()
        # Model returns: cost, log_probs, cost_dict, pi, entropy
        # Shapes: cost (B), log_probs (B), cost_dict (dict), pi (B, Seq), entropy (B)
        msg_len = 5
        batch_size = 4
        
        def model_side_effect(input, *args, **kwargs):
            # Check input batch size. Input is dict, e.g. input['depot']
            if isinstance(input, dict):
                first_val = next(iter(input.values()))
                current_batch_size = first_val.size(0)
            else:
                current_batch_size = 4
                
            return (
                torch.randn(current_batch_size, requires_grad=True), # cost
                torch.randn(current_batch_size, requires_grad=True), # log_probs
                {'total': torch.randn(current_batch_size)},          # cost_dict
                torch.randn(current_batch_size, msg_len),            # pi
                torch.randn(current_batch_size, requires_grad=True)  # entropy
            )
        
        model.side_effect = model_side_effect
        # Ensure model is callable
        model.to = MagicMock(return_value=model)
        model.train = MagicMock()
        model.eval = MagicMock()
        
        optimizer = MagicMock()
        baseline = MagicMock()
        # baseline.wrap_dataset returns the dataset itself (identity)
        baseline.wrap_dataset.side_effect = lambda x: x
        baseline.unwrap_batch.side_effect = lambda x: (x, None)
        baseline.eval.return_value = (torch.zeros(4), torch.zeros(1))
        
        dataset = MagicMock()
        dataset.__len__.return_value = 4
        # Mock dataset iteration
        dataset.__getitem__ = MagicMock(return_value={'input': torch.tensor([1])})

        problem = MagicMock()
        problem.NAME = 'vrpp'

        return {
            'model': model,
            'optimizer': optimizer,
            'baseline': baseline,
            'training_dataset': dataset,
            'val_dataset': dataset,
            'problem': problem
        }

    def test_sapo_init(self, mock_deps):
        """Verify SAPOTrainer initialization and hyperparameter loading."""
        opts = {
            'rl_algorithm': 'sapo',
            'epoch_start': 0,
            'sapo_tau_pos': 0.2,
            'sapo_tau_neg': 0.9,
            'ppo_epochs': 2,
            'batch_size': 4,
            'device': torch.device('cpu'),
            'temporal_horizon': 0
        }
        
        trainer = SAPOTrainer(
            mock_deps['model'], 
            mock_deps['optimizer'], 
            mock_deps['baseline'], 
            None, # lr_scheduler
            None, # scaler
            mock_deps['val_dataset'], 
            mock_deps['problem'], 
            None, # tb_logger
            {'cost': 1.0}, # cost_weights
            opts
        )
        
        assert trainer.tau_pos == 0.2
        assert trainer.tau_neg == 0.9
        assert trainer.ppo_epochs == 2

    def test_sapo_update_step(self, mock_deps):
        """Verify SAPOTrainer execution includes SAPO update loop."""
        deps = mock_deps
        opts = {
            'rl_algorithm': 'sapo',
            'train_time': False, # Standard
            'device': torch.device('cpu'),
            'no_progress_bar': True,
            'ppo_epochs': 2, # Run update loop twice
            'ppo_mini_batch_size': 2,
            'batch_size': 4,
            'epoch_start': 0,
            'model': 'am',
            'temporal_horizon': 0,
            'focus_graph': None,
            'encoder': 'gat',
            'accumulation_steps': 1,
            'sapo_tau_pos': 0.1,
            'sapo_tau_neg': 1.0, 
            'entropy_weight': 0.01,
            'baseline': 'rollout'
        }
        
        trainer = SAPOTrainer(
            deps['model'], 
            deps['optimizer'], 
            deps['baseline'], 
            None, None, 
            deps['val_dataset'], 
            deps['problem'], 
            None, 
            {'cost': 1.0}, 
            opts
        )
        trainer.training_dataset = deps['training_dataset']
        trainer.step = 0
        
        # Mock DataLoader to return a batch
        # Dataset returns dict, DataLoader collates.
        # We manually mock the loop in train_day_sapo if needed, 
        # OR we rely on DataLoader working on the mock dataset.
        # Mock dataset needs to behave like a map-style dataset.
        # Let's make training_dataset a simple list of dicts.
        dummy_batch = {'loc': torch.randn(4, 10, 2), 'depot': torch.randn(4, 2)}
        # make_dataset usually returns a class inheriting Dataset.
        # For simplicity, we patch DataLoader in the module to return our batch.
        
        # Run one training day
        # Patch DataLoader inside sapo.py
        import logic.src.pipeline.reinforcement_learning.core.sapo as sapo_module
        original_dl = torch.utils.data.DataLoader
             
        # Mock DataLoader iterator
        class MockDL:
             def __init__(self, *args, **kwargs): pass
             def __iter__(self):
                 yield dummy_batch
                     
        sapo_module.torch.utils.data.DataLoader = MockDL
             
        # Patch prepare_batch to just return batch (identity) since our batch is already tensors
        # But prepare_batch adds things. We can mock it.
        # Alternatively, let's just make ensure dummy_batch has what's needed.
        # prepare_batch is imported in sapo.py, we can patch it there.
        sapo_module.prepare_batch = MagicMock(return_value=dummy_batch)
             
        try:
             trainer.train_day()
        finally:
             # Restore
             sapo_module.torch.utils.data.DataLoader = original_dl
        
        # Assertions
        # 1. Collection phase: train_batch called once per batch (1 batch)
        # 2. Update phase: update_sapo called.
        # Inside update_sapo: model() called PPO_EPOCHS times per batch.
        # We have 1 batch from collection. PPO epochs = 2.
        # So model() should be called 1 (collection) + 2 (update) = 3 times.
        
        # Actually in update_phase, we iterate rollouts.
        # rollouts = 1.
        # inner loop loops ppo_epochs = 2.
        # Each epoch splits into mini_batches. Batch size 4, mini 2 => 2 mini batches.
        # So 2 epochs * 2 mini-batches = 4 updates.
        # Total model calls = 1 (collection) + 4 (update) = 5.
        
        assert deps['model'].call_count >= 5
        
        # Check optimizer step
        # Called once per mini-batch update => 4 times.
        assert deps['optimizer'].step.call_count == 4

