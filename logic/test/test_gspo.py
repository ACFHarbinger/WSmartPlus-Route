import torch
import pytest
from unittest.mock import MagicMock
from logic.src.pipeline.reinforcement_learning.core.gspo import GSPOTrainer

class TestGSPO:
    """Tests for GSPO Trainer implementation."""

    @pytest.fixture
    def mock_deps(self):
        # Mock dependencies
        model = MagicMock()
        
        # Dynamic return value for model dependent on input size
        def model_side_effect(input, *args, **kwargs):
            if isinstance(input, dict):
                first_val = next(iter(input.values()))
                current_batch_size = first_val.size(0)
            else:
                current_batch_size = 4
            
            # Returns: cost, log_probs, cost_dict, pi, entropy
            # pi shape: [Batch, SeqLen]
            seq_len = 5
            return (
                torch.randn(current_batch_size, requires_grad=True), # cost
                torch.randn(current_batch_size, requires_grad=True), # log_probs
                {'total': torch.randn(current_batch_size)},          # cost_dict
                torch.randn(current_batch_size, seq_len),            # pi
                torch.randn(current_batch_size, requires_grad=True)  # entropy
            )
        
        model.side_effect = model_side_effect
        model.to = MagicMock(return_value=model)
        model.train = MagicMock()
        model.eval = MagicMock()
        
        optimizer = MagicMock()
        baseline = MagicMock()
        baseline.wrap_dataset.side_effect = lambda x: x
        baseline.unwrap_batch.side_effect = lambda x: (x, None)
        baseline.eval.return_value = (torch.zeros(4), torch.zeros(1)) 
        
        dataset = MagicMock()
        dataset.__len__.return_value = 4
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

    def test_gspo_init(self, mock_deps):
        """Verify GSPOTrainer initialization."""
        opts = {
            'rl_algorithm': 'gspo',
            'epoch_start': 0,
            'gspo_epsilon': 0.15,
            'gspo_epochs': 2,
            'batch_size': 4,
            'device': torch.device('cpu'),
            'temporal_horizon': 0
        }
        
        trainer = GSPOTrainer(
            mock_deps['model'], 
            mock_deps['optimizer'], 
            mock_deps['baseline'], 
            None, None, 
            mock_deps['val_dataset'], 
            mock_deps['problem'], 
            None, 
            {'cost': 1.0}, 
            opts
        )
        
        assert trainer.epsilon == 0.15
        assert trainer.gspo_epochs == 2

    def test_gspo_update_step(self, mock_deps):
        """Verify GSPOTrainer execution includes GSPO update loop."""
        deps = mock_deps
        opts = {
            'rl_algorithm': 'gspo',
            'train_time': False, 
            'device': torch.device('cpu'),
            'no_progress_bar': True,
            'gspo_epochs': 2, 
            'batch_size': 4,
            'epoch_start': 0,
            'model': 'am',
            'temporal_horizon': 0,
            'focus_graph': None,
            'encoder': 'gat',
            'accumulation_steps': 1,
            'gspo_epsilon': 0.2,
            'entropy_weight': 0.01,
            'baseline': 'rollout'
        }
        
        trainer = GSPOTrainer(
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
        
        # Mock DataLoader
        dummy_batch = {'depot': torch.randn(4, 2)}
        
        # Patch DataLoader inside gspo.py
        import logic.src.pipeline.reinforcement_learning.core.gspo as gspo_module
        original_dl = torch.utils.data.DataLoader
        
        class MockDL:
             def __init__(self, *args, **kwargs): pass
             def __iter__(self):
                 yield dummy_batch
                 
        gspo_module.torch.utils.data.DataLoader = MockDL
        gspo_module.prepare_batch = MagicMock(return_value=dummy_batch)
        # Mock prepare_batch to return dict, because TimeTrainer might use it.
        # But wait, TimeTrainer calls prepare_time_dataset which is mocked/not called here?
        # trainer.initialize_training_dataset called manually in real code, here we set dataset manually.
        # train_batch calls prepare_batch. 
        # But we mocked train_batch call? No, we call train_day_gspo which calls train_batch.
        # We need prepare_batch to be functional or mocked in reinforcement_learning.core.reinforce too?
        # gspo imports prepare_batch from epoch.
        # train_batch is in TimeTrainer (parent), which imports from ...
        # Let's inspect where train_batch comes from. BaseReinforceTrainer or TimeTrainer.
        # StandardTrainer has train_batch. TimeTrainer inherits StandardTrainer?
        # Let's assume TimeTrainer inherits StandardTrainer.
        
        try:
             trainer.train_day()
        finally:
             gspo_module.torch.utils.data.DataLoader = original_dl
        
        # Assertions
        # 1. Collection: 1 call
        # 2. Update: gspo_epochs * batch_count (2 * 1) = 2 calls
        # Total = 3 calls
        assert deps['model'].call_count >= 3
        
        # Optimizer step called once per update
        # 2 updates
        assert deps['optimizer'].step.call_count == 2
