import pytest
import torch
from unittest.mock import MagicMock
from logic.src.pipeline.reinforcement_learning.core.dr_grpo import DRGRPOTrainer

class TestDRGRPO:
    """Tests for DR-GRPO Trainer implementation."""

    @pytest.fixture
    def mock_deps(self):
        model = MagicMock()
        
        # Dynamic return value for model dependent on input size
        def model_side_effect(input, return_pi=False, expert_pi=None, imitation_mode=False, **kwargs):
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
        model.__call__ = MagicMock(side_effect=model_side_effect)
        
        # Allow setting attributes
        model.decode_type = 'sampling'

        optimizer = MagicMock()
        baseline = MagicMock()
        baseline.wrap_dataset.side_effect = lambda x: x
        baseline.unwrap_batch.side_effect = lambda x: (x, None)
        
        def baseline_eval_side_effect(input, c=None):
             if isinstance(input, dict):
                 first = next(iter(input.values()))
                 bs = first.size(0)
             else:
                 bs = 4
             return (torch.zeros(bs), torch.zeros(1))
        
        baseline.eval.side_effect = baseline_eval_side_effect 
        
        dataset = MagicMock()
        dataset.__len__.return_value = 4
        # Return a tensor that can be repeated
        dataset.__getitem__ = MagicMock(return_value={'input': torch.tensor([1.0, 2.0])})

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

    def test_dr_grpo_init(self, mock_deps):
        opts = {
            'batch_size': 4,
            'dr_grpo_group_size': 8,
            'dr_grpo_epsilon': 0.2,
            'dr_grpo_epochs': 3,
            'ppo_mini_batch_size': 16,
            'device': 'cpu',
            'no_progress_bar': True,
            'w_waste': 1.0, 'w_length': 1.0, 'w_overflows': 1.0, 'w_lost': 1.0, 'w_penalty': 1.0, 'w_prize': 1.0,
            'lr_model': 1e-4, 'lr_critic_value': 1e-4, 'max_grad_norm': 1.0, 'entropy_weight': 0.01,
            'model': 'am', 'problem': 'vrpp', 'baseline': None, 'enable_scaler': False,
            'epoch_start': 0
        }
        
        trainer = DRGRPOTrainer(
            mock_deps['model'],
            mock_deps['optimizer'],
            mock_deps['baseline'],
            MagicMock(), # lr_scheduler
            MagicMock(), # scaler
            mock_deps['training_dataset'],
            mock_deps['problem'],
            MagicMock(), # tb_logger
            {k: opts[k] for k in ['w_waste', 'w_length', 'w_overflows', 'w_lost', 'w_penalty', 'w_prize']},
            opts
        )
        
        assert trainer.group_size == 8
        assert trainer.epsilon == 0.2
        assert trainer.dr_grpo_epochs == 3
        mock_deps['model'].set_decode_type.assert_called_with('sampling')

    def test_dr_grpo_train_day(self, mock_deps):
        opts = {
            'batch_size': 2,
            'dr_grpo_group_size': 4, # G=4
            'dr_grpo_epsilon': 0.2,
            'dr_grpo_epochs': 1,
            'ppo_mini_batch_size': 8, # Process all (2*4=8)
            'device': 'cpu',
            'no_progress_bar': True,
            'w_waste': 1.0, 'w_length': 1.0, 'w_overflows': 1.0, 'w_lost': 1.0, 'w_penalty': 1.0, 'w_prize': 1.0,
            'lr_model': 1e-4, 'lr_critic_value': 1e-4, 'max_grad_norm': 1.0, 'entropy_weight': 0.0,
            'model': 'am', 'problem': 'vrpp', 'baseline': 'rollout', 'bl_alpha': 0.1, 'enable_scaler': False,
            'epoch_start': 0, 'temporal_horizon': 0, 'focus_graph': None, 'focus_size': 0, 'train_time': False
        }
        
        trainer = DRGRPOTrainer(
            mock_deps['model'],
            mock_deps['optimizer'],
            mock_deps['baseline'],
            MagicMock(),
            MagicMock(),
            mock_deps['training_dataset'],
            mock_deps['problem'],
            MagicMock(),
            {k: opts[k] for k in ['w_waste', 'w_length', 'w_overflows', 'w_lost', 'w_penalty', 'w_prize']},
            opts
        )
        
        # Mock init_dataset to set valid dataset
        trainer.training_dataset = mock_deps['training_dataset']
        
        # We need to ensure 'prepare_batch' works or is mocked if imported generally.
        # Since it is imported in dr_grpo.py from logic...core.epoch, we can't easily mock it locally 
        # unless we patch 'logic.src.pipeline.reinforcement_learning.core.dr_grpo.prepare_batch'.
        # However, prepare_batch is simple tensor moving. With 'cpu' device it should pass.
        
        # We need to ensure the dataloader returns something valid.
        # The mock dataset returns {'input': tensor([1, 2])}.
        # Collation might fail if not handled? 
        # Pytorch default collate handles tensors.
        
        trainer.train_day()
             
        # Check that optimizer step was called
        assert mock_deps['optimizer'].step.called
        assert mock_deps['optimizer'].zero_grad.called
