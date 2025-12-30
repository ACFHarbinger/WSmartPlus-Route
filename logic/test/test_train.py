import torch
import pandas as pd

from unittest.mock import MagicMock
from logic.src.pipeline.train import (
    run_training, 
    hyperparameter_optimization,
    train_reinforcement_learning,
    train_meta_reinforcement_learning
)
from logic.src.pipeline.reinforcement_learning import reinforce, epoch, post_processing


class TestTrain:
    """Tests for train.py orchestration logic."""

    def test_run_training_dispatches_train(self, mocker):
        """Verify run_training dispatches to train_reinforcement_learning for 'train' command."""
        mock_train_func = mocker.patch('logic.src.pipeline.train.train_reinforcement_learning')
        
        args = {'seed': 42, 'train_time': False}
        run_training(args, 'train')
        
        mock_train_func.assert_called_once()
        # Should call with epoch loop if train_time is False
        assert mock_train_func.call_args[0][0] == args
        assert mock_train_func.call_args[0][1] == reinforce.train_reinforce_epoch

    def test_run_training_dispatches_train_time(self, mocker):
        """Verify run_training uses train_reinforce_over_time if train_time is True."""
        mock_train_func = mocker.patch('logic.src.pipeline.train.train_reinforcement_learning')
        
        args = {'seed': 42, 'train_time': True}
        run_training(args, 'train')
        
        assert mock_train_func.call_args[0][1] == reinforce.train_reinforce_over_time

    def test_run_training_dispatches_hp_optim(self, mocker):
        """Verify run_training dispatches to hyperparameter_optimization."""
        mock_hop = mocker.patch('logic.src.pipeline.train.hyperparameter_optimization')
        
        args = {'seed': 42}
        run_training(args, 'hp_optim')
        
        mock_hop.assert_called_once_with(args)

    def test_hyperparameter_optimization_dispatch(self, mocker):
        """Verify hyperparameter_optimization calls correct search method."""
        # Mock directory creation to avoid OS errors
        mocker.patch('os.makedirs')
        mocker.patch('json.dump')
        mocker.patch('builtins.open')
        
        # Mocks for search methods and train_best
        mock_grid = mocker.patch('logic.src.pipeline.train.grid_search', return_value={'lr': 0.1})
        mock_train_final = mocker.patch('logic.src.pipeline.train.train_reinforcement_learning')
        mocker.patch('logic.src.pipeline.train.setup_cost_weights')
        
        opts = {
            'cpu_cores': 1, 'save_dir': 'test', 'hop_method': 'gs', 
            'train_best': True, 'run_name': 'test_run'
        }
        
        hyperparameter_optimization(opts)
        
        mock_grid.assert_called_once()
        mock_train_final.assert_called_once()


class TestTrainFunctions:
    """Tests for core training functions in train.py."""

    def test_train_meta_reinforcement_learning_dispatch(self, mocker):
        """Verify MRL method dispatch logic."""
        mock_train_rl = mocker.patch('logic.src.pipeline.train.train_reinforcement_learning')
        
        # Test CB dispatch
        opts = {'mrl_method': 'cb'}
        from logic.src.pipeline.reinforcement_learning.reinforce import train_reinforce_over_time_cb
        train_meta_reinforcement_learning(opts)
        mock_train_rl.assert_called_with(opts, train_reinforce_over_time_cb)
        
        # Test Unknown
        opts = {'mrl_method': 'unknown'}
        ret = train_meta_reinforcement_learning(opts)
        assert ret == 1

    def test_train_reinforcement_learning_orchestration(self, mocker):
        """Verify full training setup orchestration."""
        # Mock ALL setup functions and dependencies
        mocker.patch('logic.src.pipeline.train.setup_cost_weights', return_value={})
        mocker.patch('logic.src.pipeline.train.TbLogger')
        mocker.patch('os.makedirs')
        mocker.patch('builtins.open')
        mocker.patch('json.dump')
        mocker.patch('logic.src.pipeline.train.load_problem')
        mocker.patch('logic.src.pipeline.train.load_data')
        
        mock_model = MagicMock()
        mock_baseline = MagicMock()
        mocker.patch('logic.src.pipeline.train.setup_model_and_baseline', return_value=(mock_model, mock_baseline))
        mocker.patch('logic.src.pipeline.train.setup_optimizer_and_lr_scheduler', return_value=(MagicMock(), MagicMock()))
        mocker.patch('logic.src.pipeline.train.wandb')
        mocker.patch('torch.cuda.is_available', return_value=False)
        mocker.patch('torch.cuda.amp.GradScaler')
        mocker.patch('torch.save')
        
        # Mock training function
        # Expects: model, optimizer -> (model, optimizer) or similar? 
        # Actually train_reinforcement_learning loops and calls train_func.
        # But here mock_train_func is the *second argument* to train_reinforcement_learning?
        # No, train_reinforcement_learning(opts, model=None)...
        # Wait, the test calls `train_reinforcement_learning(opts, mock_train_func)`.
        # `train_reinforcement_learning` signature in `logic/src/pipeline/train.py`:
        # def train_reinforcement_learning(opts, model=None, baseline=None):
        # The test passes `mock_train_func` as `model`?
        # Ah, the test in line 123 calls `train_reinforcement_learning(opts, mock_train_func)`.
        # This implies `mock_train_func` is the MODEL.
        # So `dataset` iteration inside `epoch` calls `model(x)`.
        # But `train_reinforcement_learning` calls `reinforce.train_reinforce_over_time...`.
        # I should check `train.py`.
        # Assuming `mock_train_func` is passed as `model` and called?
        # No, `mock_train_func` is likely expected to be `train_reinforce_...` function?
        # But `train_reinforcement_learning` imports it.
        # Let's verify `train_reinforcement_learning` args.
        # View `train.py` if needed.
        # Assuming the test is testing `train_reinforcement_learning` wrapper.
        # If `train_reinforcement_learning` calls `reinforce.train_reinforce_over_time`, and that is mocked?
        # The test doesn't seem to mock `reinforce.train...`.
        # It mocks `logic.src.pipeline.train.setup_model_and_baseline`.
        # It mocks `logic.src.pipeline.train.wandb`.
        # It defines `mock_train_func`.
        # Line 123: `train_reinforcement_learning(opts, mock_train_func)`.
        # If `mock_train_func` is the model, then `train_reinforce_over_time` (which is actual code) will call it. 
        # But `train_reinforce_over_time` requires a model.
        # This test (line 97 `TestTrain`) seems to be testing `train.py`.
        # I'll leave this one alone if it's just checking specific calls, unless it fails.
        # The failure was in `TestReinforce` tests.
        
        mock_train_func = MagicMock(return_value=(mock_model, MagicMock()))
        
        opts = {
            'no_tensorboard': False, 'log_dir': 'logs', 'run_name': 'test', 'problem': 'vrpp', 'graph_size': 10,
            'save_dir': 'save', 'no_cuda': True, 'load_path': None, 'resume': None, 
            'wandb_mode': 'disabled', 'eval_only': False, 'enable_scaler': False,
            'train_time': False, 'epoch_start': 0, 'n_epochs': 1,
            # dummy vals for make_dataset
             'val_size': 10, 'area': None, 'waste_type': None, 'dm_filepath': None,
             'val_dataset': None, 'data_distribution': None, 'vertex_method': None, 
             'distance_method': None, 'edge_threshold': None, 'edge_method': None,
             'focus_graph': None, 'eval_focus_size': None, 'final_dir': 'temp'
        }
        
        train_reinforcement_learning(opts, mock_train_func)
        
        mock_train_func.assert_called()


class TestReinforce:
    """Tests for enforce.py training logic."""

    def test_train_batch_reinforce(self, mocker, mock_train_model, mock_optimizer, mock_baseline):
        """Verify train_batch_reinforce computes loss and steps optimizer."""
        # Setup inputs
        scaler = None
        epoch = 1
        batch_id = 0
        step = 0
        batch = {'inputs': torch.tensor([1, 2])} # Dummy batch
        tb_logger = MagicMock()
        cost_weights = {'length': 1.0, 'waste': 1.0}
        opts = {
            'device': 'cpu', 'train_time': False, 'accumulation_steps': 1,
            'max_grad_norm': 1.0, 'log_step': 10, 'baseline': None,
            'problem': 'vrpp', 'no_tensorboard': True, 'wandb_mode': 'disabled'
        }
        
        # Configure model return values (5 values)
        mock_train_model.return_value = (torch.tensor(0.0), torch.zeros(2), {}, torch.zeros(2,2), torch.tensor(0.0))

        # Mock baseline wrap/unwrap
        mock_baseline.unwrap_batch.return_value = (batch['inputs'], None) # x, bl_val

        # Call function
        pi, c_dict, l_dict, cost = reinforce.train_batch_reinforce(
            mock_train_model, mock_optimizer, mock_baseline, scaler, 
            epoch, batch_id, step, batch, tb_logger, cost_weights, opts
        )

        # Assertions
        assert 'nll' in l_dict
        assert 'reinforce_loss' in l_dict
        mock_optimizer.step.assert_called_once()
        mock_optimizer.zero_grad.assert_called_once()
        
    def test_train_single_day(self, mocker, mock_train_model, mock_optimizer, mock_baseline):
        """Verify _train_single_day iterates dataloader and calls complete_train_pass."""
        # Mocks
        mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.tqdm', side_effect=lambda x, **kwargs: x)
        mocker.patch('time.time', return_value=0.0)
        mock_prepare = mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.prepare_batch', return_value={})
        mock_batch_train = mocker.patch(
            'logic.src.pipeline.reinforcement_learning.reinforce.train_batch_reinforce',
            return_value=(torch.tensor([[1]]), {'c': 1}, {'l': 1}, torch.tensor([10.0]))
        )
        mock_complete = mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.complete_train_pass', return_value=None)
        mock_log_epoch = mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.log_epoch')
        mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.get_loss_stats', return_value=pd.Series({'l': 1}))
        
        # Inputs
        day_dataset = [1, 2, 3] # Dummy dataset
        val_dataset = []
        opts = {'batch_size': 1, 'no_progress_bar': True}
        cost_weights = {}
        loss_keys = ['l']
        table_df = pd.DataFrame(columns=['l'])

        reinforce._train_single_day(
            mock_train_model, mock_optimizer, mock_baseline, None, None, None,
            day_dataset, val_dataset, None, 1, 0, cost_weights, loss_keys, table_df, opts
        )
        
        assert mock_batch_train.call_count == 3 # 3 items, batch_size 1
        mock_complete.assert_called_once()
        mock_log_epoch.assert_called_once()

    def test_mrl_cb_update(self, mocker, mock_train_model, mock_optimizer, mock_baseline):
        """Verify Contextual Bandit update logic in train_reinforce_over_time_cb."""
        # Mocks
        mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.prepare_time_dataset', 
                     return_value=(0, MagicMock(), ['length', 'waste', 'total'], pd.DataFrame(), None))
        mock_train_day = mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce._train_single_day')
        # Return: step, log_pi, daily_loss, daily_total_samples, current_weights
        mock_train_day.return_value = (0, [], {'length': torch.tensor([10.0]), 'waste': torch.tensor([5.0])}, 10, {'length': 0.5, 'waste': 0.5}, {})
        
        mock_bandit = mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.WeightContextualBandit')
        mock_bandit_instance = mock_bandit.return_value
        mock_bandit_instance.update.return_value = {'trials': 1}
        mock_bandit_instance.get_best_config.return_value = {'length': 0.5, 'waste': 0.5}

        mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.update_time_dataset')
        mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.log_training')
        
        opts = {
            'epoch_start': 0, 'n_epochs': 1, 'cb_min_epsilon': 0.1, 'cb_epsilon_decay': 0.9,
            'no_tensorboard': True, 'mrl_history': 10
        }
        cost_weights = {'length': 1.0, 'waste': 1.0}
        
        reinforce.train_reinforce_over_time_cb(
            mock_train_model, mock_optimizer, mock_baseline, None, None, None, None, None, cost_weights, opts
        )
        
        mock_bandit_instance.update.assert_called_once()
        mock_train_day.assert_called_once()

    def test_train_reinforce_over_time(self, mocker, mock_train_model, mock_optimizer, mock_baseline):
        """Verify main time loop functionality."""
        # Mocks
        mock_ds = MagicMock()
        mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.prepare_time_dataset', 
                     return_value=(0, mock_ds, [], pd.DataFrame(), ()))
        mock_train_day = mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce._train_single_day',
                                      return_value=(0, [], {}, 0, {}, {}))
        mock_update_ds = mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.update_time_dataset',
                                     return_value=mock_ds)
        mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.log_training')
        mock_post = mocker.patch('logic.src.pipeline.reinforcement_learning.reinforce.post_processing_optimization')
        
        opts = {
            'epoch_start': 0, 'n_epochs': 2, 'no_tensorboard': True, 'post_processing_epochs': 5
        }
        
        reinforce.train_reinforce_over_time(
            mock_train_model, mock_optimizer, mock_baseline, None, None, None, None, None, {}, opts
        )
        
        assert mock_train_day.call_count == 2
        assert mock_update_ds.call_count == 1
        mock_post.assert_called_once() # Post processing triggered

class TestEpoch:
    """Tests for epoch.py functions."""

    def test_prepare_batch_shapes(self, mocker):
        """Verify batch preparation handles temporal horizon."""
        dataset = MagicMock()
        dataset.__len__.return_value = 10 # Fix length
        # Mock fill_history
        dataset.fill_history = torch.randn(10, 5, 3) # 10 samples, 5 nodes, 3 days
        
        batch = {'inputs': torch.randn(2, 5), 'fill': torch.randn(2, 5)} # Batch size 2
        opts = {
            'model': 'tam', 'temporal_horizon': 3, 'focus_graph': None, 'encoder': 'gat'
        }
        
        dataloader = MagicMock()
        dataloader.batch_size = 2
        
        processed_batch = epoch.prepare_batch(batch, 0, dataset, dataloader, opts)
        
        assert 'fill_history' in processed_batch
        assert processed_batch['fill_history'].shape == (2, 5, 3)

    def test_validate_update(self, mocker, mock_train_model):
        """Verify validate_update computes costs and new weights."""
        # Setup
        dataset = MagicMock()
        dataset.dist_matrix = torch.zeros(5, 5)
        # Mock dataloader iteration
        mock_loader = mocker.patch('logic.src.pipeline.reinforcement_learning.epoch.torch.utils.data.DataLoader')
        mock_loader.return_value = [{'input': torch.tensor([1])}] # One batch
        
        mocker.patch('logic.src.pipeline.reinforcement_learning.epoch.prepare_batch', return_value={'input': torch.tensor([1])})
        
        # Manually attach compute_batch_sim to bypass spec check
        mock_train_model.compute_batch_sim = MagicMock(return_value=(
            torch.tensor([10.0]), # ucost
            {'overflows': torch.tensor([1.0]), 'kg': torch.tensor([100.0]), 'km': torch.tensor([10.0])}, # costs
            {'attention_weights': torch.tensor([0.1]), 'graph_masks': torch.tensor([0])} # attn
        ))
        
        cw_dict = {'overflows': 1.0, 'waste': 1.0, 'length': 1.0}
        opts = {
            'eval_batch_size': 1, 'device': 'cpu', 'temporal_horizon': 0, 'model': 'am',
            'no_progress_bar': True, 'adaptation_rate': 0.1, 'constraint': 3.0
        }
        
        new_cw, avg_cost, all_costs = epoch.validate_update(mock_train_model, dataset, cw_dict, opts)
        
        assert isinstance(new_cw, dict)
        assert 'overflows' in new_cw
        assert avg_cost.item() == 10.0

class TestPostProcessing:
    """Tests for post_processing.py."""
    
    def test_efficiency_optimizer_forward(self):
        """Verify model forward pass shape."""
        model = post_processing.EfficiencyOptimizer(10)
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 10)
        
    def test_post_processing_loop(self, mocker, mock_train_model):
        """Verify optimization loop runs."""
        dataset = [{'waste': [10], 'dist': [[0]]}] * 2 # Dummy dataset
        
        mocker.patch('logic.src.pipeline.reinforcement_learning.post_processing.DataLoader', return_value=dataset)
        mocker.patch('logic.src.pipeline.reinforcement_learning.post_processing.decode_routes', return_value=[[0, 1, 0]])
        # Fix: return tensors to allow loss.backward()
        mocker.patch('logic.src.pipeline.reinforcement_learning.post_processing.calculate_efficiency', 
                     return_value=(torch.tensor(1.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)))
        
        # Mock main model output
        mock_train_model.return_value = torch.randn(1, 10)
        
        pp_model = post_processing.post_processing_optimization(
            mock_train_model, dataset, epochs=1
        )
        
        assert isinstance(pp_model, post_processing.EfficiencyOptimizer)


