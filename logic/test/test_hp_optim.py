import pytest
import shutil
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch, ANY, call

from logic.src.pipeline.reinforcement_learning.hpo import (
    compute_focus_dist_matrix,
    optimize_model,
    validate,
    distributed_evolutionary_algorithm,
    bayesian_optimization,
    hyperband_optimization,
    random_search,
    grid_search,
    differential_evolutionary_hyperband_optimization,
    _ray_tune_trainable
)
from logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.dehb import (
    DifferentialEvolutionHyperband,
    get_config_space
)


class TestHPOFunctions:

    @patch('logic.src.pipeline.reinforcement_learning.hpo.load_focus_coords')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.compute_distance_matrix')
    def test_compute_focus_dist_matrix(self, mock_compute_dist, mock_load_coords):
        mock_load_coords.return_value = np.zeros((5, 2))
        mock_compute_dist.return_value = np.zeros((5, 5))
        
        result = compute_focus_dist_matrix(20, 'graph', 'area')
        
        assert isinstance(result, torch.Tensor)
        mock_load_coords.assert_called_once()
        mock_compute_dist.assert_called_once()

    @patch('logic.src.pipeline.reinforcement_learning.hpo.load_problem')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.load_data')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.setup_model_and_baseline')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.setup_optimizer_and_lr_scheduler')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.train_reinforce_epoch')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.validate')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.dump')
    def test_optimize_model(self, mock_json_dump, mock_open, mock_makedirs,
                          mock_validate, mock_train, mock_setup_opt, mock_setup_model,
                          mock_load_data, mock_load_prob, hpo_opts):
        
        mock_prob = MagicMock()
        mock_prob.make_dataset.return_value = MagicMock()
        mock_load_prob.return_value = mock_prob
        
        mock_model = MagicMock()
        mock_baseline = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_baseline)
        
        mock_setup_opt.return_value = (MagicMock(), MagicMock())
        mock_validate.return_value = (0.5, 0.5, {})

        result = optimize_model(hpo_opts, {}, metric='loss')
        
        assert result == (0.5, 0.5, {})
        mock_train.assert_called()
        mock_validate.assert_called()

    @patch('logic.src.pipeline.reinforcement_learning.hpo.get_inner_model')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.set_decode_type')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.move_to')
    @patch('torch.utils.data.DataLoader')
    def test_validate(self, mock_dataloader, mock_move_to, mock_set_decode, mock_get_inner, hpo_opts):
        mock_model = MagicMock()
        mock_dataset = MagicMock()
        mock_dataloader.return_value = [MagicMock()] # Single batch
        
        mock_inner = MagicMock()
        mock_inner.compute_batch_sim.return_value = (
            torch.tensor([1.0]), 
            {'overflows': torch.tensor([0.0]), 'kg': torch.tensor([10.0]), 'km': torch.tensor([5.0])}, 
            {'attention_weights': torch.tensor([0.1]), 'graph_masks': torch.tensor([0])}
        )
        mock_get_inner.return_value = mock_inner
        
        # Mock move_to to return whatever is passed (or identity for tensors)
        def side_effect(x, dev):
            return x
        mock_move_to.side_effect = side_effect

        dist_matrix = torch.zeros(5, 5)
        avg_cost, avg_ucost, all_costs = validate(mock_model, mock_dataset, 'overflows', dist_matrix, hpo_opts)
        
        assert isinstance(avg_cost, torch.Tensor)
        mock_set_decode.assert_called_with(mock_model, "greedy")

    @patch('logic.src.pipeline.reinforcement_learning.hpo.optimize_model')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.setup_cost_weights')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.algorithms.eaSimple')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.tools')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.creator')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.base.Toolbox')
    def test_distributed_evolutionary_algorithm(self, mock_toolbox_cls, mock_creator, mock_tools, 
                                              mock_eaSimple, mock_setup_weights, mock_optimize, hpo_opts):
        mock_toolbox = mock_toolbox_cls.return_value
        
        # Mock population and best individual
        mock_ind = MagicMock()
        mock_ind.fitness.values = [0.1]
        mock_ind.__iter__.return_value = [0.1, 0.1, 0.1, 0.1] # weights
        mock_tools.selBest.return_value = [mock_ind]
        
        mock_optimize.return_value = (0.1, 0.1, {})
        
        best_params = distributed_evolutionary_algorithm(hpo_opts)
        
        assert isinstance(best_params, dict)
        assert 'w_lost' in best_params
        mock_eaSimple.assert_called()

    @patch('logic.src.pipeline.reinforcement_learning.hpo.optimize_model')
    @patch('optuna.create_study')
    @patch('joblib.dump')
    @patch('os.makedirs')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.plot_optimization_history')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.plot_param_importances')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.plot_intermediate_values')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.dump')    
    def test_bayesian_optimization(self, mock_json, mock_open, mock_plot_inter, mock_plot_param, mock_plot_hist,
                                 mock_makedirs, mock_dump, mock_create_study, mock_optimize, hpo_opts):
        mock_study = MagicMock()
        mock_trial = MagicMock()
        mock_trial.value = 0.1
        mock_trial.params = {'w_lost': 0.1}
        mock_study.best_trial = mock_trial
        mock_study.trials = [mock_trial]
        mock_create_study.return_value = mock_study
        
        mock_optimize.return_value = (0.1, 0.1, {})
        
        mock_fig = MagicMock()
        mock_plot_hist.return_value = mock_fig
        mock_plot_param.return_value = mock_fig
        mock_plot_inter.return_value = mock_fig

        best_params = bayesian_optimization(hpo_opts)
        
        assert best_params == {'w_lost': 0.1}
        mock_study.optimize.assert_called()


    @patch('logic.src.pipeline.reinforcement_learning.hpo.tune.run')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.ray.init')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.HyperBandScheduler')
    def test_hyperband_optimization(self, mock_scheduler, mock_init, mock_tune_run, hpo_opts):
        mock_analysis = MagicMock()
        mock_analysis.get_best_config.return_value = {'w_lost': 0.1}
        mock_analysis.get_best_trial.return_value = MagicMock(last_result={'score': 0.1})
        mock_tune_run.return_value = mock_analysis
        
        best_config = hyperband_optimization(hpo_opts)
        
        assert best_config == {'w_lost': 0.1}
        mock_tune_run.assert_called()

    @patch('logic.src.pipeline.reinforcement_learning.hpo.tune.run')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.ray.init')
    def test_random_search(self, mock_init, mock_tune_run, hpo_opts):
        mock_analysis = MagicMock()
        mock_analysis.get_best_config.return_value = {'w_lost': 0.1}
        mock_analysis.get_best_trial.return_value = MagicMock(last_result={'score': 0.1})
        mock_tune_run.return_value = mock_analysis

        best_config = random_search(hpo_opts)
        
        assert best_config == {'w_lost': 0.1}
        mock_tune_run.assert_called()

    @patch('logic.src.pipeline.reinforcement_learning.hpo.tune.run')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.ray.init')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.ASHAScheduler')
    def test_grid_search(self, mock_asha, mock_init, mock_tune_run, hpo_opts):
        mock_result = MagicMock()
        mock_trial = MagicMock()
        mock_trial.config = {'w_lost': 0.1}
        mock_trial.last_result = {'validation_metric': 0.1}
        mock_result.get_best_trial.return_value = mock_trial
        
        mock_tune_run.return_value = mock_result
        
        hpo_opts['grid'] = [0.1, 0.2] # Provide grid
        best_config = grid_search(hpo_opts)
        
        assert best_config == {'w_lost': 0.1}
        mock_tune_run.assert_called()

    @patch('logic.src.pipeline.reinforcement_learning.hpo.DifferentialEvolutionHyperband')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.compute_focus_dist_matrix')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.optimize_model')
    def test_differential_evolutionary_hyperband_optimization(self, mock_optimize, mock_compute_dist, mock_dehb_cls, hpo_opts):
        mock_dehb = mock_dehb_cls.return_value
        mock_dehb.run.return_value = (None, 1.0, []) # traj, runtime, history
        mock_dehb.get_incumbents.return_value = ({'w_lost': 0.1}, 0.1)
        
        mock_optimize.return_value = (0.1, 0.1, {}) # cost, ucost, dict
        
        best_config = differential_evolutionary_hyperband_optimization(hpo_opts)
        
        assert best_config == {'w_lost': 0.1}
        mock_dehb.run.assert_called()
        
    @patch('logic.src.pipeline.reinforcement_learning.hpo.optimize_model')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.tune.report')
    @patch('logic.src.pipeline.reinforcement_learning.hpo.tune.get_trial_id')
    def test_ray_tune_trainable(self, mock_get_trial_id, mock_report, mock_optimize, hpo_opts):
        config = {'w_lost': 0.1, 'w_waste': 0.1, 'w_length': 0.1, 'w_overflows': 0.1}
        mock_optimize.return_value = (0.1, 0.1, {})
        mock_get_trial_id.return_value = "trial_id"
        
        _ray_tune_trainable(hpo_opts, config)
        
        mock_optimize.assert_called()
        mock_report.assert_called_with(score=0.1)


class TestDEHB:
    
    def test_get_config_space(self, hpo_opts):
        cs = get_config_space(hpo_opts)
        assert hasattr(cs, 'get_hyperparameters')
        hps = cs.get_hyperparameters()
        names = [hp.name for hp in hps]
        assert 'w_lost' in names
        assert 'w_prize' in names
        assert 'w_length' in names
        assert 'w_overflows' in names

    @patch('logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.dehb.logger')
    @patch('logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.dehb.Client')
    def test_dehb_init(self, mock_client, mock_logger, hpo_opts):
        cs = get_config_space(hpo_opts)
        f = MagicMock()
        
        dehb = DifferentialEvolutionHyperband(
            cs=cs,
            f=f,
            min_fidelity=1,
            max_fidelity=10,
            n_workers=1,
            client=None, # Use None to avoid actual Dask client logic if possible or mock it
            output_path='test_dehb_output'
        )
        
        assert dehb.min_fidelity == 1
        assert dehb.max_fidelity == 10
        assert dehb.dimensions == 4 # 4 params in wcrp config space
