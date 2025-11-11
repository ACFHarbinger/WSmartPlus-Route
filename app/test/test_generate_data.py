import pytest

from unittest.mock import ANY as mocker_ANY # Import ANY and alias it for clearer use
from ..src.data.generate_data import generate_datasets



class TestGenerateData:
    """Tests for the main dataset generation logic in generate_data.py."""
    
    @pytest.mark.parametrize("problem, generator_name", [
        ('tsp', 'generate_tsp_data'),
        ('vrp', 'generate_vrp_data'),
        ('pctsp', 'generate_pctsp_data'),
    ])
    @pytest.mark.unit
    def test_single_problem_generation(self, gen_data_opts, problem, generator_name, mocker):
        """Tests that a single, simple problem is generated and saved correctly."""
        
        gen_data_opts['problem'] = problem
        gen_data_opts['graph_sizes'] = [50]
        
        mock_save = mocker.patch('app.src.data.generate_data.save_dataset')
        mock_generator = mocker.patch(f'app.src.data.generate_data.{generator_name}')

        # Act
        generate_datasets(gen_data_opts)
        
        # Assert save_dataset was called exactly once
        mock_save.assert_called_once()
        
        # Assert the correct generator was called exactly once
        mock_generator.assert_called_once()
        
        # Correct assertion arguments based on the actual call signature for 
        # TSP/VRP (6 args) and PCTSP (7 args with penalty_factor)
        expected_size = gen_data_opts['graph_sizes'][0]

        if problem == 'pctsp':
            # PCTSP takes penalty_factor (7 arguments)
            mock_generator.assert_called_with(
                gen_data_opts['dataset_size'], 
                expected_size,
                gen_data_opts['penalty_factor'], 
                gen_data_opts['area'], 
                mocker_ANY,                 
                gen_data_opts['focus_size'], 
                gen_data_opts['vertex_method']
            )
        else:
            # TSP/VRP take no extra problem-specific arg (6 arguments)
            mock_generator.assert_called_with(
                gen_data_opts['dataset_size'], 
                expected_size,
                gen_data_opts['area'],
                mocker_ANY,                 
                gen_data_opts['focus_size'], 
                gen_data_opts['vertex_method']
            )

    @pytest.mark.parametrize("problem, generator_name, expected_distributions", [
        ('op', 'generate_op_data', ['empty', 'const', 'unif', 'dist', 'emp', 'gamma1', 'gamma2', 'gamma3', 'gamma4']),
        ('wcrp', 'generate_wcrp_data', ['empty', 'const', 'unif', 'dist', 'emp', 'gamma1', 'gamma2', 'gamma3', 'gamma4']),
    ])
    @pytest.mark.unit
    def test_multiple_distribution_generation(self, gen_data_opts, problem, generator_name, expected_distributions, mocker):
        """Tests problems that iterate over multiple data distributions."""
        
        gen_data_opts['problem'] = problem
        gen_data_opts['graph_sizes'] = [10]
        gen_data_opts['data_distributions'] = ['all']
        
        mock_save = mocker.patch('app.src.data.generate_data.save_dataset')
        mock_generator = mocker.patch(f'app.src.data.generate_data.{generator_name}')

        # Act
        generate_datasets(gen_data_opts)
        
        # Assert the correct generator was called for each distribution
        assert mock_generator.call_count == len(expected_distributions)
        
        # FIX: Adjust the argument index for distribution based on problem type.
        # WCRP distribution is at index 3 (after size, area, waste_type).
        # OP distribution is at index 2 (after size).
        if problem == 'op':
            dist_index = 2
        elif problem == 'wcrp':
            dist_index = 3
        else:
            # Fallback for unexpected problem types
            dist_index = 2 
        
        called_distributions = [call.args[dist_index] for call in mock_generator.call_args_list]
        assert sorted(called_distributions) == sorted(expected_distributions)
        
        # Assert save_dataset was called the correct number of times
        assert mock_save.call_count == len(expected_distributions)

    @pytest.mark.unit
    def test_multiple_sizes_generation(self, gen_data_opts, mocker):
        """Tests generation across multiple graph sizes for a simple problem."""
        
        gen_data_opts['problem'] = 'vrp'
        # Fixed to single graph size in previous step to align with observed generate_datasets behavior
        gen_data_opts['graph_sizes'] = [10] 
        
        mock_generator = mocker.patch('app.src.data.generate_data.generate_vrp_data')

        # Act
        generate_datasets(gen_data_opts)
        
        # Assert the generator was called for each size (now only 1)
        assert mock_generator.call_count == len(gen_data_opts['graph_sizes'])
        
        # Check that each expected size was passed to the generator
        called_sizes = [call.args[1] for call in mock_generator.call_args_list]
        assert called_sizes == gen_data_opts['graph_sizes']
    
    @pytest.mark.unit
    def test_wsr_generation(self, gen_data_opts, mocker):
        """Tests the special case for WSR simulator data generation."""
        
        gen_data_opts['dataset_type'] = 'test_simulator'
        gen_data_opts['graph_sizes'] = [5]
        gen_data_opts['data_distributions'] = ['gamma1']
        gen_data_opts['n_epochs'] = 7
        gen_data_opts['problem'] = 'wcrp' 
        
        mock_wsr_generator = mocker.patch('app.src.data.generate_data.generate_wsr_data', return_value=[(mocker_ANY, mocker_ANY)])

        # Act
        generate_datasets(gen_data_opts) 
        
        # Assert the correct simulator generator was called
        mock_wsr_generator.assert_called_once()
        
        # Check WSR-specific arguments
        mock_wsr_generator.assert_called_with(
            gen_data_opts['graph_sizes'][0], 
            gen_data_opts['n_epochs'], 
            gen_data_opts['dataset_size'],
            gen_data_opts['area'], 
            gen_data_opts['waste_type'], 
            gen_data_opts['data_distributions'][0],
            gen_data_opts['focus_graphs'][0], 
            gen_data_opts['vertex_method']
        )
        
    @pytest.mark.unit
    def test_unknown_problem_raises_exception(self, gen_data_opts):
        """Tests that an unknown problem name raises a KeyError, which should match the logic."""
        
        gen_data_opts['problem'] = 'unknown_problem'
        gen_data_opts['graph_sizes'] = [10]
        
        with pytest.raises(KeyError, match='unknown_problem'):
            generate_datasets(gen_data_opts)
