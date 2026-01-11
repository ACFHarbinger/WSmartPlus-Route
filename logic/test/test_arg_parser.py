"""Tests for CLI argument parsing utilities."""
import sys
import copy
import pytest

from unittest.mock import patch
from logic.src.utils.arg_parser import (
    parse_params,
    ConfigsParser, LowercaseAction
)


class TestConfigsParser:
    """Test suite for ConfigsParser class"""
    @pytest.mark.arg_parser
    def test_lowercase_action(self):
        """Test LowercaseAction converts strings to lowercase"""
        parser = ConfigsParser()
        parser.add_argument('--test', action=LowercaseAction)
        args = parser.parse_args(['--test', 'UPPERCASE'])
        assert args.test == 'uppercase'
    
    @pytest.mark.arg_parser
    def test_lowercase_action_none(self):
        """Test LowercaseAction handles None values"""
        parser = ConfigsParser()
        parser.add_argument('--test', action=LowercaseAction, default=None)
        args = parser.parse_args([])
        assert args.test is None

    @pytest.mark.arg_parser
    def test_parse_process_args_basic(self):
        """
        Test basic parsing functionality where arguments are separated normally.
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest='command')
        cmd_parser = subparsers.add_parser('run')
        cmd_parser.add_argument('--value', type=int)
        
        # Test input: ['run', '--value', '10']
        command, args = parser.parse_process_args(['run', '--value', '10'])
        
        assert command == 'run'
        assert args['value'] == 10

    @pytest.mark.arg_parser
    def test_parse_process_args_nargs_string_splitting(self):
        """
        Test the custom logic: splitting a single string with spaces 
        into multiple arguments when nargs is defined.
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest='command')
        cmd_parser = subparsers.add_parser('run')
        # Arg with nargs='+'
        cmd_parser.add_argument('--list_items', nargs='+', type=str)
        
        # Scenario: Arguments passed as a single string "a b c" instead of 'a', 'b', 'c'
        # This triggers lines 66-70 in parse_process_args
        raw_args = ['run', '--list_items', 'item1 item2 item3']
        command, args = parser.parse_process_args(raw_args)
        
        assert command == 'run'
        assert args['list_items'] == ['item1', 'item2', 'item3']

    @pytest.mark.arg_parser
    def test_parse_process_args_nargs_standard(self):
        """
        Test that standard space-separated arguments still work 
        and aren't negatively affected by the splitting logic.
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest='command')
        cmd_parser = subparsers.add_parser('run')
        cmd_parser.add_argument('--list_items', nargs='+', type=str)
        
        # Standard input: ['run', '--list_items', 'item1', 'item2']
        raw_args = ['run', '--list_items', 'item1', 'item2']
        command, args = parser.parse_process_args(raw_args)
        
        assert command == 'run'
        assert args['list_items'] == ['item1', 'item2']

    @pytest.mark.arg_parser
    def test_parse_process_args_default_sys_argv(self):
        """
        Test that the method defaults to sys.argv[1:] if args=None.
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest='command')
        cmd_parser = subparsers.add_parser('test_cmd')
        cmd_parser.add_argument('--flag', action='store_true')
        
        # Mock sys.argv
        with patch.object(sys, 'argv', ['script_name.py', 'test_cmd', '--flag']):
            command, args = parser.parse_process_args(None)
            
            assert command == 'test_cmd'
            assert args['flag'] is True
    
    @pytest.mark.arg_parser
    def test_parser_initialization(self):
        """Test that parser initializes correctly"""
        parser = ConfigsParser(description="Test parser")
        assert parser is not None
        assert isinstance(parser, ConfigsParser)
    
    @pytest.mark.arg_parser
    def test_parse_command_valid(self):
        """Test parsing valid commands"""
        with patch.object(sys, 'argv', ['script.py', 'train']):
            parser = ConfigsParser()
            subparsers = parser.add_subparsers(dest="command")
            subparsers.add_parser('train')
            command = parser.parse_command()
            assert command == 'train'
    
    @pytest.mark.arg_parser
    def test_parse_command_invalid(self):
        """Test parsing invalid commands exits"""
        with patch.object(sys, 'argv', ['script.py', 'invalid_command']):
            parser = ConfigsParser()
            subparsers = parser.add_subparsers(dest="command")
            subparsers.add_parser('train')
            with pytest.raises(SystemExit):
                parser.parse_command()
    
    @pytest.mark.arg_parser
    def test_error_message(self, capsys):
        """Test error message printing"""
        parser = ConfigsParser()
        with pytest.raises(Exception):
            parser.error_message("Test error", print_help=False)
        captured = capsys.readouterr()
        assert "Test error" in captured.out

    @pytest.mark.edge_case
    def test_missing_required_command(self):
        """Test error when no command is provided"""
        with patch.object(sys, 'argv', ['script.py']):
            parser = ConfigsParser()
            with pytest.raises(SystemExit):        
                parse_params()
    
    @pytest.mark.edge_case
    def test_waste_type_normalization(self):
        """Test waste type normalization"""
        args = [
            'script.py', 'train',
            '--batch_size', '256',
            '--epoch_size', '128000',
            '--waste_type', 'Plas-tic'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['waste_type'] == 'plastic'
    
    @pytest.mark.edge_case
    def test_nargs_multiple_values(self):
        """Test arguments with multiple values (nargs='+')"""
        args = [
            'script.py', 'hp_optim',
            '--batch_size', '256',
            '--epoch_size', '128000',
            '--lrs_milestones', '7', '14', '21', '28'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['lrs_milestones'] == [7, 14, 21, 28]
    
    @pytest.mark.edge_case
    def test_af_urange_default(self):
        """Test activation function urange default values"""
        args = [
            'script.py', 'train',
            '--batch_size', '256',
            '--epoch_size', '128000'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['af_urange'] == [0.125, 1/3]

    @pytest.mark.edge_case
    def test_parse_process_args_mixed_types(self):
        """
        Test that splitting logic works correctly when mixed with other arguments
        and ignores flags (starts with '-').
        """
        parser = ConfigsParser()
        subparsers = parser.add_subparsers(dest='command')
        cmd_parser = subparsers.add_parser('run')
        cmd_parser.add_argument('--numbers', nargs='+', type=int)
        cmd_parser.add_argument('--other', type=str)
        
        # Input with numbers as a single string and a separate flag following it
        raw_args = ['run', '--numbers', '1 2 3', '--other', 'value']
        command, args = parser.parse_process_args(raw_args)
        
        assert command == 'run'
        # Should be converted to ints by the type=int in add_argument after splitting
        assert args['numbers'] == [1, 2, 3] 
        assert args['other'] == 'value'


class TestFileSystemCommand:
    """Test suite for file system command"""
    
    @pytest.mark.file_system
    def test_fs_update_command(self, base_file_system_update_args):
        """Test file system update command"""
        args = base_file_system_update_args + [
            '--output_key', 'waste',
            '--update_value', '1.5'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert command == ('file_system', 'update')
            assert args_dict['target_entry'] == 'path/to/file.pkl'
            assert args_dict['update_value'] == 1.5
    
    @pytest.mark.file_system
    def test_fs_delete_command(self, base_file_system_delete_args):
        """Test file system delete command"""
        args = base_file_system_delete_args + [
            '--log_dir', 'logs',
            '--log',
            '--delete_preview'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert command == ('file_system', 'delete')
            assert args_dict['log_dir'] == 'logs'
            assert args_dict['delete_preview'] is True
    
    @pytest.mark.file_system
    def test_fs_cryptography_command(self, base_file_system_crypto_args):
        """Test file system cryptography command"""
        args = base_file_system_crypto_args + [
            '--symkey_name', 'mykey',
            '--salt_size', '16',
            '--key_length', '32'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert command == ('file_system', 'cryptography')
            assert args_dict['symkey_name'] == 'mykey'
            assert args_dict['salt_size'] == 16

    @pytest.mark.file_system
    def test_missing_fs_subcommand(self):
        """Test error when no file system subcommand is provided"""
        # Scenario: User enters 'file_system' but forgets 'update', 'delete', or 'cryptography'
        with patch.object(sys, 'argv', ['script.py', 'file_system']):
            with pytest.raises(SystemExit):
                parse_params()

    @pytest.mark.file_system
    def test_invalid_fs_subcommand(self):
        """Test error when an invalid file system subcommand is provided"""
        # Scenario: User enters a subcommand that doesn't exist
        with patch.object(sys, 'argv', ['script.py', 'file_system', 'destroy']):
            with pytest.raises(SystemExit):
                parse_params()

    @pytest.mark.file_system
    def test_fs_update_mutual_exclusivity(self, base_file_system_update_args):
        """
        Test that update_operation and stats_function are mutually exclusive.
        We mock the mapping dictionaries to ensure the ActionFactory accepts the values,
        allowing the code to reach the validation logic in validate_file_system_args.
        """
        args = base_file_system_update_args + [
            '--update_operation', 'op_test',
            '--stats_function', 'stat_test'
        ]
        
        # Mock the maps used by UpdateFunctionMapActionFactory so our test inputs are considered valid
        # This allows us to bypass the ValueError in the Action and hit the AssertionError in validation
        with patch('logic.src.utils.arg_parser.OPERATION_MAP', {'op_test': 1}), \
            patch('logic.src.utils.arg_parser.STATS_FUNCTION_MAP', {'stat_test': 1}), \
            patch.object(sys, 'argv', args):
            
            # Expect AssertionError: "'update_operation' and 'stats_function' arguments are mutually exclusive"
            # note: parse_params catches the AssertionError and re-raises it via error_message()
            with pytest.raises(AssertionError, match="mutually exclusive"):
                parse_params()

    @pytest.mark.file_system
    def test_fs_input_keys_multiple(self, base_file_system_update_args):
        """Test input_keys argument accepting multiple values (nargs='*')"""
        args = base_file_system_update_args + [
            '--input_keys', 'key1', 'key2', 'key3'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert command == ('file_system', 'update')
            assert args_dict['input_keys'] == ['key1', 'key2', 'key3']


class TestGenDataCommand:
    """Test suite for generate data command"""
    
    @pytest.mark.gen_data
    def test_gen_data_basic(self, base_gen_data_args):
        """Test basic data generation"""
        args = base_gen_data_args + ['50']
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert command == 'gen_data'
            assert args_dict['problem'] == 'vrpp'
            assert args_dict['dataset_size'] == 10000
            assert args_dict['graph_sizes'] == [20, 50]
    
    @pytest.mark.gen_data
    def test_gen_data_with_focus_graphs(self, base_gen_data_args):
        """Test data generation with focus graphs"""
        args = base_gen_data_args[:2] + [
            '--graph_sizes', '20',
            '--focus_graphs', 'path/to/graph.txt',
            '--focus_size', '100',
            '--area', 'riomaior',
            '--waste_type', 'plastic',
            '--problem', 'vrpp'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['focus_graphs'] == ['path/to/graph.txt']
            assert args_dict['focus_size'] == 100
            assert args_dict['area'] == 'riomaior'
    
    @pytest.mark.gen_data
    def test_gen_data_filename_validation(self, base_gen_data_args):
        """Test that filename requires single dataset"""
        args = base_gen_data_args[:3] + [
            'all',
            '--graph_sizes', '20', '50',
            '--filename', 'test.pkl'
        ]
        print('#######' * 10 + 'Args:', args)
        with patch.object(sys, 'argv', args):
            with pytest.raises(AssertionError, match="Can only specify filename when generating a single dataset"):
                command, args_dict = parse_params()


class TestTrainCommand:
    """Test suite for train command"""

    @pytest.mark.train
    def test_train_default_parameters(self, base_train_args):
        """Test train command with default parameters"""
        with patch.object(sys, 'argv', base_train_args):
            command, args = parse_params()
            assert command == 'train'
            assert args['problem'] == 'vrpp'
            assert args['graph_size'] == 20
            assert args['batch_size'] == 256
            assert args['n_epochs'] == 25
    
    @pytest.mark.train
    def test_train_model_parameters(self, base_train_args):
        """Test train command with model parameters"""
        args = base_train_args + [
            '--model', 'am',
            '--encoder', 'gat',
            '--embedding_dim', '128',
            '--hidden_dim', '512',
            '--n_encode_layers', '3'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['model'] == 'am'
            assert args_dict['encoder'] == 'gat'
            assert args_dict['embedding_dim'] == 128
            assert args_dict['hidden_dim'] == 512
    
    @pytest.mark.train
    def test_train_optimizer_parameters(self, base_train_args):
        """Test train command with optimizer parameters"""
        args = base_train_args + [
            '--optimizer', 'adam',
            '--lr_model', '0.0001',
            '--lr_scheduler', 'exp',
            '--lr_decay', '0.99'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['optimizer'] == 'adam'
            assert args_dict['lr_model'] == 0.0001
            assert args_dict['lr_scheduler'] == 'exp'
            assert args_dict['lr_decay'] == 0.99
    
    @pytest.mark.train
    def test_train_epoch_size_validation(self):
        """Test that epoch_size must be multiple of batch_size"""
        args = [
            'script.py', 'train',
            '--batch_size', '256',
            '--epoch_size', '100000'  # Not divisible by 256
        ]
        with patch.object(sys, 'argv', args):
            with pytest.raises(AssertionError, match="Epoch size must be integer multiple of batch size"):
                parse_params()
    
    @pytest.mark.train
    def test_train_baseline_warmup(self, base_train_args):
        """Test baseline warmup epochs configuration"""
        args = base_train_args + ['--baseline', 'rollout']
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['bl_warmup_epochs'] == 1
    
    @pytest.mark.train
    def test_train_area_validation(self, base_train_args):
        """Test area parameter validation"""
        args = base_train_args + ['--area', 'Rio-Maior']
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['area'] == 'riomaior'
    
    @pytest.mark.train
    def test_train_edge_threshold_float(self, base_train_args):
        """Test edge_threshold as float"""
        args = base_train_args + ['--edge_threshold', '0.5']
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['edge_threshold'] == 0.5
            assert isinstance(args_dict['edge_threshold'], float)
    
    @pytest.mark.train
    def test_train_edge_threshold_int(self, base_train_args):
        """Test edge_threshold as integer"""
        args = base_train_args + ['--edge_threshold', '10']
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['edge_threshold'] == 10
            assert isinstance(args_dict['edge_threshold'], int)
    
    @pytest.mark.train
    def test_train_activation_functions(self, base_train_args):
        """Test various activation functions"""
        activations = ['gelu', 'relu', 'tanh', 'sigmoid', 'mish']
        for activation in activations:
            args = base_train_args + ['--activation', activation]
            with patch.object(sys, 'argv', args):
                command, args_dict = parse_params()
                assert args_dict['activation'] == activation
    
    @pytest.mark.train
    def test_train_cuda_and_scaler(self, base_train_args):
        """Test CUDA and scaler flags"""
        args = base_train_args + ['--no_cuda', '--enable_scaler']
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['no_cuda'] is True
            assert args_dict['enable_scaler'] is True


class TestHPOptimCommand:
    """Test suite for hyperparameter optimization command"""
    
    @pytest.mark.hp_optim
    def test_hp_optim_bayesian(self, base_hp_optim_args):
        """Test hyperparameter optimization with Bayesian method"""
        args = base_hp_optim_args + [
            '--n_trials', '20',
            '--metric', 'val_loss'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert command == 'hp_optim'
            assert args_dict['hop_method'] == 'bo'
            assert args_dict['n_trials'] == 20
            assert args_dict['metric'] == 'val_loss'
    
    @pytest.mark.hp_optim
    def test_hp_optim_grid_search(self, base_hp_optim_args):
        """Test hyperparameter optimization with grid search"""
        args = base_hp_optim_args[:-1] + [
            'gs',
            '--grid', '0.0', '0.5', '1.0'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['hop_method'] == 'gs'
            assert args_dict['grid'] == [0.0, 0.5, 1.0]
    
    @pytest.mark.hp_optim
    def test_hp_optim_evolutionary(self, base_hp_optim_args):
        """Test hyperparameter optimization with evolutionary algorithm"""
        args = base_hp_optim_args[:-1] + [
            'dea',
            '--n_pop', '20',
            '--n_gen', '10',
            '--mutpb', '0.2'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['hop_method'] == 'dea'
            assert args_dict['n_pop'] == 20
            assert args_dict['n_gen'] == 10


class TestMRLTrainCommand:
    """Test suite for MRL train command"""
    
    @pytest.mark.mrl_train
    def test_mrl_train_default(self, base_mrl_args):
        """Test MRL train with default parameters"""
        with patch.object(sys, 'argv', base_mrl_args):
            command, args_dict = parse_params()
            assert command == 'mrl_train'
            assert args_dict['mrl_method'] == 'cb'
            assert args_dict['mrl_history'] == 10
    
    @pytest.mark.mrl_train
    def test_mrl_contextual_bandits(self, base_mrl_args):
        """Test MRL with contextual bandits parameters"""
        args = base_mrl_args + [
            '--cb_exploration_method', 'ucb',
            '--cb_num_configs', '10',
            '--cb_epsilon_decay', '0.995'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['cb_exploration_method'] == 'ucb'
            assert args_dict['cb_num_configs'] == 10
            assert args_dict['cb_epsilon_decay'] == 0.995


class TestEvalCommand:
    """Test suite for evaluation command"""
    
    @pytest.mark.eval
    def test_eval_basic(self, base_eval_args):
        """Test basic evaluation"""
        args = base_eval_args + [
            '--eval_batch_size', '256'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert command == 'eval'
            assert args_dict['datasets'] == ['dataset1.pkl']
            assert args_dict['eval_batch_size'] == 256
    
    @pytest.mark.eval
    def test_eval_multiple_datasets(self, base_eval_args):
        """Test evaluation with multiple datasets"""
        args = base_eval_args[:3] + [
            'dataset1.pkl', 'dataset2.pkl', 'dataset3.pkl',
            '--decode_type', 'sampling',
            '--width', '10'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert len(args_dict['datasets']) == 3
            assert args_dict['decode_type'] == 'sampling'
    
    @pytest.mark.eval
    def test_eval_area_normalization(self, base_eval_args):
        """Test area name normalization in eval"""
        args = base_eval_args[:3] + [
            'test.pkl',
            '--area', 'Rio-Maior'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['area'] == 'riomaior'


class TestTestCommand:
    """Test suite for simulator test command"""

    @pytest.mark.test_sim
    def test_test_basic(self, base_test_args):
        """Test basic simulator test"""
        args = copy.deepcopy(base_test_args)
        args.insert(4, "policy2")
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert command == 'test_sim'
            assert args_dict['policies'] == ['policy1', 'policy2']
            assert args_dict['days'] == 31
            assert args_dict['size'] == 50
    
    @pytest.mark.test_sim
    def test_test_days_validation(self, base_test_args):
        """Test days parameter validation"""
        args = base_test_args[:-3] + ['0']
        with patch.object(sys, 'argv', args):
            with pytest.raises(AssertionError, match="Must run the simulation for 1 or more days"):
                parse_params()
    
    @pytest.mark.test_sim
    def test_test_samples_validation(self, base_test_args):
        """Test samples parameter validation"""
        args = base_test_args[:-4] + [
            '--n_samples', '0'
        ]
        with patch.object(sys, 'argv', args):
            with pytest.raises(AssertionError, match="Number of samples must be non-negative integer"):
                parse_params()
    
    @pytest.mark.test_sim
    def test_test_with_gurobi_params(self, base_test_args):
        """Test simulator with Gurobi parameters"""
        args = base_test_args[:-5] + [
            'gurobi',
            '--gurobi_param', '0.84',
            '--cpu_cores', '4'
        ]
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['gurobi_param'] == [0.84]
            assert args_dict['cpu_cores'] == 4


class TestGUICommand:
    """Test suite for GUI command"""
    
    @pytest.mark.gui
    def test_gui_basic(self, base_gui_args):
        """Test basic GUI command"""
        with patch.object(sys, 'argv', base_gui_args):
            command, args_dict = parse_params()
            assert command == 'gui'
            assert args_dict['app_style'] == 'fusion'
    
    @pytest.mark.gui
    def test_gui_with_style(self, base_gui_args):
        """Test GUI with custom style"""
        args = base_gui_args + ['--app_style', 'Windows']
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['app_style'] == 'windows'  # LowercaseAction
    
    @pytest.mark.gui
    def test_gui_test_mode(self, base_gui_args):
        """Test GUI test mode"""
        args = base_gui_args + ['--test_only']
        with patch.object(sys, 'argv', args):
            command, args_dict = parse_params()
            assert args_dict['test_only'] is True
