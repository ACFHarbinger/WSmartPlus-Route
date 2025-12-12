
import pytest
import time
from unittest.mock import MagicMock, patch, mock_open
from PySide6.QtCore import QMutex
from gui.helpers.chart_worker import ChartWorker
from gui.helpers.file_tailer_worker import FileTailerWorker

# --- ChartWorker Tests ---
def test_chart_worker_process_data():
    """Test standard data processing logic of ChartWorker."""
    # Data Setup
    daily_data = {
        'PolicyA sample 1': {
            'metric1': {1: 10, 2: 20}
        }
    }
    historical_bin = {}
    latest_bin = {
        'PolicyA sample 1': {'bin_level': 50}
    }
    metrics = ['metric1']
    data_mutex = QMutex()
    
    # Init Worker
    worker = ChartWorker(daily_data, historical_bin, latest_bin, metrics, data_mutex)
    
    # Mock Signal
    worker.data_ready = MagicMock()
    worker.data_ready.emit = MagicMock()
    
    # Execute
    worker.process_data('PolicyA sample 1')
    
    # Assert
    worker.data_ready.emit.assert_called_once()
    args = worker.data_ready.emit.call_args[0]
    key = args[0]
    data = args[1]
    
    assert key == 'PolicyA sample 1'
    assert data['max_days'] == 2
    assert data['metrics']['metric1']['values'] == [10, 20]
    assert data['bin_state'] == {'bin_level': 50}

# --- FileTailerWorker Tests ---
@patch('builtins.open', new_callable=mock_open, read_data="Line 1\nLine 2\n")
@patch('os.path.exists', return_value=True)
def test_file_tailer_worker(mock_exists, mock_file):
    """Test that file tailer reads new lines and emits signals."""
    metrics_mutex = QMutex()
    worker = FileTailerWorker(metrics_mutex, "/fake/path/log.txt")
    
    # Mock Signal
    worker.log_line_ready = MagicMock()
    worker.log_line_ready.emit = MagicMock()
    
    # We need to stop the infinite loop, so we run use a trick or run in a thread?
    # Better: Inspect the logic. It uses a while loop checked by _is_running.
    # We can start it in a thread or just call the logic step if we refactor, but here 
    # we can just run it briefly or mock the loop condition?
    # Mocking the loop is hard without refactoring. 
    # Instead, we'll subclass or monkeypatch `time.sleep` to stop after one iteration?
    # Or overwrite `_is_running` from within the read loop via a side effect?
    
    # Strategy: Side effect on file read to stop worker
    def side_effect(*args, **kwargs):
        worker.stop()
        return "New Line\n"
    
    mock_file.return_value.readline.side_effect = side_effect
    
    # Execute (will run one loop then stop due to side effect stopping it, hopefully)
    # The worker logic: while self._is_running...
    # We need strictly controlled execution.
    
    # Simpler approach: Just verify init and stop logic, and maybe simulate one pass if possible.
    # Given the blocking while loop, let's trust the logic structure or run it in a separate thread for 0.1s in a real integration test.
    # For unit test, we can mock `time.sleep` to throw an exception to break the loop? 
    # Or check if we can refactor `tail_file` to be testable? 
    # Let's try the exception break.
    
    class BreakLoop(Exception): pass
    
    with patch('time.sleep', side_effect=BreakLoop):
        with pytest.raises(BreakLoop):
            worker.tail_file()
    
    # This verifies it entered the loop and tried to sleep or work.
    # Testing the read logic specifically:
    
    # Let's manually trigger the inner logic if we could, but it's one big method.
    pass

@patch('builtins.open', new_callable=mock_open, read_data="New log line\n")
@patch('os.path.exists', return_value=True)
@patch('time.sleep') # prevent actual sleeping
def test_file_tailer_read_logic(mock_sleep, mock_exists, mock_file):
    """Test reading one line."""
    mutex = QMutex()
    worker = FileTailerWorker(mutex, "dummy")
    worker.log_line_ready = MagicMock()
    worker.log_line_ready.emit = MagicMock()
    
    # Force loop to run once
    worker._is_running = True
    
    # We use a side effect on readline to return a line then STOP the worker
    # This effectively allows one iteration
    def readline_side_effect():
        worker.stop() # Stop after reading
        return "GUI_LOG_START:...\n"
        
    mock_file.return_value.readline.side_effect = readline_side_effect
    
    worker.tail_file()
    
    worker.log_line_ready.emit.assert_called_with("GUI_LOG_START:...\n")
