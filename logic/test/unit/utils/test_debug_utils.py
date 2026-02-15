from unittest.mock import MagicMock, patch
import pytest
from logic.src.utils.validation.debug_utils import watch

class TestDebugUtils:
    """Class for debug_utils tests."""

    @patch("logic.src.utils.validation.debug_utils.sys._getframe")
    @patch("logic.src.utils.validation.debug_utils.sys.settrace")
    def test_watch_variable(self, mock_settrace, mock_getframe):
        """Test enabling the watcher."""
        # Mock frame setup
        mock_frame = MagicMock()
        mock_frame.f_locals = {"my_var": 10}
        mock_getframe.return_value = mock_frame

        watch("my_var")

        assert mock_settrace.called
        tracer_func = mock_settrace.call_args[0][0]
        assert callable(tracer_func)

        # Test callback
        mock_frame.f_locals = {"my_var": 20}
        callback = MagicMock()

        # Re-initialize watch with callback
        mock_frame.f_locals = {"my_var": 10}
        watch("my_var", callback=callback)
        tracer_func_with_cb = mock_settrace.call_args[0][0]

        # Trigger tracer
        mock_frame.f_locals = {"my_var": 99}
        tracer_func_with_cb(mock_frame, "line", None)

        callback.assert_called_with(10, 99, mock_frame)

    @patch("logic.src.utils.validation.debug_utils.sys._getframe")
    @patch("logic.src.utils.validation.debug_utils.sys.settrace")
    def test_watch_not_found(self, mock_settrace, mock_getframe):
        """Test watch raises NameError if var not found."""
        mock_frame = MagicMock()
        mock_frame.f_locals = {}
        mock_frame.f_globals = {}
        mock_getframe.return_value = mock_frame

        with pytest.raises(NameError):
            watch("missing_var")

    @patch("logic.src.utils.validation.debug_utils.sys._getframe")
    @patch("logic.src.utils.validation.debug_utils.sys.settrace")
    @patch("logic.src.utils.validation.debug_utils.traceback.extract_stack")
    def test_default_callback(self, mock_extract, mock_settrace, mock_getframe):
        """Test the default print callback."""
        mock_frame = MagicMock()
        mock_frame.f_locals = {"v": 1}
        mock_getframe.return_value = mock_frame

        mock_stack_item = MagicMock()
        mock_stack_item.filename = "file.py"
        mock_stack_item.lineno = 10
        mock_stack_item.name = "func"
        mock_extract.return_value = [mock_stack_item, mock_stack_item]

        watch("v")
        tracer = mock_settrace.call_args[0][0]

        mock_frame.f_locals = {"v": 2}

        with patch("builtins.print") as mock_print:
            tracer(mock_frame, "line", None)
            assert mock_print.called
