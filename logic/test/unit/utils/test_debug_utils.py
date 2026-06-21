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

    @patch("logic.src.utils.validation.debug_utils.sys._getframe")
    @patch("logic.src.utils.validation.debug_utils.sys.settrace")
    def test_watch_global_variable(self, mock_settrace, mock_getframe):
        """Test watching a global variable when not in locals."""
        mock_frame = MagicMock()
        mock_frame.f_locals = {}
        mock_frame.f_globals = {"global_var": "hello"}
        mock_getframe.return_value = mock_frame

        callback = MagicMock()
        watch("global_var", callback=callback)
        tracer = mock_settrace.call_args[0][0]

        mock_frame.f_globals = {"global_var": "world"}
        tracer(mock_frame, "line", None)
        callback.assert_called_with("hello", "world", mock_frame)

    @patch("logic.src.utils.validation.debug_utils.sys._getframe")
    @patch("logic.src.utils.validation.debug_utils.sys.settrace")
    def test_watch_event_and_frame_checks(self, mock_settrace, mock_getframe):
        """Test that tracer ignores non-line events or mismatching frames."""
        mock_frame = MagicMock()
        mock_frame.f_locals = {"v": 1}
        mock_getframe.return_value = mock_frame

        callback = MagicMock()
        watch("v", callback=callback)
        tracer = mock_settrace.call_args[0][0]

        # Scenario A: event is not "line"
        mock_frame.f_locals = {"v": 2}
        tracer(mock_frame, "call", None)
        callback.assert_not_called()

        # Scenario B: frame is not caller_frame
        other_frame = MagicMock()
        other_frame.f_locals = {"v": 3}
        tracer(other_frame, "line", None)
        callback.assert_not_called()

    @patch("logic.src.utils.validation.debug_utils.sys._getframe")
    @patch("logic.src.utils.validation.debug_utils.sys.settrace")
    def test_watch_all_locals(self, mock_settrace, mock_getframe):
        """Test watch_all tracking of all local variables."""
        mock_frame = MagicMock()
        mock_frame.f_locals = {"a": 10, "b": "init"}
        mock_getframe.return_value = mock_frame

        from logic.src.utils.validation.debug_utils import watch_all
        callback = MagicMock()
        watch_all(callback=callback)

        assert mock_settrace.called
        tracer = mock_settrace.call_args[0][0]

        # Modify variables
        mock_frame.f_locals = {"a": 10, "b": "changed", "c": 99}
        tracer(mock_frame, "line", None)

        # callback should be called for "b" change and "c" creation
        assert callback.call_count == 2
        callback.assert_any_call("b", "init", "changed", mock_frame)
        callback.assert_any_call("c", None, 99, mock_frame)

    @patch("logic.src.utils.validation.debug_utils.sys._getframe")
    @patch("logic.src.utils.validation.debug_utils.sys.settrace")
    @patch("logic.src.utils.validation.debug_utils.traceback.extract_stack")
    def test_watch_all_default_callback(self, mock_extract, mock_settrace, mock_getframe):
        """Test default print callback for watch_all."""
        mock_frame = MagicMock()
        mock_frame.f_locals = {"a": 1}
        mock_getframe.return_value = mock_frame

        mock_stack_item = MagicMock()
        mock_stack_item.filename = "file.py"
        mock_stack_item.lineno = 20
        mock_stack_item.name = "func"
        mock_extract.return_value = [mock_stack_item, mock_stack_item]

        from logic.src.utils.validation.debug_utils import watch_all
        watch_all()
        tracer = mock_settrace.call_args[0][0]

        mock_frame.f_locals = {"a": 2}
        with patch("builtins.print") as mock_print:
            tracer(mock_frame, "line", None)
            assert mock_print.called
