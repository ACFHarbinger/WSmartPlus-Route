import sys
import traceback

from types import FrameType
from typing import Any, Callable, Optional


def watch(var_name: str,
          callback: Optional[Callable[[Any, Any, FrameType], None]] = None,
          *,
          frame_depth: int = 1) -> None:
    """
    Watch a variable by name. Prints every change with line number.
    Works with numpy arrays, lists, dicts, primitives – anything.
    """
    caller_frame = sys._getframe(frame_depth)
    if callback is None:
        def callback(old, new, frame):
            stack = traceback.extract_stack(frame)[:-1]
            caller = stack[-1]
            print(f"DEBUG → {var_name}: {old} → {new}")
            print(f"        at {caller.filename}:{caller.lineno} in {caller.name}")

    old_value = [None]

    def tracer(frame: FrameType, event: str, arg) -> Callable:
        if event != 'line':
            return tracer
        if frame is not caller_frame:
            return tracer

        # safely read the variable (locals first)
        try:
            new_value = frame.f_locals[var_name]
        except KeyError:
            new_value = frame.f_globals.get(var_name)

        if new_value is not old_value[0]:
            callback(old_value[0], new_value, frame)
            old_value[0] = new_value
        return tracer

    try:
        initial = caller_frame.f_locals[var_name]
    except KeyError:
        try:
            initial = caller_frame.f_globals[var_name]
        except KeyError:
            raise NameError(f"Name '{var_name}' not found in locals or globals")
    
    old_value[0] = initial
    caller_frame.f_trace = tracer
    sys.settrace(tracer)
