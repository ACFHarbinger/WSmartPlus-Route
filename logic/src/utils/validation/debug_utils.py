"""
Debugging utilities for the WSmart+ Route framework.

This module provides tools for monitoring runtime state, including:
- Variable watching (tracing value changes).
"""

import sys
import traceback
from types import FrameType
from typing import Any, Callable, Optional


def watch(
    var_name: str,
    callback: Optional[Callable[[Any, Any, FrameType], None]] = None,
    *,
    frame_depth: int = 1,
) -> None:
    """
    Watch a variable by name. Prints every change with line number.
    Works with numpy arrays, lists, dicts, primitives – anything.

    Args:
        var_name (str): Name of the variable to watch.
        callback (callable, optional): Custom callback (old, new, frame) -> None.
        frame_depth (int, optional): Stack depth to watch from. Defaults to 1.

    Raises:
        NameError: If variable is not found in locals or globals.
    """
    caller_frame = sys._getframe(frame_depth)
    active_callback = callback
    if active_callback is None:

        def default_callback(old, new, frame):
            """
            Default callback for the tracer.

            Args:
                old: Old value.
                new: New value.
                frame: Current stack frame.
            """
            stack = traceback.extract_stack(frame)[:-1]
            caller = stack[-1]
            print(f"DEBUG → {var_name}: {old} → {new}")
            print(f"        at {caller.filename}:{caller.lineno} in {caller.name}")

        active_callback = default_callback

    old_value = [None]

    def tracer(frame: FrameType, event: str, arg) -> Callable:
        """
        Trace function for sys.settrace.

        Args:
            frame: Stack frame.
            event: Event type.
            arg: Argument.

        Returns:
            The tracer function.
        """
        if event != "line":
            return tracer
        if frame is not caller_frame:
            return tracer

        # safely read the variable (locals first)
        try:
            new_value = frame.f_locals[var_name]
        except KeyError:
            new_value = frame.f_globals.get(var_name)

        if new_value is not old_value[0]:
            active_callback(old_value[0], new_value, frame)
            old_value[0] = new_value
        return tracer

    try:
        initial = caller_frame.f_locals[var_name]
    except KeyError:
        try:
            initial = caller_frame.f_globals[var_name]
        except KeyError as e:
            raise NameError(f"Variable '{var_name}' not found in local or global scope") from e

    old_value[0] = initial
    sys.settrace(tracer)
    print(f"Watching variable: '{var_name}' (Initial: {initial})")


def watch_all(
    callback: Optional[Callable[[str, Any, Any, FrameType], None]] = None,
    *,
    frame_depth: int = 1,
) -> None:
    """
    Watch all local variables in the caller's frame. Prints every change with line number.

    Args:
        callback (callable, optional): Custom callback (var_name, old, new, frame) -> None.
        frame_depth (int, optional): Stack depth to watch from. Defaults to 1.
    """
    caller_frame = sys._getframe(frame_depth)
    active_callback = callback
    if active_callback is None:

        def default_callback(var_name: str, old: Any, new: Any, frame: FrameType) -> None:
            """
            Default callback for the tracer.

            Args:
                var_name: Name of the variable.
                old: Old value.
                new: New value.
                frame: Current stack frame.
            """
            stack = traceback.extract_stack(frame)[:-1]
            caller = stack[-1]
            print(f"DEBUG → {var_name}: {old} → {new}")
            print(f"        at {caller.filename}:{caller.lineno} in {caller.name}")

        active_callback = default_callback

    old_locals = dict(caller_frame.f_locals)

    def tracer(frame: FrameType, event: str, arg: Any) -> Callable:
        """
        Trace function for sys.settrace.

        Args:
            frame: Stack frame.
            event: Event type.
            arg: Argument.

        Returns:
            The tracer function.
        """
        if event != "line":
            return tracer
        if frame is not caller_frame:
            return tracer

        current_locals = frame.f_locals
        for name, new_value in current_locals.items():
            if name not in old_locals:
                active_callback(name, None, new_value, frame)
                old_locals[name] = new_value
            elif old_locals[name] is not new_value:
                active_callback(name, old_locals[name], new_value, frame)
                old_locals[name] = new_value
        return tracer

    sys.settrace(tracer)
    print("Watching all local variables.")
