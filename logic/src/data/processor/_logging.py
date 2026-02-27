"""Internal logging helper for processor events."""

import contextlib
import sys


def _log_processor_event(event_name, variable_name="data", event_type="mutate", shape=None, **kwargs):
    """Log a data processing event to the active tracking run (if any)."""
    try:
        source_line = sys._getframe(1).f_lineno
    except Exception:
        source_line = 0

    with contextlib.suppress(Exception):
        from logic.src.tracking.core.run import get_active_run

        run = get_active_run()
        if run is not None:
            metadata = {
                "event": event_name,
                "variable_name": variable_name,
                "source_file": "data/processor",
                "source_line": source_line,
            }
            metadata.update(kwargs)
            safe_meta = {}
            for k, v in metadata.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)
            run.log_dataset_event(event_type, shape=shape, metadata=safe_meta)
