# PyInstaller runtime hook: frozen apps do not run `site`, so the interactive
# builtins it installs are missing. gurobipy's compiled core references
# `help` at import time (NameError otherwise); provide it before app imports.
import builtins

if not hasattr(builtins, "help"):
    import pydoc

    builtins.help = pydoc.help
