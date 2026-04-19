"""
Smoke test for consolidated Joint Solvers.

Verifies that all joint selection-and-construction policies are correctly
registered, instantiable, and executable via the main RouteConstructorFactory.
"""

import os
import pytest
import numpy as np

# Set environment variable for protobuf compatibility
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from logic.src.policies.route_construction.base.factory import RouteConstructorFactory
from logic.src.policies.selection_and_construction.base.registry import JointPolicyRegistry


def test_registry_discovery():
    """Ensure all joint solvers are registered in the JointPolicyRegistry."""
    # Importing factory triggers registration
    RouteConstructorFactory.ensure_registered()
    solvers = JointPolicyRegistry.list_policies()

    assert "nds_brkga" in solvers
    assert "jsa" in solvers
    assert "jgo" in solvers


@pytest.mark.parametrize("solver_name", ["nds_brkga", "jsa", "jgo"])
def test_factory_instantiation(solver_name):
    """Ensure the main RouteConstructorFactory can instantiate the policies natively."""
    RouteConstructorFactory.ensure_registered()
    policy = RouteConstructorFactory.get_adapter(solver_name)

    # Check that it's the correct policy class (not an adapter)
    from logic.src.policies.selection_and_construction import (
        NDSBRKGAPolicy, JointSAPolicy, JointGreedyPolicy
    )
    expected_classes = {
        "nds_brkga": NDSBRKGAPolicy,
        "jsa": JointSAPolicy,
        "jgo": JointGreedyPolicy
    }
    assert isinstance(policy, expected_classes[solver_name])
    assert hasattr(policy, "execute")


if __name__ == "__main__":
    print("Discovering solvers...")
    RouteConstructorFactory.ensure_registered()
    print(f"Registered Joint Solvers: {JointPolicyRegistry.list_policies()}")

    from logic.src.policies.route_construction.base.registry import RouteConstructorRegistry
    print(f"Main Registry has matching entries: {[n for n in ['nds_brkga', 'jsa', 'jgo'] if RouteConstructorRegistry.get(n)]}")

    for name in ["nds_brkga", "jsa", "jgo"]:
        try:
            policy = RouteConstructorFactory.get_adapter(name)
            print(f"Instantiated {name} -> {type(policy).__name__}")
        except Exception as e:
            print(f"Failed to instantiate {name}: {e}")
