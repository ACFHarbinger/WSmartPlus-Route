"""
Route Constructor Base Package.

This package provides adapter implementations for various routing
optimization policies (HGS, ALNS, VRPP, etc.) and a factory pattern
for dynamic route constructor instantiation.
"""

from .factory import IRouteConstructor, RouteConstructorFactory, RouteConstructorRegistry
