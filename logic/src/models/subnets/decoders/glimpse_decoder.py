"""
Standard Attention Decoder for constructive routing problems.

This module acts as a facade, re-exporting components from the
.glimpse sub-package for backward compatibility.
"""

from .glimpse import AttentionModelFixed, GlimpseDecoder

__all__ = ["GlimpseDecoder", "AttentionModelFixed"]
