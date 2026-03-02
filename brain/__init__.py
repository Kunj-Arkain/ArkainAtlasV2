"""
engine.brain — Cognitive Architecture (v3.3.1 Truth Accretion Engine)
======================================================================
This package contains both:
  - Legacy modules (tools.py, adapter.py, learning.py) for backward compat
  - v3 Truth Accretion Engine (orchestrator, active inference, correlated MC, etc.)

Usage (v3):
    from engine.brain.orchestrator_v3 import TruthAccretionOrchestrator
    from engine.brain.tools_unified import create_tool_registry

Usage (legacy, still works):
    from engine.brain.tools import ToolRegistry
    from engine.brain.agents import AGENT_ROLES
"""

__version__ = "3.3.1"
