"""Agent-specific warmup interactions."""
from collections import defaultdict

from safe_grid_agents.common.agents import RandomAgent

def noop_warmup(agent, env, history, args):
    """Warm up with noop."""
    return agent, env, history, args


WARMUP_MAP = defaultdict(
    lambda: noop_warmup
)
