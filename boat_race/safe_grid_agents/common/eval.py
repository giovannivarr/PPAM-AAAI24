"""Agent-specific evaluation interactions."""
import numpy as np

import safe_grid_agents.common.utils.meters
from safe_grid_agents.common.utils import track_metrics
from collections import defaultdict


def default_eval(agent, env, eval_history, args):
    """Evaluate an agent (default interaction)."""
    print("#### EVAL ####")
    eval_over = False
    t = 0
    state, done = env.reset(), False
    if agent.mask:
        agent.set_last_pos('west')
        #agent.set_last_pos(agent.get_pos(state))

    next_animation = [np.copy(env.render(mode="rgb_array"))]

    while True:
        if done:
            if not eval_over:
                eval_history = track_metrics(eval_history, env, eval=True, write=False)
                state, done = env.reset(), False
            else:
                break
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        if agent.mask:
            # Check whether a new checkpoint has been reached and, if it was, update the agent's memory
            agent.update_last_checkpoint(state)

        t += 1
        eval_over = t >= args.eval_timesteps

    eval_history = track_metrics(eval_history, env, eval=True, write=True)
    eval_history["returns"].reset(reset_history=True)
    eval_history["safeties"].reset()
    eval_history["margins"].reset()
    eval_history["margins_support"].reset()
    eval_history["period"] += 1
    return eval_history


EVAL_MAP = defaultdict(lambda: default_eval, {})
