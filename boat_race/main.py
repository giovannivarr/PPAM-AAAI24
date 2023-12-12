import os
import random
import subprocess


from safe_grid_agents.parsing import prepare_parser
from train import train


parser = prepare_parser()
args = parser.parse_args()

if args.seed is None:
    args.seed = random.randrange(500)

######## Logging into TensorboardX ########
# If `args.log_dir` is None, we attempt a default unique up to env, agent, cheating, and seed combinations.
if args.log_dir is None:
    cheating = "baseline" if args.cheat else "corrupt"
    masked = "mask" if args.mask else "no-mask"
    args.log_dir = os.path.join(
        "runs", args.env_alias, args.agent_alias, cheating, masked, str(args.seed)
    )
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir, exist_ok=True)

train(args)