from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

config = config.get_config("pi0_aloha_sim")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base")

# Create a trained policy.
# policy = policy_config.create_trained_policy(config, checkpoint_dir)