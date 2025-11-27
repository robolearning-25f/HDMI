from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore
from torchrl.data import CompositeSpec, TensorSpec

from .ppo_roa import PPOConfig as PPOROAConfig, PPOROA, OBJECT_KEY, DEPTH_KEY
from .common import CMD_KEY, OBS_KEY, OBS_PRIV_KEY, STEREO_KEY


@dataclass
class PPOPerceptConfig(PPOROAConfig):
    """
    PPOROA variant that additionally threads stereo observations through `in_keys`.
    """

    _target_: str = "active_adaptation.learning.ppo.ppo_percept.PPOPerceptROA"
    name: str = "ppo_roa_percept"
    in_keys: List[str] = (CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, STEREO_KEY)


class PPOPerceptROA(PPOROA):
    """
    Same architecture/logic as PPOROA, but its config includes STEREO_KEY in `in_keys`.
    """

    def __init__(
        self,
        cfg: PPOPerceptConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)


cs = ConfigStore.instance()
cs.store("ppo_roa_percept_train", node=PPOPerceptConfig(phase="train", vecnorm="train", entropy_coef_start=0.001, entropy_coef_end=0.001), group="algo")
cs.store("ppo_roa_percept_adapt", node=PPOPerceptConfig(phase="adapt", vecnorm="eval", entropy_coef_start=0.00, entropy_coef_end=0.00), group="algo")
cs.store("ppo_roa_percept_finetune", node=PPOPerceptConfig(phase="finetune", vecnorm="eval", entropy_coef_start=0.001, entropy_coef_end=0.001), group="algo")
cs.store("ppo_roa_percept_train_est", node=PPOPerceptConfig(phase="train_est", vecnorm="eval", entropy_coef_start=0.00, entropy_coef_end=0.00, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, DEPTH_KEY, STEREO_KEY)), group="algo")
cs.store("ppo_roa_percept_adapt_est", node=PPOPerceptConfig(phase="adapt_est", vecnorm="eval", entropy_coef_start=0.00, entropy_coef_end=0.00, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, DEPTH_KEY, STEREO_KEY)), group="algo")
