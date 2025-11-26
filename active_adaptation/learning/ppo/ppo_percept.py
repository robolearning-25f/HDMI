from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore
from torchrl.data import CompositeSpec, TensorSpec

from .ppo_roa import (
    PPOConfig as PPOROAConfig,
    PPOROA,
    PHASE_HANDLERS,
)
from .ppo_roa import OBJECT_KEY, DEPTH_KEY
from .common import CMD_KEY, OBS_KEY, OBS_PRIV_KEY, STEREO_KEY


def _augment_handler(base_handler, extra_key: str):
    class Handler(base_handler.__class__):
        name = base_handler.name
        required_in_keys = tuple(base_handler.required_in_keys) + (extra_key,)
        required_inference_keys = tuple(base_handler.required_inference_keys) + (extra_key,)

        def rollout_modules(self, ppo: "PPOPerceptROA"):
            return base_handler.rollout_modules(ppo)

        def rollout_modules_inference(self, ppo: "PPOPerceptROA"):
            return base_handler.rollout_modules_inference(ppo)

        def rollout_out_keys(self, ppo: "PPOPerceptROA"):
            return base_handler.rollout_out_keys(ppo)

        def train(self, ppo: "PPOPerceptROA", tensordict):
            return base_handler.train(ppo, tensordict)

    return Handler()


PERCEPT_PHASE_HANDLERS = {k: _augment_handler(v, STEREO_KEY) for k, v in PHASE_HANDLERS.items()}


@dataclass
class PPOPerceptConfig(PPOROAConfig):
    """
    PPOROA variant that carries stereo/rgb camera observations alongside the existing keys.
    The policy still ignores the stereo tensor, so legacy weights remain compatible.
    """

    _target_: str = "active_adaptation.learning.ppo.ppo_percept.PPOPerceptROA"
    name: str = "ppo_roa_percept"
    in_keys: List[str] = (CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, STEREO_KEY)


class PPOPerceptROA(PPOROA):
    """
    Same architecture/logic as PPOROA, but enforces presence of the stereo/rgb camera observation
    for both training and inference. The network inputs remain unchanged; stereo is available
    in the tensordict for downstream analysis without altering model shapes.
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
        # Temporarily zero-out stereo influence on the network by keeping the same PPOROA architecture.
        # Stereo data is still present in the tensordict for analysis.
        self.phase_handler = PERCEPT_PHASE_HANDLERS[cfg.phase]
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        self.phase_handler = PERCEPT_PHASE_HANDLERS[self.cfg.phase]


cs = ConfigStore.instance()
cs.store("ppo_roa_percept_train", node=PPOPerceptConfig(phase="train", vecnorm="train", entropy_coef_start=0.001, entropy_coef_end=0.001), group="algo")
cs.store("ppo_roa_percept_adapt", node=PPOPerceptConfig(phase="adapt", vecnorm="eval", entropy_coef_start=0.00, entropy_coef_end=0.00), group="algo")
cs.store("ppo_roa_percept_finetune", node=PPOPerceptConfig(phase="finetune", vecnorm="eval", entropy_coef_start=0.001, entropy_coef_end=0.001), group="algo")
cs.store("ppo_roa_percept_train_est", node=PPOPerceptConfig(phase="train_est", vecnorm="eval", entropy_coef_start=0.00, entropy_coef_end=0.00, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, DEPTH_KEY, STEREO_KEY)), group="algo")
cs.store("ppo_roa_percept_adapt_est", node=PPOPerceptConfig(phase="adapt_est", vecnorm="eval", entropy_coef_start=0.00, entropy_coef_end=0.00, in_keys=(CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, DEPTH_KEY, STEREO_KEY)), group="algo")
