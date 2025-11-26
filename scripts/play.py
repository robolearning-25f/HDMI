import torch
import hydra
import rerun as rr
from omegaconf import OmegaConf
from isaaclab.app import AppLauncher
from torchrl.envs.utils import set_exploration_type, ExplorationType
from scripts.helpers import make_env_policy


@hydra.main(config_path="../cfg", config_name="play", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    rr.init("HDMI Humanoid Policy", spawn=False)
    rr.connect_grpc()
    
    app_launcher = AppLauncher(cfg.app)
    simulation_app = app_launcher.app

    env, policy, _ = make_env_policy(cfg)
    
    rollout_policy = (
        policy.get_inference_policy()
        if hasattr(policy, "get_inference_policy")
        else policy.get_rollout_policy("eval")
    )
    
    env.base_env.eval()
    td_ = env.reset()
    assert not env.base_env.training
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        torch.compiler.cudagraph_mark_step_begin()
        
        while True:
            td_ = rollout_policy(td_)
            _, td_ = env.step_and_maybe_reset(td_)
    
    env.close()
    simulation_app.close()

if __name__ == "__main__": main()
