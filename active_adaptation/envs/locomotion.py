import os
import json
import torch
from isaaclab.utils import configclass

import active_adaptation
from active_adaptation.envs.base import _Env

class SimpleEnv(_Env):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = self.scene.articulations["robot"]
        
        if self.backend == "isaac" and self.sim.has_gui():
            from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController
            from isaaclab.envs import ViewerCfg
            # hacks to make IsaacLab happy. we don't use them.
            self.lookat_env_i = (
                self.scene._default_env_origins.cpu() 
                - torch.tensor(self.cfg.viewer.lookat)
            ).norm(dim=-1).argmin().item()
            self.cfg.viewer.env_index = self.lookat_env_i
            self.manager_visualizers = {}
            self.window = BaseEnvWindow(self, window_name="IsaacLab")
            self.viewport_camera_controller = ViewportCameraController(
                self,
                ViewerCfg(self.cfg.viewer.eye, self.cfg.viewer.lookat, origin_type="env")
            )

            look_at_env_id = self.lookat_env_i
            self.sim.set_camera_view(
                eye=self.scene.env_origins[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=self.scene.env_origins[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )

        self.action_buf: torch.Tensor = self.action_manager.action_buf
        self.last_action: torch.Tensor = self.action_manager.applied_action

    def setup_scene(self):
        import active_adaptation.envs.scene as scene

        if active_adaptation.get_backend() == "isaac":
            import isaaclab.sim as sim_utils
            from isaaclab.scene import InteractiveSceneCfg
            from isaaclab.assets import AssetBaseCfg, ArticulationCfg
            from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
            from active_adaptation.assets import ROBOTS, OBJECTS, get_asset_meta
            from active_adaptation.envs.terrain import TERRAINS
            
            env_spacing = self.cfg.viewer.get("env_spacing", 2.0)
            scene_cfg = InteractiveSceneCfg(num_envs=self.cfg.num_envs, env_spacing=env_spacing, replicate_physics=False)
            scene_cfg.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
            scene_cfg.robot: ArticulationCfg = ROBOTS[self.cfg.robot.name]
            
            if hasattr(self.cfg.robot, 'override_params'):
                from active_adaptation.utils import update_class_from_dict
                update_class_from_dict(scene_cfg.robot, self.cfg.robot.override_params, _ns="")
            
            scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
            robot_type = self.cfg.robot.get("robot_type", self.cfg.robot.name)
            scene_cfg.robot.spawn.usd_path = scene_cfg.robot.spawn.usd_path.format(ROBOT_TYPE=robot_type)

            # if self.cfg.command._target_ == "active_adaptation.envs.mdp.commands.hdmi.command.RobotObjectTracking":
            if "object_asset_name" in self.cfg.command:
                extra_object_names = self.cfg.command.get("extra_object_names", [])
                for extra_obj_name in extra_object_names:
                    extra_obj_cfg = OBJECTS[extra_obj_name]
                    extra_obj_cfg.prim_path = "{ENV_REGEX_NS}/" + extra_obj_name
                    setattr(scene_cfg, extra_obj_name, extra_obj_cfg)

                obj_name = self.cfg.command.object_asset_name
                obj_contact_body_name = self.cfg.command.object_body_name

                obj_cfg = OBJECTS[obj_name]
                obj_cfg.prim_path = "{ENV_REGEX_NS}/" + obj_name
                obj_type = self.cfg.command.get("object_type", obj_name)
                obj_cfg.spawn.usd_path = obj_cfg.spawn.usd_path.format(OBJECT_TYPE=obj_type)
                print(f"Using object type {obj_type} with asset {obj_cfg.spawn.usd_path}")
                setattr(scene_cfg, obj_name, obj_cfg)

                # add contact sensor to the box
                eef_names = self.cfg.command.get("contact_eef_body_name", [])
                contact_geom_prim_path = "{ENV_REGEX_NS}/" + obj_name + "/" + obj_contact_body_name

                for eef_name in eef_names:
                    contact_sensor_name = f"{eef_name}_{obj_name}_contact_forces"
                    eef_prim_path = "{ENV_REGEX_NS}/Robot/" + eef_name
                    setattr(scene_cfg, contact_sensor_name, ContactSensorCfg(
                        prim_path=eef_prim_path,
                        history_length=0,
                        track_air_time=False,
                        filter_prim_paths_expr=[contact_geom_prim_path],
                    ))
                    
            body_scale_rand = self.cfg.randomization.get("body_scale", None)
            if body_scale_rand is not None:
                from active_adaptation.assets.spawn import clone
                asset = getattr(scene_cfg, body_scale_rand.name)
                spawn_func = asset.spawn.func.__wrapped__
                asset.spawn.func = clone(spawn_func)
                asset.spawn.scale_range = tuple(body_scale_rand.scale_range)
                asset.spawn.homogeneous_scale = body_scale_rand.get("homogeneous_scale", False)
                print(f"Randomized {body_scale_rand.name} scale to {asset.spawn.scale_range}")

            scene_cfg.terrain = TERRAINS[self.cfg.terrain]
            scene_cfg.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*(ankle_roll|wrist_.*)_link", 
                history_length=3,
                track_air_time=True
            )

            tiled_camera_cfg = self._create_tiled_camera_cfg(sim_utils, TiledCameraCfg)
            if tiled_camera_cfg is not None:
                scene_cfg.tiled_camera = tiled_camera_cfg
            
            sim_cfg = sim_utils.SimulationCfg(
                dt=self.cfg.sim.isaac_physics_dt,
                render=sim_utils.RenderCfg(
                    rendering_mode="quality",
                    # antialiasing_mode="FXAA",
                    # enable_global_illumination=True,
                    # enable_reflections=True,
                ),
                device=f"cuda:{active_adaptation.get_local_rank()}"
            )
            
            # slightly reduces GPU memory usage
            sim_cfg.physx.gpu_found_lost_pairs_capacity = 2538320 # 2**20
            sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 61999079 # 2**26
            sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23
            sim_cfg.physx.enable_stabilization = False
            
            self.sim, self.scene = scene.create_isaaclab_sim_and_scene(sim_cfg, scene_cfg)

            # set camera view for "/OmniverseKit_Persp" camera
            self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
            try:
                import omni.replicator.core as rep
                # create render product
                self._render_product = rep.create.render_product(
                    "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
                )
                
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            except ModuleNotFoundError as e:
                print("Set app.enable_cameras=true to use cameras.")
            
            try:
                from active_adaptation.utils.debug import DebugDraw
                self.debug_draw = DebugDraw()
                print("[INFO] Debug Draw API enabled.")
            except ModuleNotFoundError:
                print()
            
            asset_meta = get_asset_meta(self.scene["robot"])
            path = os.path.join(os.getcwd(), "asset_meta.json")
            print(f"Saving asset meta to {path}")
            with open(path, "w") as f:
                json.dump(asset_meta, f, indent=4)
        
        else:
            from active_adaptation.envs.mujoco import MJScene, MJSim
            from active_adaptation.assets_mjcf import ROBOTS

            @configclass
            class SceneCfg:
                robot = ROBOTS[self.cfg.robot.name]
                contact_forces = "robot"
            
            self.scene = MJScene(SceneCfg())
            self.sim = MJSim(self.scene)

        
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if init_root_state is not None and not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.

    def render(self, mode: str="human"):
        # look_at_env_id = self.lookat_env_i
        # self.sim.set_camera_view(
        #     eye=self.robot.data.root_pos_w[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.eye),
        #     target=self.robot.data.root_pos_w[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.lookat)
        # )
        return super().render(mode)

    def _create_tiled_camera_cfg(self, sim_utils, TiledCameraCfg):
        return None


class PerceptualSimpleEnv(SimpleEnv):
    """SimpleEnv variant that always mounts a tiled RGB-D camera on the humanoid head."""

    def __init__(self, cfg):
        cfg.enable_cameras = True
        super().__init__(cfg)
    
    @property
    def need_render(self) -> bool: return True

    def _camera_prim_path(self) -> str:
        return "/World/envs/env_.*/Robot/head_link/perceptual_camera"

    def _camera_offset(self):
        # Place the camera slightly forward and above the head so it looks ahead.
        return (0.18, 0.0, 0.08), (0.5, -0.5, 0.5, -0.5)

    def _create_tiled_camera_cfg(self, sim_utils, TiledCameraCfg):
        if not self.cfg.get("enable_cameras", False):
            return None

        camera_spawn_cfg = sim_utils.PinholeCameraCfg(
            focal_length=7.6,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
        )

        pos_offset, rot_offset = self._camera_offset()
        return TiledCameraCfg(
            prim_path=self._camera_prim_path(),
            spawn=camera_spawn_cfg,
            offset=TiledCameraCfg.OffsetCfg(
                pos=pos_offset,
                rot=rot_offset,
                convention="ros"
            ),
            data_types=["rgb", "depth", "distance_to_image_plane"],
            update_latest_camera_pose=True,
            update_period=0.02,
            width=self.cfg.camera_width,
            height=self.cfg.camera_height,
        )
