import os
import json
import torch

import active_adaptation
from active_adaptation.envs.base import _Env

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from active_adaptation.assets import ROBOTS, OBJECTS, get_asset_meta
from active_adaptation.envs.terrain import TERRAINS
from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController
from isaaclab.envs import ViewerCfg
    

class SimpleEnv(_Env):
    def __init__(self, cfg, enable_debug_draw: bool=True):        
        super().__init__(cfg, enable_debug_draw=enable_debug_draw)
        self.robot = self.scene.articulations["robot"]
        
        if self.sim.has_gui():
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
            
        env_spacing = self.cfg.viewer.get("env_spacing", 2.0)
        scene_cfg = InteractiveSceneCfg(num_envs=self.cfg.num_envs, env_spacing=env_spacing, replicate_physics=False)
        scene_cfg.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        robot: ArticulationCfg = ROBOTS[self.cfg.robot.name]
        scene_cfg.robot = robot
        
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

        tiled_camera_cfg = self._create_tiled_camera_cfg(sim_utils)
        if tiled_camera_cfg is not None:
            scene_cfg.tiled_camera_l = tiled_camera_cfg[0]
            scene_cfg.tiled_camera_r = tiled_camera_cfg[1]
        
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
        
        if self.enable_debug_draw:
            try:
                from active_adaptation.utils.debug import DebugDraw
                self.debug_draw = DebugDraw()
                print("[INFO] Debug Draw API enabled.")
            except ModuleNotFoundError:
                print()
        else:
            print("[INFO] Debug Draw API disabled.")
        
        asset_meta = get_asset_meta(self.scene["robot"])
        path = os.path.join(os.getcwd(), "asset_meta.json")
        print(f"Saving asset meta to {path}")
        with open(path, "w") as f:
            json.dump(asset_meta, f, indent=4)

    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if init_root_state is not None and not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.

    def render(self, mode: str="human"):
        return super().render(mode)

    def _create_tiled_camera_cfg(self, sim_utils) -> tuple[TiledCameraCfg, TiledCameraCfg] | None:
        return None


class PerceptualSimpleEnv(SimpleEnv):
    """SimpleEnv variant that always mounts a tiled RGB-D camera on the humanoid head."""

    def __init__(self, cfg, enable_debug_draw: bool=True):
        cfg.enable_cameras = True
        super().__init__(cfg, enable_debug_draw=enable_debug_draw)
    
    @property
    def need_render(self) -> bool: return True

    def _camera_prim_path_left(self) -> str:
        return "{ENV_REGEX_NS}/Robot/d435_link/rgbd_l"
    
    def _camera_prim_path_right(self) -> str:
        return "{ENV_REGEX_NS}/Robot/d435_link/rgbd_r"

    def _camera_offset_left(self):
        # Place the camera slightly forward and above the head so it looks ahead.
        return (0.18, -0.05, 0.08), (0.5, -0.5, 0.5, -0.5)

    def _camera_offset_right(self):
        return (0.18, 0.05, 0.08), (0.5, -0.5, 0.5, -0.5)

    def _create_tiled_camera_cfg(self, sim_utils) -> tuple[TiledCameraCfg, TiledCameraCfg] | None:
        if not self.cfg.get("enable_cameras", False):
            return None

        camera_spawn_cfg = sim_utils.PinholeCameraCfg(
            focal_length=7.6,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
        )
        # camera_spawn_cfg = sim_utils.PinholeCameraCfg(
        #     focal_length=7.6,
        #     focus_distance=1_000.0,
        #     horizontal_aperture=10.5,
        #     clipping_range=(0.1, 1.0e5),
        # )

        pos_offset_l, rot_offset_l = self._camera_offset_left()
        left_camera_cfg = TiledCameraCfg(
            prim_path=self._camera_prim_path_left(),
            spawn=camera_spawn_cfg,
            offset=TiledCameraCfg.OffsetCfg(
                pos=pos_offset_l,
                rot=rot_offset_l,
                convention="ros"
            ),
            data_types=["rgb", "depth", "distance_to_image_plane"],
            update_latest_camera_pose=True,
            update_period=0.02,
            width=self.cfg.camera_width,
            height=self.cfg.camera_height,
        )
        
        pos_offset_r, rot_offset_r = self._camera_offset_right()
        right_camera_cfg = TiledCameraCfg(
            prim_path=self._camera_prim_path_right(),
            spawn=camera_spawn_cfg,
            offset=TiledCameraCfg.OffsetCfg(
                pos=pos_offset_r,
                rot=rot_offset_r,
                convention="ros"
            ),
            data_types=["rgb", "depth", "distance_to_image_plane"],
            update_latest_camera_pose=True,
            update_period=0.02,
            width=self.cfg.camera_width,
            height=self.cfg.camera_height,
        )

        return (left_camera_cfg, right_camera_cfg)
