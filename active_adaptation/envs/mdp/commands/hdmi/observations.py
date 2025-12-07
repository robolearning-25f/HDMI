from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking, RobotObjectTracking
from active_adaptation.envs.mdp.base import Observation as BaseObservation

import torch
from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    quat_apply,
    matrix_from_quat,
    yaw_quat,
    wrap_to_pi
)
from active_adaptation.utils.math import batchify
quat_apply_inverse = batchify(quat_apply_inverse)

RobotTrackObservation = BaseObservation[RobotTracking]

class ref_joint_pos_future(RobotTrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_pos_future_.view(self.num_envs, -1)

class ref_joint_vel_future(RobotTrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_vel_future_.view(self.num_envs, -1)

class ref_joint_pos_action(RobotTrackObservation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        action_manager = self.env.action_manager
        action_joint_names = action_manager.joint_names
        self.action_indices_motion = [self.command_manager.dataset.joint_names.index(joint_name) for joint_name in action_joint_names]

    def compute(self):
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[:, self.action_indices_motion]
        return ref_joint_pos

class ref_joint_pos_action_policy(RobotTrackObservation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        action_manager = self.env.action_manager
        action_joint_names = action_manager.joint_names
        self.action_indices_motion = [self.command_manager.dataset.joint_names.index(joint_name) for joint_name in action_joint_names]

        self.action_scaling = action_manager.action_scaling
        self.default_joint_pos = action_manager.default_joint_pos[:, action_manager.joint_ids]

    def compute(self):
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[:, self.action_indices_motion]
        ref_joint_action = (ref_joint_pos - self.default_joint_pos) / self.action_scaling
        return ref_joint_action

class ref_root_pos_future_b(RobotTrackObservation):
    """
    Reference root position in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_pos_future_b = torch.zeros(self.num_envs, num_future_steps, 3, device=self.device)

    def update(self):
        ref_root_pos_future_w = self.command_manager.ref_root_pos_future_w # shape: [num_envs, num_future_steps, 3]
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :] # shape: [num_envs, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]
        
        ref_root_pos_future_b = quat_apply_inverse(robot_root_quat_w, ref_root_pos_future_w - robot_root_pos_w)
        self.ref_root_pos_future_b = ref_root_pos_future_b

    def compute(self):
        return self.ref_root_pos_future_b.view(self.num_envs, -1)
    
class ref_root_ori_future_b(RobotTrackObservation):
    """
    Reference root orientation in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_ori_future_b = torch.zeros(self.num_envs, num_future_steps, 2, 3, device=self.device)

    def update(self):
        ref_root_quat_future_w = self.command_manager.ref_root_quat_future_w # shape: [num_envs, num_future_steps, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]
        
        ref_root_quat_future_b = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(ref_root_quat_future_w),
            ref_root_quat_future_w
        )
        ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
        self.ref_root_ori_future_b = ref_root_ori_future_b[:, :, :2, :]

    def compute(self):
        return self.ref_root_ori_future_b.reshape(self.num_envs, -1)

class ref_body_pos_future_local(RobotTrackObservation):
    """
    Reference body position in motion root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_pos_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        ref_body_pos_future_w = self.command_manager.ref_body_pos_future_w    # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_pos_w = self.command_manager.ref_root_pos_w[:, None, None, :].clone() # shape: [num_envs, 1, 1, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        ref_root_pos_w[..., 2] = 0.0
        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_pos_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
        self.ref_body_pos_future_local = ref_body_pos_future_local
    
    def compute(self):
        return self.ref_body_pos_future_local.view(self.num_envs, -1)

class ref_body_ori_future_local(RobotTrackObservation):
    """
    Reference body orientation in motion root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_ori_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, 3, device=self.device)
    
    def update(self):
        ref_body_quat_future_w = self.command_manager.ref_body_quat_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_root_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)
    
    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :].reshape(self.num_envs, -1)

class diff_body_pos_future_local(RobotTrackObservation):
    """
    Reference body position in each motion root frame - Robot body position in robot root frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_pos_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)

    def update(self):
        ref_body_pos_future_w = self.command_manager.ref_body_pos_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_pos_w = self.command_manager.ref_root_pos_w[:, None, None, :].clone() # shape: [num_envs, 1, 1, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]

        robot_body_pos_w = self.command_manager.robot_body_pos_w # shape: [num_envs, num_tracking_bodies, 3]
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :].clone() # shape: [num_envs, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        ref_root_pos_w[..., 2] = 0.0
        robot_root_pos_w[..., 2] = 0.0
        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_body_pos_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
        robot_body_pos_local = quat_apply_inverse(robot_root_quat_w, robot_body_pos_w - robot_root_pos_w)

        self.diff_body_pos_future_local = ref_body_pos_future_local - robot_body_pos_local.unsqueeze(1)

    def compute(self):
        return self.diff_body_pos_future_local.view(self.num_envs, -1)
    
class diff_body_lin_vel_future_local(RobotTrackObservation):
    """
    Reference body linear velocity in motion root frame - Robot body linear velocity in robot root frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_lin_vel_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        ref_body_lin_vel_future_w = self.command_manager.ref_body_lin_vel_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]
        robot_body_lin_vel_w = self.command_manager.robot_body_lin_vel_w # shape: [num_envs, num_tracking_bodies, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_body_lin_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_lin_vel_future_w)
        robot_body_lin_vel_local = quat_apply_inverse(robot_root_quat_w, robot_body_lin_vel_w)

        self.diff_body_lin_vel_future_local = ref_body_lin_vel_future_local - robot_body_lin_vel_local.unsqueeze(1)

    def compute(self):
        return self.diff_body_lin_vel_future_local.view(self.num_envs, -1)

    
class diff_body_ori_future_local(RobotTrackObservation):
    """
    Reference body orientation in motion root frame - Robot body orientation in robot root frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_ori_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, 3, device=self.device)

    def update(self):
        ref_body_quat_future_w = self.command_manager.ref_body_quat_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]
        robot_body_quat_w = self.command_manager.robot_body_quat_w # shape: [num_envs, num_tracking_bodies, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_root_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w
        )
        robot_body_quat_local = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(robot_body_quat_w),
            robot_body_quat_w
        ).unsqueeze(1)
        diff_body_quat_future = quat_mul(
            quat_conjugate(robot_body_quat_local).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_local
        )
        self.diff_body_ori_future_local = matrix_from_quat(diff_body_quat_future)

    def compute(self):
        return self.diff_body_ori_future_local[:, :, :, :2, :].reshape(self.num_envs, -1)

class diff_body_ang_vel_future_local(RobotTrackObservation):
    """
    Reference body linear velocity in motion root frame - Robot body linear velocity in robot root frame.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_ang_vel_future_local = torch.zeros(self.num_envs, self.command_manager.num_future_steps, self.command_manager.num_tracking_bodies, 3, device=self.device)
    
    def update(self):
        ref_body_ang_vel_future_w = self.command_manager.ref_body_ang_vel_future_w # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_quat_w = self.command_manager.ref_root_quat_w[:, None, None, :] # shape: [num_envs, 1, 1, 4]
        robot_body_ang_vel_w = self.command_manager.robot_body_ang_vel_w # shape: [num_envs, num_tracking_bodies, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        ref_root_quat_w = yaw_quat(ref_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_body_ang_vel_future_local = quat_apply_inverse(ref_root_quat_w, ref_body_ang_vel_future_w)
        robot_body_ang_vel_local = quat_apply_inverse(robot_root_quat_w, robot_body_ang_vel_w)

        self.diff_body_ang_vel_future_local = ref_body_ang_vel_future_local - robot_body_ang_vel_local.unsqueeze(1)

    def compute(self):
        return self.diff_body_ang_vel_future_local.view(self.num_envs, -1)

class ref_motion_phase(RobotTrackObservation):
    def compute(self):
        return (self.command_manager.t / self.command_manager.motion_len).unsqueeze(1)


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = torch.unbind(quat, dim=-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw

RobotObjectTrackObservation = BaseObservation[RobotObjectTracking]

class ref_contact_pos_b(RobotObjectTrackObservation):
    """
    Reference end-effector target position in robot root frame
    """
    def __init__(self, noise_std: float=0.0, episodic_noise_std: float=0.0, yaw_only: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std
        self.yaw_only = yaw_only
        self.ref_contact_pos_b = torch.zeros_like(self.command_manager.contact_target_pos_w)

        self.step_noise = torch.zeros_like(self.command_manager.contact_target_pos_w)
        self.episodic_noise = torch.zeros_like(self.command_manager.contact_target_pos_w)
    
    def reset(self, env_ids):
        if self.episodic_noise_std > 0.0:
            self.episodic_noise[env_ids] = torch.empty(len(env_ids), *self.command_manager.contact_target_pos_w.shape[1:], device=self.device).uniform_(-1, 1) * self.episodic_noise_std
    
    def update(self):
        if self.noise_std > 0.0:
            self.step_noise = torch.randn_like(self.command_manager.contact_target_pos_w).clamp(-3, 3) * self.noise_std
        if getattr(self.command_manager, 'valid_tracker', False):
            ref_contact_target_pos_w = self.command_manager.tracker_contact_target_pos_w
        else:
            ref_contact_target_pos_w = self.command_manager.contact_target_pos_w # shape: [num_envs, n, 3]
        
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :] # shape: [num_envs, 1, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]

        if self.yaw_only:
            robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_contact_pos_b = quat_apply_inverse(robot_root_quat_w, ref_contact_target_pos_w - robot_root_pos_w)
        if self.noise_std > 0.0:
            noise = torch.randn_like(ref_contact_pos_b).clamp(-1, 1) * self.noise_std
            ref_contact_pos_b += noise
        self.ref_contact_pos_b = ref_contact_pos_b + self.episodic_noise + self.step_noise

    def compute(self):
        return self.ref_contact_pos_b.view(self.num_envs, -1)

class diff_contact_pos_b(RobotObjectTrackObservation):
    """
    Reference end-effector target position in robot root frame - Robot end-effector position in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_contact_pos_b = torch.zeros_like(self.command_manager.contact_target_pos_w)

    def update(self):
        ref_contact_target_pos_w = self.command_manager.contact_target_pos_w # shape: [num_envs, n, 3]
        contact_eef_pos_w = self.command_manager.contact_eef_pos_w # shape: [num_envs, n, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :] # shape: [num_envs, 1, 4]
        
        diff_contact_pos_w = ref_contact_target_pos_w - contact_eef_pos_w
        self.diff_contact_pos_b = quat_apply_inverse(robot_root_quat_w, diff_contact_pos_w)

    def compute(self):
        return self.diff_contact_pos_b.view(self.num_envs, -1)
    
class object_xy_b(RobotObjectTrackObservation):
    """
    Object position in robot root frame
    """
    def __init__(self, noise_std: float=0.0, episodic_noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.object_xy_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std

        self.step_noise = torch.zeros(self.num_envs, 2, device=self.device)
        self.episodic_noise = torch.zeros(self.num_envs, 2, device=self.device)

    def reset(self, env_ids):
        if self.episodic_noise_std > 0.0:
            self.episodic_noise[env_ids] = torch.empty(len(env_ids), 2, device=self.device).uniform_(-1, 1) * self.episodic_noise_std

    def update(self):
        if self.noise_std > 0.0:
            self.step_noise = torch.randn_like(self.object_xy_b).clamp(-3, 3) * self.noise_std
        object_pos_w = self.command_manager.object.data.root_link_pos_w # shape: [num_envs, 3]
        robot_root_pos_w = self.command_manager.robot_root_pos_w # shape: [num_envs, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]
        robot_root_quat_w = yaw_quat(robot_root_quat_w)

        self.object_xy_b = quat_apply_inverse(robot_root_quat_w, object_pos_w - robot_root_pos_w)[:, :2] + self.episodic_noise + self.step_noise

    def compute(self):
        return self.object_xy_b.view(self.num_envs, -1)

class object_heading_b(RobotObjectTrackObservation):
    """
    Object orientation in robot root frame
    """
    def __init__(self, noise_std: float=0.0, episodic_noise_std: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.object_yaw_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std

        self.step_noise = torch.zeros_like(self.object_yaw_b)
        self.episodic_noise = torch.zeros_like(self.object_yaw_b)

    def reset(self, env_ids):
        if self.episodic_noise_std > 0.0:
            self.episodic_noise[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(-1, 1) * self.episodic_noise_std

    def update(self):
        if self.noise_std > 0.0:
            self.step_noise = torch.randn_like(self.object_yaw_b).clamp(-3, 3) * self.noise_std
        object_quat_w = self.command_manager.object.data.root_link_quat_w # shape: [num_envs, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]

        object_yaw_w = yaw_from_quat(object_quat_w)
        robot_root_yaw_w = yaw_from_quat(robot_root_quat_w)
        
        self.object_yaw_b = wrap_to_pi(object_yaw_w - robot_root_yaw_w)[:, None] + self.episodic_noise + self.step_noise

    def compute(self):
        object_heading_b = torch.cat([torch.cos(self.object_yaw_b), torch.sin(self.object_yaw_b)], dim=-1).view(self.num_envs, -1)
        return object_heading_b
    
    
class object_pos_b(RobotObjectTrackObservation):
    """
    Object position in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

    def update(self):
        object_pos_w = self.command_manager.object.data.root_link_pos_w # shape: [num_envs, 3]
        robot_root_pos_w = self.command_manager.robot_root_pos_w # shape: [num_envs, 3]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]

        self.object_pos_b = quat_apply_inverse(robot_root_quat_w, object_pos_w - robot_root_pos_w)

    def compute(self):
        return self.object_pos_b.view(self.num_envs, -1)

class object_ori_b(RobotObjectTrackObservation):
    """
    Object orientation in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_ori_b = torch.zeros(self.num_envs, 3, 3, device=self.device)

    def update(self):
        object_quat_w = self.command_manager.object.data.root_link_quat_w # shape: [num_envs, 4]
        robot_root_quat_w = self.command_manager.robot_root_quat_w # shape: [num_envs, 4]

        object_quat_b = quat_mul(
            quat_conjugate(robot_root_quat_w).expand_as(object_quat_w),
            object_quat_w
        )
        self.object_ori_b = matrix_from_quat(object_quat_b)

    def compute(self):
        return self.object_ori_b.view(self.num_envs, -1)
    
class object_joint_pos(RobotObjectTrackObservation):
    """
    Object joint position
    """
    def compute(self):
        return self.command_manager.object_joint_pos.unsqueeze(1)

class object_joint_vel(RobotObjectTrackObservation):
    """
    Object joint velocity
    """
    def compute(self):
        return self.command_manager.object_joint_vel.unsqueeze(1)

class object_joint_torque(RobotObjectTrackObservation):
    """
    Object joint torque
    """
    def compute(self):
        return self.command_manager.object.data.applied_torque

class object_joint_friction(RobotObjectTrackObservation):
    """ Object joint friction
    """
    def compute(self):
        return self.command_manager.object._custom_friction.unsqueeze(1)

class object_joint_damping(RobotObjectTrackObservation):
    """ Object joint damping
    """
    def compute(self):
        return self.command_manager.object._custom_damping.unsqueeze(1)

class diff_object_pos_future(RobotObjectTrackObservation):
    """
    Object position in robot root frame - Robot end-effector position in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_object_pos_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, 3, device=self.device)

    def update(self):
        ref_object_pos_future_w = self.command_manager.ref_object_pos_future_w # shape: [num_envs, num_future_steps, 3]
        object_pos_w = self.command_manager.object.data.root_link_pos_w.unsqueeze(1)
        diff_object_pos_future_w = ref_object_pos_future_w - object_pos_w

        object_quat_w = self.command_manager.object.data.root_quat_w.unsqueeze(1) # shape: [num_envs, 1, 4]
        self.diff_object_pos_future_b = quat_apply_inverse(object_quat_w, diff_object_pos_future_w)
    
    def compute(self):
        return self.diff_object_pos_future_b.view(self.num_envs, -1)

class diff_object_ori_future(RobotObjectTrackObservation):
    """
    Object orientation in robot root frame - Robot end-effector orientation in robot root frame
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_object_ori_future_b = torch.zeros(self.num_envs, self.command_manager.num_future_steps, 3, 3, device=self.device)

    def update(self):
        ref_object_quat_future_w = self.command_manager.ref_object_quat_future_w # shape: [num_envs, num_future_steps, 4]
        object_quat_w = self.command_manager.object.data.root_link_quat_w.unsqueeze(1) # shape: [num_envs, 1, 4]
        
        diff_object_quat_future = quat_mul(
            quat_conjugate(object_quat_w).expand_as(ref_object_quat_future_w),
            ref_object_quat_future_w
        )
        self.diff_object_ori_future_b = matrix_from_quat(diff_object_quat_future)

    def compute(self):
        return self.diff_object_ori_future_b.view(self.num_envs, -1)

class diff_object_joint_pos_future(RobotObjectTrackObservation):
    """
    Object joint position - Robot end-effector joint position
    """
    def compute(self):
        ref_object_joint_pos_future = self.command_manager.ref_object_joint_pos_future
        object_joint_pos = self.command_manager.object_joint_pos
        diff_object_joint_pos_future = ref_object_joint_pos_future - object_joint_pos.unsqueeze(1)
        return diff_object_joint_pos_future

class ref_object_contact_future(RobotObjectTrackObservation):
    def compute(self):
        return self.command_manager.ref_object_contact_future.view(self.num_envs, -1)

import numpy as np

def two_d_to_three_d(
    uv: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    cam_quat_w: torch.Tensor,
    cam_pos_w: torch.Tensor,
) -> torch.Tensor:
    # uv: (N, 2) in pixel coordinates
    # depth: (H, W)
    # K: (3, 3)
    # cam_quat_w: (4,) world rotation of camera (w, x, y, z)
    # cam_pos_w: (3,) world position of camera

    device = uv.device
    depth = depth.to(device)
    K = K.to(device)
    cam_quat_w = cam_quat_w.to(device)
    cam_pos_w = cam_pos_w.to(device)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    H, W = depth.shape[-2], depth.shape[-1]
    N = uv.shape[0]

    u = torch.clamp(torch.round(uv[:, 0]).long(), 0, W - 1)
    v = torch.clamp(torch.round(uv[:, 1]).long(), 0, H - 1)

    d = depth[v, u]  # (N,)

    x_cam = (uv[:, 0] - cx) * d / fx
    y_cam = (uv[:, 1] - cy) * d / fy
    z_cam = d

    cam_pt = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N, 3)

    if cam_quat_w.dim() == 1:
        cam_quat_w = cam_quat_w.unsqueeze(0).expand(N, -1)  # (N, 4)
    if cam_pos_w.dim() == 1:
        cam_pos_w = cam_pos_w.unsqueeze(0).expand(N, -1)    # (N, 3)

    world_pt = quat_apply(cam_quat_w, cam_pt) + cam_pos_w   # (N, 3)
    return world_pt
    
from torchvision import utils
from cotracker_wrapper import SingleEnvCoTrackerWrapper

class tracker(RobotObjectTrackObservation):
    """
    Point tracking using left tiled camera
    Gets ref_contact_target_pos_w from tracker and converts to robot frame
    Similar structure to ref_contact_pos_b but specifically for vision-based tracking
    """
    def __init__(self, camera_name: str = "tiled_camera_l", noise_std: float=0.0, episodic_noise_std: float=0.0, yaw_only: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.tracker = []
        self.num_contact_point = self.command_manager.contact_target_pos_w.shape[1]
        for _ in range(self.num_envs):
            self.tracker.append(SingleEnvCoTrackerWrapper(device=self.device, max_points=self.num_contact_point))
        self.tracker_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        camera = self.env.scene.sensors[camera_name]
        self.camera_size = torch.tensor([camera.image_shape[0], camera.image_shape[1]], device=self.device)
        self.contact_point_w = torch.zeros_like(self.command_manager.contact_target_pos_w)
        self.ref_contact_pos_b = torch.zeros_like(self.command_manager.contact_target_pos_w)

        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std
        self.yaw_only = yaw_only
        self.step_noise = torch.zeros_like(self.command_manager.contact_target_pos_w)
        self.episodic_noise = torch.zeros_like(self.command_manager.contact_target_pos_w)


    def reset(self, env_ids):
        super().reset(env_ids)
        self.tracker_initialized[env_ids] = False

    def update(self):
        # ============================================================
        # 3. Vision-Based Tracking (if enabled and main process)
        # ============================================================
        # Get contact targets in world frame
        contact_w = self.command_manager.contact_target_pos_w
        self.contact_point_w = self.command_manager.contact_target_pos_w

        sensor = self.env.scene.sensors.get("tiled_camera_l", None)
        # --------------------------------------------------------
        # 3a. Get Camera Pose
        # --------------------------------------------------------
        # TODO: use the ID instead of assuming d435 is 14
        head_pos_w = self.command_manager.asset.data.body_link_pos_w[:, 14]
        head_quat_w = self.command_manager.asset.data.body_link_quat_w[:, 14]

        # Camera offset from head link
        # TODO: get the cam_off by call function instead of fixed
        cam_off_pos = torch.tensor([0., -0.05, 0.]).repeat(self.num_envs, 1).to(self.device)  # (x, y, z)
        cam_off_quat = torch.tensor([0.5, -0.5, 0.5, -0.5]).repeat(self.num_envs, 1).to(self.device)  # (w, x, y, z)

        cam_pos_w = head_pos_w + quat_apply(head_quat_w, cam_off_pos)
        cam_quat_w = quat_mul(head_quat_w, cam_off_quat)

        # --------------------------------------------------------
        # 3b. Project 3D Contact Points to 2D Image
        # --------------------------------------------------------
        # Transform contact points to camera frame
        intrinsic = sensor.data.intrinsic_matrices

        cam_pos_w_repeat = cam_pos_w[:, None].repeat(1, self.num_contact_point, 1)
        cam_quat_w_repeat = cam_quat_w[:, None].repeat(1, self.num_contact_point, 1)

        contact_cam = quat_apply_inverse(cam_quat_w_repeat, contact_w - cam_pos_w_repeat)

        zs = contact_cam[..., 2:3] + 1e-8
        xy_norm = contact_cam[..., :2] / zs  # Shape: (B, M, 2)

        # Forward project to pixel coordinates
        contact_point_2d = torch.matmul(
            xy_norm,                                      # (B, M, 2)
            intrinsic[:, :2, :2].transpose(-1, -2)        # (B, 2, 2)
        ) + intrinsic[:, :2, 2].unsqueeze(1)              # Result: (B, M, 2)                
        
        # --------------------------------------------------------
        # 3d. Initialize and Update Tracker
        # --------------------------------------------------------

        # Get RGB frame
        for env_id in range(self.num_envs):
            if not (((5 < contact_point_2d[env_id]) & (contact_point_2d[env_id] < self.camera_size)).all() and (getattr(self.env, "timestamp", 0) > 0)):
                continue
    
            input_frame = sensor.data.output["rgb"][env_id].permute(2, 0, 1).unsqueeze(0)

            # Initialize tracker if not initialized
            if not self.tracker_initialized[env_id]:
                self.tracker[env_id].reset(input_frame, contact_point_2d[env_id])
                self.tracker_initialized[env_id] = True

            # Update tracker
            tracked_points, visibility = self.tracker[env_id].update(input_frame)
            current_points = tracked_points[-1]  # [num_points, 2]
            current_visibility = visibility[-1]  # [num_points]
            

            input_frame[0, :, int(contact_point_2d[env_id, 0, 1]), int(contact_point_2d[env_id, 0, 0])] = 255.0  # mark tracked point in debug image
            input_frame[0, :, int(contact_point_2d[env_id, 1, 1]), int(contact_point_2d[env_id, 1, 0])] = 255.0  # mark tracked point in debug image
            input_frame[0, :2, int(current_points[0, 1]), int(current_points[0, 0])] = 255.0  # mark tracked point in debug image
            input_frame[0, :2, int(current_points[1, 1]), int(current_points[1, 0])] = 255.0  # mark tracked point in debug image
            utils.save_image(input_frame / 255.0, f"viz_{env_id}.png")

            # Check if all points are visible
            if not current_visibility.all():
                return

            # --------------------------------------------------------
            # 3e. Reconstruct 3D from Tracked 2D Points
            # --------------------------------------------------------
            # Get depth image
            depth = sensor.data.output["depth"][env_id, ..., 0]
            K_tensor = intrinsic[env_id]

            print(env_id, '>', current_points, contact_point_2d[env_id])

            # Reconstruct 3D points from tracked 2D points + depth
            tracked_point = two_d_to_three_d(current_points, depth, K_tensor, cam_quat_w[env_id], cam_pos_w[env_id])

            # Validate reconstruction
            if tracked_point is not None and tracked_point.shape[0] == self.command_manager.num_eefs:
                self.contact_point_w[env_id] = tracked_point
                
        
    def compute(self):
        robot_root_pos_w = self.command_manager.robot_root_pos_w[:, None, :]
        robot_root_quat_w = self.command_manager.robot_root_quat_w[:, None, :]
        if self.yaw_only:
            robot_root_quat_w = yaw_quat(robot_root_quat_w)

        ref_contact_pos_b = quat_apply_inverse(robot_root_quat_w, self.contact_point_w - robot_root_pos_w)
        if self.noise_std > 0.0:
            noise = torch.randn_like(ref_contact_pos_b).clamp(-1, 1) * self.noise_std
            ref_contact_pos_b += noise
        self.ref_contact_pos_b = ref_contact_pos_b + self.episodic_noise + self.step_noise        

        return self.ref_contact_pos_b.view(self.num_envs, -1)
