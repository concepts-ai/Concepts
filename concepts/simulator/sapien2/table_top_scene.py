import os
import os.path as osp
import time
from typing import Optional, List, Union, NamedTuple

import numpy as np
import open3d as o3d
from PIL import Image

import mplib
import mplib.pymp as pymp
from mplib.pymp.collision_detection import fcl
from sapien.core import Actor, Joint, LinkBase
from sapien.core import CameraEntity, Pose, URDFLoader, Articulation
from sapien.core import PhysicalMaterial, Drive
from sapien.utils import Viewer

from concepts.math.rotationlib_wxyz import euler2quat as rpy2wxyz, quat_mul as quat_mul_wxyz, wxyz2xyzw, rotate_vector_wxyz
from concepts.simulator.sapien2.camera import get_depth_img, get_rgba_img
from concepts.simulator.sapien2.mesh_utils import get_actor_mesh
from concepts.simulator.sapien2.scene_base import SapienSceneBase
from concepts.simulator.sapien2.sapien_utils import create_box



PANDA_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)),
    '../../assets/robots/franka_description/robots/panda'
)


class PandaRobotSapien(object):
    # TODO(Yuyao @ 2024-0903): in the future, create RobotBaseSapien and inherit from it
    def __init__(
            self,
            scene: SapienSceneBase,
            root_pose: Pose = Pose(),
            init_qpos: np.ndarray = np.array([-0.45105, -0.38886, 0.45533, -2.19163, 0.13169, 1.81720, 0.51563, 0, 0]),
            stiffness: float = 1000,
            damping: float = 200,
            config: dict = {},
    ):
        loader = scene.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.articulation: Articulation = loader.load(osp.join(PANDA_PATH, 'panda.urdf'), config=config)
        self.active_joints: List[Joint] = self.articulation.get_active_joints()
        self.end_effector: LinkBase = self.articulation.get_links()[-3]
        self.mp_planner: Optional[mplib.Planner] = None
        self.attach_drive: Optional[Drive] = None
        self.articulation.set_qpos(init_qpos)
        for i, joint in enumerate(self.active_joints):
            joint.set_drive_property(stiffness=stiffness, damping=damping)
            joint.set_drive_target(init_qpos[i])
        self.articulation.set_root_pose(root_pose)

    def set_drive_target(self, qpos, vel_qpos=None):
        for i in range(7):
            self.active_joints[i].set_drive_target(qpos[i])
            if vel_qpos is not None:
                self.active_joints[i].set_drive_velocity_target(vel_qpos[i])

    def set_up_planner(self):
        link_names = [link.get_name() for link in self.articulation.get_links()]
        joint_names = [joint.get_name() for joint in self.articulation.get_active_joints()]
        floor = fcl.Box([2, 2, 0.2])  # create a 2 x 2 x 0.1m box
        # create a collision object for the floor, with a 10cm offset in the z direction
        floor_fcl_collision_object = fcl.CollisionObject(floor, pymp.Pose())
        floor_fcl_object = fcl.FCLObject('floor', pymp.Pose([0, 0, -0.1], [1, 0, 0, 0]),
                                         [floor_fcl_collision_object], [pymp.Pose()])
        self.planner = mplib.Planner(
            urdf=osp.join(PANDA_PATH, 'panda.urdf'),
            srdf=osp.join(PANDA_PATH, 'panda.srdf'),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            # joint_vel_limits=np.ones(7) * 0.5,
            joint_vel_limits=np.ones(7) * 0.1,
            joint_acc_limits=np.ones(7) * 0.1,
            objects=[floor_fcl_object]
        )
        # NOTE: this doesn't work, working around by recomputing the pose when calling planner
        # planner.set_base_pose(pymp.Pose(robot.get_root_pose().p, robot.get_root_pose().q))




class TableTopScene(SapienSceneBase):
    def __init__(
        self,
        fps: float = 240.0,
    ):
        super().__init__(fps)
        self.ground_base = self.scene.add_ground(
            altitude=-0.1,
            render_half_size=[10, 10, 0.1],
            material=self.scene.create_physical_material(0.5, 0.5, 0.1)
        )
        self.plane = create_box(self.scene, Pose([1, 0, -0.05]), [0.8, 1, 0.05], color=[0.9,0.9,0.9,1])

        self.robot: Optional[PandaRobotSapien] = None
        self.robots: List[PandaRobotSapien] = []

        self.is_hiding_env_visual = False

    def set_up_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        floor = fcl.Box([2, 2, 0.2])  # create a 2 x 2 x 0.1m box
        # create a collision object for the floor, with a 10cm offset in the z direction
        floor_fcl_collision_object = fcl.CollisionObject(floor, pymp.Pose())
        # a very small offset of 0.0001 is used to prevent the collision between link0 and the floor
        floor_fcl_object = fcl.FCLObject('floor', pymp.Pose([0, 0, -0.1001], [1, 0, 0, 0]), [floor_fcl_collision_object], [pymp.Pose()])
        self.planner = mplib.Planner(
            urdf="assets/panda/panda.urdf",
            srdf="assets/panda/panda.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            # joint_vel_limits=np.ones(7) * 0.5,
            joint_vel_limits=np.ones(7) * 0.1,
            joint_acc_limits=np.ones(7) * 0.1,
            objects=[floor_fcl_object]
        )

    def update_env_pcd(self, exclude_ids: list[int] = None, pcd_resolution = 1e-3, verbose=False, idx: int = None):
        """Update the point cloud of the environment for planner collision avoidance"""
        all_actor_ids = [
            actor.get_id() for actor in self.scene.get_all_actors() if len(actor.get_collision_shapes()) > 0
        ]
        if self.robot is None:
            if idx is None:
                robot_actor_ids = [link.get_id() for link in self.robot_1.get_links()] + [link.get_id() for link in self.robot_2.get_links()]
            else:
                if idx == 1:
                    robot_actor_ids = [link.get_id() for link in self.robot_1.get_links()]
                    all_actor_ids += [link.get_id() for link in self.robot_2.get_links()]
                else:
                    robot_actor_ids = [link.get_id() for link in self.robot_2.get_links()]
                    all_actor_ids += [link.get_id() for link in self.robot_1.get_links()]
        else:
            robot_actor_ids = [link.get_id() for link in self.robot.get_links()]
        ground_actor_id = self.ground.get_id()  # ground collision avoidance is handled in the planner
        excluded_actor_ids = robot_actor_ids + [ground_actor_id]
        if exclude_ids is not None:
            excluded_actor_ids += exclude_ids
        all_object_ids = list(set(all_actor_ids) - set(excluded_actor_ids))
        pcds = []
        for object_id in all_object_ids:
            if object_id != self.plane.get_id():
                current_actor = self.scene.find_actor_by_id(object_id)
                if current_actor is None:
                    current_actor = self.scene.find_articulation_link_by_link_id(object_id)
                if current_actor is not None:
                    try:
                        pcds.append(get_actor_pcd(current_actor, 100000))
                    except AttributeError:
                        print(f'Actor id {object_id} does not have a collision mesh')
            else:
                x = np.linspace(0.2, 0.8, 1000)
                y = np.linspace(-0.4, 0.4, 1000)
                x, y = np.meshgrid(x, y)
                x = x.flatten()
                y = y.flatten()
                z = np.zeros_like(x)
                pcds.append(np.stack([x, y, z], axis=-1))

        if len(pcds) == 0:
            pcd = np.array([[1e6, 1e6, 1e6]])  # empty point cloud
        else:
            pcd = np.concatenate(pcds, axis=0)

        if verbose:
            # visualize the pcd
            print(pcd.shape)
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
            o3d.visualization.draw_geometries([o3d_pcd])

        if self.planner is None:
            # transform the pcd to the robots' coordinate
            pcd1 = (self.robot_1.get_root_pose().inv().to_transformation_matrix() @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T).T[:, :3]
            pcd2 = (self.robot_2.get_root_pose().inv().to_transformation_matrix() @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T).T[:, :3]
            self.planner_1.update_point_cloud(pcd1, resolution=pcd_resolution)
            self.planner_2.update_point_cloud(pcd2, resolution=pcd_resolution)
        else:
            self.planner.update_point_cloud(pcd, resolution=pcd_resolution)

        return pcd # return the point cloud for debugging

    def grasp_center_ik(self, grasp_center: np.ndarray, ee_quat_wxyz: np.ndarray, start_qpos: np.ndarray, mask: list = None, threshold: float = 1e-3, exclude_ids: Optional[List[int]] = None)\
            -> tuple[Union[np.ndarray, None], np.ndarray]:
        if self.robot is None:
            raise ValueError("No robot in the scene")
        pos_delta = np.array([0, 0, 0.1])
        hand_pos = grasp_center - rotate_vector_wxyz(pos_delta, ee_quat_wxyz)
        return self.ee_ik(Pose(p=hand_pos, q=ee_quat_wxyz), start_qpos, mask, threshold, exclude_ids), hand_pos

    def ee_ik(
            self,
            ee_pose: Pose,
            start_qpos: np.ndarray,
            mask: list = None,
            threshold: float = 1e-3,
            exclude_ids: Optional[List[int]] = None,
            return_closest: bool = False,
            verbose: bool = False,
            pcd_resolution = 1e-3
    ) -> Union[np.ndarray, None]:
        if self.robot is None:
            raise ValueError("No robot in the scene")
        self.update_env_pcd(exclude_ids, pcd_resolution=pcd_resolution)
        if mask is None:
            mask = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        status, qpos = self.planner.IK(
            goal_pose=pymp.Pose(ee_pose.p, ee_pose.q),
            start_qpos=start_qpos,
            mask=mask,
            threshold=threshold,
            return_closest=return_closest,
            verbose=verbose
        )
        if status == "Success":
            if return_closest:
                return qpos
            else:
                return qpos[0]
        else:
            return None

    def ee_ik_without_collision_check(
            self,
            ee_pose: Pose,
            start_qpos: np.ndarray,
            mask: list = None,
            threshold: float = 1e-3,
            return_closest: bool = False,
            verbose: bool = False,
    ) -> Union[np.ndarray, None]:
        self.planner.remove_point_cloud()
        if mask is None:
            mask = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        status, qpos = self.planner.IK(
            goal_pose=pymp.Pose(ee_pose.p, ee_pose.q),
            start_qpos=start_qpos,
            mask=mask,
            threshold=threshold,
            return_closest=return_closest,
            verbose=verbose
        )
        if status == "Success":
            if return_closest:
                return qpos
            else:
                return qpos[0]
        else:
            return None

    def follow_path(self, result, check_collision=False, collision_obj_1=None, collision_obj_2=None, threshold=1e-3, camera=None, camera_interval=4):
        n_step = result['position'].shape[0]
        collision = False
        images = []
        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
                external=False)
            self.robot.set_qf(qf)
            # for j in range(7):
            #     self.active_joints[j].set_drive_target(result['position'][i][j])
            #     self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.set_drive_target(result['position'][i], result['velocity'][i])
            self.scene.step()
            if check_collision:
                collisions = get_contacts_by_id(self.scene, collision_obj_1, collision_obj_2, threshold)
                if len(collisions) > 0:
                    collision = True
                    break
            if i % 4 == 0:
                self.scene.update_render()
                if self.viewer is not None:
                    self.viewer.render()
                if camera is not None and i % camera_interval == 0:
                    image = get_rgba_img(camera=camera)
                    images.append(image)

        if camera is not None:
            return collision, images
        else:
            return collision

    def open_gripper(self, gripper_target=0.04):
        qpos = self.robot.get_qpos()
        for i, joint in enumerate(self.active_joints):
            if i < 7:
                joint.set_drive_target(qpos[i])
            else:
                joint.set_drive_target(gripper_target)

        for i in range(int(self.fps)):
            qf = self.robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
                external=False)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                if self.viewer is not None:
                    self.viewer.render()


    def close_gripper(self, gripper_target=0.01):
        qpos = self.robot.get_qpos()
        for i, joint in enumerate(self.active_joints):
            if i < 7:
                joint.set_drive_target(qpos[i])
            else:
                joint.set_drive_target(gripper_target)
        for i in range(int(self.fps)):
            qf = self.robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
                external=False)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                if self.viewer is not None:
                    self.viewer.render()

    def attach_object(self, object: Actor):
        if self.attach_drive is not None:
            raise ValueError("An object is already attached")
        self.attach_drive = self.scene.create_drive(
            self.end_effector,
            Pose(),
            object,
            object.get_pose().inv() * self.end_effector.get_pose()
        )
        self.attach_drive.lock_motion(True, True, True, True, True, True)

    def detach_object(self):
        if self.attach_drive is not None:
            self.scene.remove_drive(self.attach_drive)
            self.attach_drive = None
        else:
            print("No object attached")

    def planner_attach_obj(self, object: Actor):
        object_mesh = get_actor_mesh(object, to_world_frame=False)
        os.makedirs('mesh_cache', exist_ok=True)
        random_path = f'mesh_cache/{generate_random_string()}.obj'
        object_mesh.export(random_path)
        object_pose = object.get_pose()
        ee_pose = self.end_effector.get_pose()
        object_pose_rel_ee = ee_pose.inv() * object_pose
        self.planner.update_attached_mesh(random_path, pose=pymp.Pose(object_pose_rel_ee.p, object_pose_rel_ee.q))
        os.remove(random_path)

    def planner_detach_obj(self):
        self.planner.detach_object('panda_9_mesh', also_remove=True)

    def step(self):
        self.scene.step()

    def update_render(self):
        self.scene.update_render()

    def hide_robot_visual(self):
        if self.robot is None:
            raise ValueError("No robot in the scene, cannot hide robot visual")
        for link in self.robot.get_links():
            link.hide_visual()

    def unhide_robot_visual(self):
        if self.robot is None:
            raise ValueError("No robot in the scene, cannot unhide robot visual")
        for link in self.robot.get_links():
            link.unhide_visual()

    def create_viewer(
            self,
            resolutions: tuple[int, int] = (1440, 1440),
            camera_xyz: tuple[float, float, float] = (1.2, 0.25, 0.4),
            camera_rpy: tuple[float, float, float] = (0.0, -0.4, 2.7),
            near: float = 0.05,
            far: float = 100,
            fovy: float = 1,
    ) -> Viewer:
        self.viewer = Viewer(self.renderer, resolutions=resolutions)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(*camera_xyz)
        self.viewer.set_camera_rpy(*camera_rpy)
        self.viewer.window.set_camera_parameters(near, far, fovy)

        return self.viewer

    def add_camera(
            self,
            direction: str = '+x',
            fovy: float = None
    ) -> CameraEntity:
        if fovy is None:
            fovy = np.deg2rad(60)
        camera = self.scene.add_camera(
            name=direction,
            fovy=fovy,
            width=768,
            height=768,
            near=0.05,
            far=100
        )
        camera.set_pose(DIRECTION2POSE[direction])
        return camera

    def get_picture(
            self,
            direction: str = '+x',
            additional_translation: np.ndarray = None,
            additional_rotation: np.ndarray = None,
            get_depth: bool = False,
            debug_viewer: bool = False
    ) -> tuple[Image.Image, CameraEntity] or tuple[Image.Image, np.ndarray, CameraEntity]:
        camera = self.add_camera(direction)
        if additional_translation is not None:
            pose = camera.get_pose()
            p, q = pose.p, pose.q
            p += additional_translation
            if additional_rotation is not None:
                q = quat_mul_wxyz(additional_rotation, q)
            camera.set_pose(Pose(p=p, q=q))
        if debug_viewer:
            viewer = self.create_viewer()
            while not viewer.closed:
                # self.step()
                self.update_render()
                viewer.render()

        self.update_render()
        image = get_rgba_img(camera=camera)
        image = Image.fromarray(image).convert('RGB')
        if debug_viewer:
            image.show()

        if get_depth:
            depth_img = get_depth_img(camera)
            return image, depth_img, camera
        return image, camera

    def get_8_pictures(self, debug_viewer: bool = False) -> Image:
        cameras = []
        for direction in ['+y+z', '+y-z', '+x+y', '-x+y', '-y+z', '-y-z', '+x-y', '-x-y']:
            cameras.append(self.add_camera(direction))
        if debug_viewer:
            viewer = self.create_viewer()
            while not viewer.closed:
                self.step()
                self.update_render()
                viewer.render()

        self.step()
        self.update_render()

        imgs = []
        for i in range(8):
            camera = cameras[i]
            image = get_rgba_img(camera=camera)
            image = Image.fromarray(image)
            if debug_viewer:
                image.show()
            imgs.append(image)
        return imgs