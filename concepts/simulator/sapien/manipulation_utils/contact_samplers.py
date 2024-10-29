import itertools
import json
import time
from dataclasses import dataclass
from time import time
from typing import Optional, Iterator, Sequence, Callable, cast, List

import jacinle
import numpy as np
import open3d as o3d
import sapien.core as sapien
import tabulate

from mani_skill2.utils.trimesh_utils import get_actor_mesh
from mplib import pymp
from open3d import geometry
from sapien.core.pysapien import Scene, Actor, Pose
from sapien.utils import Viewer

from tasks.mugScene import MugScene
from tasks.tableScene import TableScene
from utils.mesh_utils import trimesh_to_open3d_mesh
from utils.rotation_utils import enumerate_quaternion_from_vectors, quat_mul, rotate_vector, wxyz2xyzw, xyzw2wxyz, \
    cross, find_orthogonal_vector, get_quaternion_from_axes
from utils.sapien_utils import get_contacts_by_id, get_contacts_with_articulation  # , ActorSaver

logger = jacinle.logging.get_logger(__file__)


@dataclass
class GraspParameter(object):
    point1: np.ndarray
    normal1: np.ndarray
    point2: np.ndarray
    normal2: np.ndarray
    ee_pos: np.ndarray
    ee_quat_wxyz: np.ndarray
    qpos: np.ndarray


def sample_grasp(
        table_scene: TableScene,
        object_id: int,
        gripper_distance: float = 0.08,
        pregrasp_distance: float = 0.105,
        max_intersection_distance: float = 10,
        max_attempts: int = 10000000,
        verbose: bool = False,
        max_test_points_before_first: int = 250,
        max_test_points: int = 100000000,
        batch_size: int = 100,
        surface_pointing_tol: float = 0.9,
        gripper_min_distance: float = 0.0001,
        np_random: Optional[np.random.RandomState] = None,
        guidance_center: Optional[np.ndarray] = None,
        guidance_direction: Optional[np.ndarray] = None,
        guidance_radius: float = 0.01,
        start_time=None,
        timeout=60
) -> Iterator[GraspParameter]:
    """Given the name of the object, sample a 6D grasp pose. Before calling this function, we should make sure that the gripper is open."""
    scene = table_scene.scene
    robot = table_scene.robot

    if np_random is None:
        np_random = np.random

    mesh = trimesh_to_open3d_mesh(get_actor_mesh(scene.find_actor_by_id(object_id)))
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    found = False
    nr_test_points_before_first = 0

    for _ in range(int(max_test_points / batch_size)):
        # TODO: accelerate the computation.
        if guidance_center is None:
            pcd = mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
        else:
            pcd: o3d.geometry.PointCloud = mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
            # get the points that are within the guidance radius
            feasible_point_indices = np.where(np.linalg.norm(np.asarray(pcd.points) - guidance_center, axis=1) < guidance_radius)[0][:batch_size]
            pcd = pcd.select_by_index(feasible_point_indices)

        indices = list(range(len(pcd.points)))
        np_random.shuffle(indices)
        for i in indices:
            if start_time is not None:
                if time() - start_time > timeout:
                    return
            if not found:
                nr_test_points_before_first += 1

            point = np.asarray(pcd.points[i])
            normal = np.asarray(pcd.normals[i])

            if guidance_direction is not None:
                if np.abs(np.dot(normal, guidance_direction)) > 1 - surface_pointing_tol:
                    continue

            if verbose:
                print('sample_grasp_v2_gen', 'point', point, 'normal', normal)

            point2 = point - normal * max_intersection_distance
            other_intersection = mesh_line_intersect(t_mesh, point2, normal)

            if verbose:
                print('  other_intersection', other_intersection)

            # if no intersection, try the next point.
            if other_intersection is None:
                if verbose:
                    print('  skip: no intersection')
                continue

            other_point, other_normal = other_intersection

            # if two intersection points are too close, try the next point.
            if np.linalg.norm(other_point - point) < gripper_min_distance:
                if verbose:
                    print('  skip: too close')
                continue

            # if the surface normals are too different, try the next point.
            if np.abs(np.dot(normal, other_normal)) < surface_pointing_tol:
                if verbose:
                    print('  skip: normal too different')
                continue

            grasp_center = (point + other_point) / 2
            grasp_distance = np.linalg.norm(point - other_point)
            grasp_normal = normal

            if grasp_distance > gripper_distance:
                if verbose:
                    print('  skip: too far')
                continue

            ee_d = grasp_normal
            # ee_u and ee_v are two vectors that are perpendicular to ee_d
            ee_u = find_orthogonal_vector(ee_d)
            ee_v = cross(ee_u, ee_d)

            # if verbose:
            #     print('  grasp_center', grasp_center, 'grasp_distance', grasp_distance)
            #     print('  grasp axes:\n', np.array([ee_d, ee_u, ee_v]))

            # enumerate four possible grasp orientations
            for ee_norm1 in [ee_u, ee_v, -ee_u, -ee_v]:
                ee_norm2 = cross(ee_d, ee_norm1)
                ee_quat = get_quaternion_from_axes(ee_norm2, ee_d, ee_norm1)
                ee_quat_wxyz = xyzw2wxyz(ee_quat)

                qpos, hand_pos= table_scene.grasp_center_ik(
                    grasp_center=grasp_center,
                    ee_quat_wxyz=ee_quat_wxyz,
                    start_qpos=robot.get_qpos(),
                    mask=[0, 0, 0, 0, 0, 0, 0, 1, 1], # don't change the qpos of the gripper fingers
                    threshold=1e-4
                )

                if qpos is None:
                    if verbose:
                        print('  skip: ik fail')
                    continue

                rv = collision_free_qpos(table_scene, object_id, qpos, verbose=verbose)
                if rv:
                    found = True
                    yield GraspParameter(
                        point1=point, normal1=normal,
                        point2=other_point, normal2=other_normal,
                        ee_pos=hand_pos, ee_quat_wxyz=ee_quat_wxyz,
                        qpos=qpos
                    )
                elif verbose:
                    print('    gripper pos', grasp_center)
                    print('    gripper quat', ee_quat)
                    print('    skip: collision')

        if not found and nr_test_points_before_first > max_test_points_before_first:
            if verbose:
                logger.warning(f'Failed to find a grasp after {nr_test_points_before_first} points tested.')
            return


def gen_qpos_to_qpos_trajectory(
        table_scene: TableScene,
        start_qpos,
        end_qpos: np.ndarray,
        exclude_ids: Optional[List[int]] = None,
        planning_time: float = 1.0,
        pcd_resolution: float = 1e-3,
        ignore_env: bool = False,
        verbose: bool = False
):
    if ignore_env:
        table_scene.planner.remove_point_cloud()
    else:
        table_scene.update_env_pcd(exclude_ids=exclude_ids, pcd_resolution=pcd_resolution, verbose=verbose)

    result = table_scene.planner.plan_qpos(
        [end_qpos],
        start_qpos,
        time_step=1 / table_scene.fps,
        planning_time=planning_time,
        verbose=verbose,
    )
    return result


def gen_qpos_to_pose_trajectory(
        table_scene: TableScene,
        start_qpos,
        end_pose: pymp.Pose,
        exclude_ids: Optional[List[int]] = None,
        planning_time: float = 1.0,
        pcd_resolution: float = 1e-3,
        ignore_env: bool = False,
        verbose: bool = False,
        use_screw: bool = False
):
    if ignore_env:
        table_scene.planner.remove_point_cloud()
    else:
        table_scene.update_env_pcd(exclude_ids=exclude_ids, pcd_resolution=pcd_resolution)
    if not use_screw:
        result = table_scene.planner.plan_pose(
            end_pose,
            start_qpos,
            mask=np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]),
            time_step=1 / table_scene.fps,
            planning_time=planning_time,
            verbose=verbose
        )
    else:
        result = table_scene.planner.plan_screw(
            end_pose,
            start_qpos,
            time_step=1 / table_scene.fps,
            verbose=verbose
        )
    return result


@dataclass
class PushParameter(object):
    push_pos: np.ndarray
    push_dir: np.ndarray
    distance: float


def sample_push_with_support(
        scene: Scene,
        object_id: int,
        support_id: int,
        max_attempts: int = 1000,
        batch_size: int = 100,
        push_distance_fn: Optional[Callable] = None,
        np_random: Optional[np.random.RandomState] = None,
        verbose: bool = False
) -> Iterator[PushParameter]:
    if push_distance_fn is None:
        push_distance_fn = lambda: 0.1

    if np_random is None:
        np_random = np.random

    object_mesh: geometry.TriangleMesh = trimesh_to_open3d_mesh(get_actor_mesh(scene.find_actor_by_id(object_id)))

    nr_batches = int(max_attempts / batch_size)
    feasible_point_indices = list()
    for _ in range(nr_batches):
        pcd = object_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)

        # get the contact points between the object and the support object
        contact_normal = get_single_contact_normal(scene, object_id, support_id)

        # filter out the points that are not on the contact plane
        # feasible_point_cond = np.abs(np.asarray(pcd.normals).dot(contact_normal)) < 0.02
        # TODO(Yuyao Liu @ 2024/03/28): figure out the threshold
        feasible_point_cond = np.abs(np.asarray(pcd.normals).dot(contact_normal)) < 0.1
        feasible_point_indices = np.where(feasible_point_cond)[0]

        # print(f'Found {len(feasible_point_indices)} feasible points.')

        if len(feasible_point_indices) == 0:
            continue

        # o3d.visualization.draw([pcd.select_by_index(feasible_point_indices), mesh])

        np_random.shuffle(feasible_point_indices)
        rows = list()
        for index in feasible_point_indices:
            rows.append((index, pcd.points[index], -pcd.normals[index]))

        if verbose:
            jacinle.log_function.print(tabulate.tabulate(rows, headers=['index', 'point', 'normal']))

        # create a new point cloud
        for index in feasible_point_indices:
            if verbose:
                jacinle.log_function.print('sample_push_with_support', 'point', pcd.points[index], 'normal',
                                           -pcd.normals[index])
            yield PushParameter(np.asarray(pcd.points[index]), -np.asarray(pcd.normals[index]), push_distance_fn())

    if len(feasible_point_indices) == 0:
        raise ValueError(
            f'No feasible points for {object_id} on {support_id} after {nr_batches * batch_size} attempts.')


@dataclass
class IndirectPushParameter(object):
    object_push_pos: np.ndarray
    object_push_dir: np.ndarray

    tool_pos: np.ndarray
    tool_quat_wxyz: np.ndarray

    tool_point_pos: np.ndarray
    tool_point_normal: np.ndarray

    prepush_distance: float = 0.05
    push_distance: float = 0.1

    @property
    def total_push_distance(self):
        return self.prepush_distance + self.push_distance


def load_indirect_push_parameter(file_path: str) -> IndirectPushParameter:
    with open(file_path, 'r') as f:
        data = json.load(f)
        for key in data.keys():
            if type(data[key]) is list:
                data[key] = np.array(data[key])
        return IndirectPushParameter(**data)


def sample_indirect_push_with_support(
        scene: Scene,
        tool_id: int,
        object_id: int,
        support_id: int,
        prepush_distance: float = 0.05,
        max_attempts: int = 10000000,
        batch_size: int = 1000,
        filter_push_dir: Optional[np.ndarray] = None,
        push_distance_distribution: Sequence[float] = (0.1, 0.15),
        push_distance_sample: bool = False,
        contact_normal_tol: float = 0.01,
        np_random: Optional[np.random.RandomState] = None,
        verbose: bool = False,
        check_reachability: Callable[[np.ndarray], bool] = None,
        tool_contact_point_filter: Callable[[np.ndarray], np.ndarray] = None
) -> Iterator[IndirectPushParameter]:
    if np_random is None:
        np_random = np.random

    tool_mesh: geometry.TriangleMesh = trimesh_to_open3d_mesh(get_actor_mesh(scene.find_actor_by_id(tool_id)))
    object_mesh: geometry.TriangleMesh = trimesh_to_open3d_mesh(get_actor_mesh(scene.find_actor_by_id(object_id)))

    current_tool_pose = scene.find_actor_by_id(tool_id).get_pose()
    current_tool_pos, current_tool_quat_wxyz = current_tool_pose.p, current_tool_pose.q
    current_tool_quat_xyzw = wxyz2xyzw(current_tool_quat_wxyz)

    nr_batches = int(max_attempts / batch_size)
    contact_normal = get_single_contact_normal(scene, object_id, support_id)

    for _ in range(nr_batches):
        tool_pcd = tool_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
        object_pcd = object_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)

        # feasible_object_point_cond = np.abs(np.asarray(object_pcd.normals).dot(contact_normal)) < 0.01 # 0.1 for
        # real demo.
        feasible_object_point_cond = np.abs(np.asarray(object_pcd.normals).dot(contact_normal)) < contact_normal_tol
        if filter_push_dir is not None:
            feasible_object_point_cond = np.logical_and(
                feasible_object_point_cond,
                np.asarray(object_pcd.normals, dtype=np.float32).dot(-filter_push_dir) > 0.8
            )

        feasible_object_point_indices = np.where(feasible_object_point_cond)[0]

        if tool_contact_point_filter is not None:
            # transform the pcd to world frame
            world2tool = np.linalg.inv(scene.find_actor_by_id(tool_id).pose.to_transformation_matrix())
            tool_pcd_np = np.asarray(tool_pcd.points)
            tool_pcd_homogeneous = np.concatenate([tool_pcd_np, np.ones([*tool_pcd_np.shape[:-1], 1])], axis=-1)
            tool_pcd_tool = (tool_pcd_homogeneous @ world2tool.T)[:,:-1]
            tool_contact_point_cond = tool_contact_point_filter(tool_pcd_tool)
            filtered_tool_contact_point_indices = np.where(tool_contact_point_cond)[0]
        else:
            filtered_tool_contact_point_indices = range(batch_size)

        all_index_pairs = list(itertools.product(feasible_object_point_indices, filtered_tool_contact_point_indices))
        np_random.shuffle(all_index_pairs)
        for object_index, tool_index in all_index_pairs:
            object_point_pos = np.asarray(object_pcd.points[object_index])
            object_point_normal = -np.asarray(object_pcd.normals[object_index])  # point inside

            tool_point_pos = np.asarray(tool_pcd.points[tool_index])
            tool_point_normal = np.asarray(tool_pcd.normals[tool_index])  # point outside (towards the tool)            

            # Solve for a quaternion that aligns the tool normal with the object normal
            for rotation_quat_xyzw in enumerate_quaternion_from_vectors(tool_point_normal, object_point_normal, 4):
                # This is the world coordinate for the tool point after rotation.
                new_tool_point_pos = current_tool_pos + rotate_vector(tool_point_pos - current_tool_pos,
                                                                      rotation_quat_xyzw)
                # Now compute the displacement for the tool object
                final_tool_pos = object_point_pos - new_tool_point_pos + current_tool_pos
                final_tool_pos -= object_point_normal * prepush_distance
                final_tool_quat_xyzw = quat_mul(rotation_quat_xyzw, current_tool_quat_xyzw)

                success = True
                # check collision
                init_state = scene.pack()  # backup state
                cast(Actor, scene.find_actor_by_id(tool_id)).set_pose(
                    Pose(final_tool_pos, xyzw2wxyz(final_tool_quat_xyzw)))

                scene.step()
                contacts = get_contacts_by_id(scene, tool_id)
                if len(contacts) > 0:
                    success = False
                if check_reachability is not None and success:
                    if not check_reachability(np.asarray(trimesh_to_open3d_mesh(
                            get_actor_mesh(scene.find_actor_by_id(tool_id))).sample_points_uniformly(1000,
                                                                                                     use_triangle_normal=True).points)):
                        success = False

                scene.unpack(init_state)  # reset state

                if success:
                    if push_distance_sample:
                        distances = [np_random.choice(push_distance_distribution)]
                    else:
                        distances = push_distance_distribution
                    kwargs = dict(
                        object_push_pos=object_point_pos,
                        object_push_dir=object_point_normal,
                        tool_pos=final_tool_pos,
                        tool_quat_wxyz=xyzw2wxyz(final_tool_quat_xyzw),
                        tool_point_pos=rotate_vector(tool_point_pos - current_tool_pos,
                                                     rotation_quat_xyzw) + final_tool_pos,
                        tool_point_normal=rotate_vector(tool_point_normal, rotation_quat_xyzw),
                        prepush_distance=prepush_distance
                    )
                    for distance in distances:
                        yield IndirectPushParameter(**kwargs, push_distance=distance)


def collision_free_qpos(table_scene: TableScene, object_id: int, qpos: np.ndarray, exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
    """Check whether the given qpos is collision free. The function also accepts a list of object ids to exclude (e.g., the object in hand).

    Args:
        robot: the robot.
        qpos: the qpos to check.
        exclude: the object ids to exclude.
        verbose: whether to print the collision information.

    Returns:
        True if the qpos is collision free.
    """
    #check self collision
    collisions = table_scene.planner.check_for_self_collision(state=qpos)
    if len(collisions) > 0:
        if verbose:
            print(f'  collision_free_qpos: self collision')
        return False

    # check collision with the environment
    scene = table_scene.scene
    robot = table_scene.robot
    init_state = scene.pack()  # backup state
    robot.set_qpos(qpos)

    scene.step()

    contacts = get_contacts_with_articulation(scene, robot, distance_threshold=0.0001)
    if exclude is not None:
        for c in contacts:
            if c.actor0.get_id() not in exclude and c.actor1.get_id() not in exclude:
                if verbose:
                    print(f'  collision_free_qpos: collide between {c.actor0.get_name()} and {c.actor1.get_name()}')
                scene.unpack(init_state)
                return False
    else:
        for c in contacts:
            if verbose:
                print(f'  collision_free_qpos: collide between {c.actor0.get_name()} and {c.actor1.get_name()}')
            scene.unpack(init_state)
            return False
    scene.unpack(init_state)
    return True


def mesh_line_intersect(t_mesh: o3d.t.geometry.TriangleMesh, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Intersects a ray with a mesh.

    Args:
        t_mesh: the mesh to intersect with.
        ray_origin: the origin of the ray.
        ray_direction: the direction of the ray.

    Returns:
        A tuple of (point, normal) if an intersection is found, None otherwise.
    """

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    ray = o3d.core.Tensor.from_numpy(np.array(
        [[ray_origin[0], ray_origin[1], ray_origin[2], ray_direction[0], ray_direction[1], ray_direction[2]]],
        dtype=np.float32
    ))
    result = scene.cast_rays(ray)

    # no intersection.
    if result['geometry_ids'][0] == scene.INVALID_ID:
        return None

    inter_point = np.asarray(ray_origin) + np.asarray(ray_direction) * result['t_hit'][0].item()
    inter_normal = result['primitive_normals'][0].numpy()
    return inter_point, inter_normal


def get_single_contact_normal(scene: Scene, object_id: int, support_id: int, deviation_tol: float = 0.05) -> np.ndarray:
    contacts = get_contacts_by_id(scene, object_id, support_id)

    if len(contacts) == 0:
        raise ValueError(
            f'No contact between {scene.find_actor_by_id(object_id).get_name()} and {scene.find_actor_by_id(support_id).get_name()}')

    contact_normals = np.array([point.normal for contact in contacts for point in contact.points])
    contact_normal_avg = np.mean(contact_normals, axis=0)
    contact_normal_avg /= np.linalg.norm(contact_normal_avg)

    deviations = np.abs(1 - contact_normals.dot(contact_normal_avg) / np.linalg.norm(contact_normals, axis=1))
    if np.max(deviations) > deviation_tol:
        raise ValueError(
            f'Contact normals of {scene.find_actor_by_id(object_id).get_name()} and {scene.find_actor_by_id(support_id).get_name()} are not consistent. This is likely due to multiple contact points.\n'
            f'  Contact normals: {contact_normals}\n  Deviations: {deviations}.'
        )

    return contact_normal_avg


def main_0():
    from utils.rotation_utils import rpy2wxyz
    from utils.sapien_utils import create_sphere, create_box
    from utils.camera_utils import get_rgba_img, imgs2mp4
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--indirect', type=int, default=0)
    parser.add_argument('--target', type=str, choices=['box', 'ball'], default='ball')
    parser.add_argument('--tool', type=str, choices=['hook'], default='hook')
    parser.add_argument('--output-video', type=int, default=0)
    parser.add_argument('--check-reachability', type=int, default=0)

    args = parser.parse_args()

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    fps = 100.0
    scene = engine.create_scene()
    scene.set_timestep(1 / fps)

    material = scene.create_physical_material(0.6, 0.6, 0.1)  # Create a physical material
    ground = scene.add_ground(altitude=0, render_half_size=[20, 20, 0.1], material=material)  # Add a ground

    target = None
    if args.target == 'box':
        target = create_box(
            scene,
            sapien.Pose(p=[0, 0, 1]),
            half_size=[1, 1, 1],
            color=[1, 0, 0],
            name='box'
        )
    elif args.target == 'ball':
        target = create_sphere(
            scene,
            sapien.Pose(p=[0, 0, 1]),
            radius=1,
            color=[1, 0, 0],
            name='ball'
        )
    else:
        raise ValueError(f'no such target object type {args.target}')

    if args.indirect:
        builder = scene.create_actor_builder()
        builder.add_multiple_collisions_from_file('assets/custom_obj/hook/hook_vhacd.obj')
        builder.add_visual_from_file('assets/custom_obj/hook/hook_vhacd.obj')
        tool = builder.build(name='hook')
        tool.set_pose(sapien.Pose(p=[0, 0, 1]))
    else:
        pass

    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [1, 1, 1], shadow=True)

    viewer = Viewer(renderer, resolutions=(768, 768))  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=0, y=0, z=20)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.pi / 2, y=np.pi / 2)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    scene.step()

    camera = None
    if args.output_video:
        camera: sapien.CameraEntity = scene.add_camera(
            name='camera',
            fovy=np.deg2rad(80),
            width=768,
            height=768,
            near=0.05,
            far=100
        )
        camera.set_pose(sapien.Pose(p=[0, 0, 20], q=rpy2wxyz([0, np.pi / 2, -np.pi / 2])))

    if args.indirect:
        check_reachability = None
        if args.check_reachability:
            check_reachability: Callable[[np.ndarray], bool] = lambda point_cloud: np.any(point_cloud[:, 1] >= 7)
        push_generator = sample_indirect_push_with_support(scene, tool.get_id(), target.get_id(), ground.get_id(),
                                                           contact_normal_tol=0.1, filter_push_dir=np.array([0, 1, 0]),
                                                           check_reachability=check_reachability)
    else:
        push_generator = sample_push_with_support(scene, target.get_id(), ground.get_id())

    current_marker_1 = None
    current_marker_2 = None
    step = 0

    images = []
    while not viewer.closed:  # Press key q to quit
        if step % 100 == 0:
            if current_marker_1 is not None:
                scene.remove_actor(current_marker_1)
            if current_marker_2 is not None:
                scene.remove_actor(current_marker_2)

            if args.indirect:
                push_parameter: IndirectPushParameter = next(push_generator)
                tool.set_pose(Pose(p=push_parameter.tool_pos, q=push_parameter.tool_quat_wxyz))
                current_marker_1 = create_sphere(scene, pose=Pose(p=push_parameter.object_push_pos), radius=0.1,
                                                 color=[0, 0, 1], only_visual=True)
                current_marker_2 = create_sphere(scene, pose=Pose(p=push_parameter.tool_point_pos), radius=0.1,
                                                 color=[0, 1, 0], only_visual=True)
            else:
                push_parameter: PushParameter = next(push_generator)
                current_marker_1 = create_sphere(scene, pose=Pose(p=push_parameter.push_pos), radius=0.2,
                                                 color=[0, 0, 1], only_visual=True)

        scene.update_render()  # Update the world to the renderer
        viewer.render()

        if camera is not None:
            images.append(get_rgba_img(camera=camera))

        step += 1
        if step >= 4000:
            break

    if camera is not None:
        target_name = f'{args.target}'
        tool_name = f'{args.tool}_' if args.indirect else ''
        indirect_name = 'indirect_' if args.indirect else ''
        reachability_name = 'reachable_' if args.check_reachability and args.indirect else ''
        imgs2mp4(images, f'videos/sampling_{indirect_name}{reachability_name}push_{tool_name}{target_name}.mp4')


def main():
    mug_scene = MugScene(mug_id=13, add_robot=True, fps=480)
    viewer = mug_scene.create_viewer()
    mug_scene.open_gripper()

    grasp_generator = sample_grasp(mug_scene, object_id=mug_scene.mug.get_id(), gripper_distance=0.08, verbose=False)
    for grasp in grasp_generator:
        grasp_qpos = grasp.qpos
        start_qpos = mug_scene.robot.get_qpos()
        result = gen_qpos_to_qpos_trajectory(mug_scene, start_qpos, grasp_qpos)
        mug_scene.follow_path(result)
        mug_scene.close_gripper()
        grasp_center = (grasp.point1 + grasp.point2) / 2
        new_grasp_center = grasp_center + np.array([0, 0, 0.1])
        qpos, _ = mug_scene.grasp_center_ik(
            grasp_center=new_grasp_center,
            ee_quat_wxyz=grasp.ee_quat_wxyz,
            start_qpos=mug_scene.robot.get_qpos(),
            threshold=1e-3,
            exclude_ids=[mug_scene.mug.get_id()]
        )
        result = gen_qpos_to_qpos_trajectory(mug_scene, mug_scene.robot.get_qpos(), qpos, exclude_ids=[mug_scene.mug.get_id()])
        mug_scene.follow_path(result)
        while not viewer.closed:
            mug_scene.step()
            mug_scene.update_render()
            viewer.render()
        break


if __name__ == '__main__':
    main()
