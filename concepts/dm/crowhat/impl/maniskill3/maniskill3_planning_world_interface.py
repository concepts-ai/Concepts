from typing import Any, Iterator, Optional, Union, Tuple, List, Generic, TypeVar, cast

import numpy as np
from bidict import bidict
from sapien import Scene, Pose, Entity
from sapien.pysapien.physx import PhysxArticulation, PhysxArticulationJoint, PhysxCpuSystem, PhysxContactPoint

from concepts.dm.crowhat.world.planning_world_interface import GeometricContactInfo, PlanningWorldInterface
from concepts.math.cad.mesh_utils import trimesh_to_open3d_mesh
from concepts.math.rotationlib_wxyz import wxyz2xyzw
from concepts.utils.typing_utils import Open3DPointCloud, Open3DTriangleMesh, Trimesh, Vec3f, Vec4f
from mani_skill import BaseEnv
from mani_skill.utils.geometry.trimesh_utils import get_actor_mesh
from mani_skill.utils.sapien_utils import get_pairwise_contacts

K = TypeVar('K')
V = TypeVar('V')

class _Name2Pointer(bidict[K, V], Generic[K, V]):
    pass


class ManiSkill3PlanningWorldInterface(PlanningWorldInterface):
    """
    Here I give the type correspondence between planningworld interface and sapien:
    - object: actor
    - body: articulation
    - link: link
    - object_id: per_scene_id
    In ManiSkill3, name must be unique for each actor and articulation, so I use name to identify objects.
    Note that the quaternion convention is wxyz.
    """
    def __init__(self, env: BaseEnv):
        self._sapien_scene: Scene = env.scene.sub_scenes[0]
        self._ignored_collision_pairs = list()
        self._name2object = _Name2Pointer[str, Entity]()
        self._name2body = _Name2Pointer[str, PhysxArticulation]()
        self.update_name2object()
        self.update_name2body()

    @property
    def sapien_scene(self) -> Scene:
        return self._sapien_scene

    def update_name2object(self):
        self._name2object.clear()
        for actor in self._sapien_scene.get_all_actors():
            self._name2object[actor.get_name()] = actor
        for articulation in self._sapien_scene.get_all_articulations():
            for link in articulation.get_links():
                self._name2object[link.get_name()] = link.entity

    def update_name2body(self):
        self._name2body.clear()
        for body in self._sapien_scene.get_all_articulations():
            self._name2body[body.get_name()] = body

    def find_link_by_name_with_body(self, body_name: str, link_name: str) -> Entity:
        return self._name2body[body_name].find_link_by_name(f'{body_name.replace("_", "-")}_{link_name}').entity

    def find_joint_by_name_with_body(self, body_name: str, joint_name: str) -> PhysxArticulationJoint:
        return self._name2body[body_name].find_joint_by_name(f'{body_name.replace("_", "-")}_{joint_name}')

    def _get_objects(self) -> List[Any]:
        return list(self._sapien_scene.get_all_actors())

    def _get_object_name(self, identifier: str) -> str:
        return identifier

    def _get_object_pose(self, identifier: str) -> Tuple[Vec3f, Vec4f]:
        pose = self._name2object[identifier].get_pose()
        return pose.p, pose.q

    def _set_object_pose(self, identifier: str, pose: Tuple[Vec3f, Vec4f]):
        self._name2object[identifier].set_pose(Pose(pose[0], wxyz2xyzw(pose[1])))

    def _get_link_pose(self, body_id: str, link_id: str) -> Tuple[Vec3f, Vec4f]:
        pose = self.find_link_by_name_with_body(body_id, link_id).get_pose()
        return pose.p, pose.q

    def _get_object_point_cloud(self, identifier: str, **kwargs) -> Open3DPointCloud:
        mesh: Open3DTriangleMesh = self._get_object_mesh(identifier, mode='open3d', **kwargs)
        num_points = kwargs.get('num_points', 10000)
        return mesh.sample_points_uniformly(num_points, use_triangle_normal=True)

    def _get_object_mesh(self, identifier: str, mode: str = 'open3d', **kwargs) -> Union[Open3DTriangleMesh, Trimesh]:
        mesh = get_actor_mesh(self._name2object[identifier], **kwargs)
        if mode == 'open3d':
            return trimesh_to_open3d_mesh(mesh)
        elif mode == 'trimesh':
            return mesh
        else:
            raise ValueError(f'Invalid mode: {mode}')

    def _save_world(self) -> bytes:
        # TODO (Yuyao Liu @ 2024/12/24): figure out how to check whether or not the PhysxSystem runs on CPU.
        physx_system = cast(PhysxCpuSystem, self.sapien_scene.physx_system)
        return physx_system.pack()

    def _restore_world(self, checkpoint: bytes):
        physx_system = cast(PhysxCpuSystem, self.sapien_scene.physx_system)
        physx_system.unpack(checkpoint)

    def _checkpoint_world(self) -> Iterator[Any]:
        checkpoint = self._save_world()
        yield
        self._restore_world(checkpoint)

    def _get_contact_points(
        self,
        a: Optional[Union[str, int]] = None,
        b: Optional[Union[str, int]] = None,
        max_distance: Optional[float] = None,
        ignored_collision_bodies: Optional[List[Union[str, int]]] = None
    ) -> List[GeometricContactInfo]:
        all_contacts = self.sapien_scene.get_contacts()
        if a is None or b is None:
            # TODO (Yuyao Liu @ 2024/12/24): add support for one of a and b is None
            raise NotImplementedError
        entity_a = self._name2object[a]
        entity_b = self._name2object[b]
        contacts = get_pairwise_contacts(all_contacts, entity_a, entity_b)
        geometric_contacts = list()
        for contact, in_contact in contacts:
            if ignored_collision_bodies is not None:
                raise NotImplementedError
            for point in contact.points:
                point = cast(PhysxContactPoint, point)
                if point.separation > max_distance:
                    continue
                geometric_contacts.append(GeometricContactInfo(
                    body_a=contact.bodies[0].entity.get_name(),
                    body_b=contact.bodies[1].entity.get_name(),
                    link_a=contact.bodies[0].entity.get_name(),
                    link_b=contact.bodies[1].entity.get_name(),
                    # TODO (Yuyao Liu @ 2024/12/24): check the correctness, and figure out whether we want obj-centric or in-the-world
                    position_on_a=point.position,
                    position_on_b=point.position,
                    contact_normal_on_a=point.normal,
                    contact_normal_on_b=point.normal,
                    contact_distance=point.separation
                ))
        return geometric_contacts

    def get_single_contact_normal(
            self, object_id: str,
            support_object_id: str,
            deviation_tol: float = 0.05,
            return_center: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        contacts = self._get_contact_points(object_id, support_object_id)
        if len(contacts) == 0:
            self.sapien_scene.step()
            contacts = self._get_contact_points(object_id, support_object_id)
        return self._compute_single_contact_normal_from_contacts(contacts, object_id, support_object_id, deviation_tol=deviation_tol, return_center=return_center)


    # def add_ignore_collision_pair_by_id(self, body_a, link_a, body_b, link_b):
    #     self._ignored_collision_pairs.append((body_a, link_a, body_b, link_b))
    #
    # def add_ignore_collision_pair_by_name(self, link_a, link_b):
    #     body_a, link_a = self._client.world.get_link_index(link_a)
    #     body_b, link_b = self._client.world.get_link_index(link_b)
    #     self.add_ignore_collision_pair_by_id(body_a, link_a, body_b, link_b)