from __future__ import annotations

import os
from typing import Optional, Sequence, cast

import numpy as np
from mplib import Pose, PlanningWorld, ArticulatedModel, Planner
from mplib.pymp.collision_detection import AllowedCollisionMatrix, WorldDistanceResult, AllowedCollision, \
    WorldCollisionResult
from mplib.pymp.collision_detection.fcl import FCLObject, CollisionObject, Box, Capsule, Convex, Halfspace, Sphere, \
    BVHModel, FCLModel, collide, distance
from mplib.pymp.planning.ompl import OMPLPlanner
from sapien.core import ActorBase, LinkBase, CollisionShape, BoxGeometry, CapsuleGeometry, ConvexMeshGeometry, \
    PlaneGeometry, NonconvexMeshGeometry, SphereGeometry, Scene, Articulation, ArticulationBase
from transforms3d.euler import euler2quat

from concepts.simulator.sapien2.srdf_exporter import export_srdf
from concepts.simulator.sapien2.table_top_scene import PANDA_PATH
from concepts.simulator.sapien2.urdf_exporter import export_kinematic_chain_urdf


# TODO: link names?
def convert_object_name(obj: ActorBase) -> str:
    """
    Constructs a unique name for the corresponding mplib object.
    This is necessary because mplib objects assume unique names.

    :param obj: a SAPIEN object
    :return: the unique mplib object name
    """
    if isinstance(obj, ArticulationBase):
        return f"{obj.name}_{obj.get_links()[0].get_id()}"
    if isinstance(obj, ActorBase):
        return f"{obj.name}_{obj.get_id()}"
    else:
        raise NotImplementedError(f"Unknown SAPIEN object type {type(obj)}")


class SapienPlanningWorld(PlanningWorld):
    def __init__(
        self,
        sapien_scene: Scene,
        planned_articulations: list[Articulation] = [],  # noqa: B006
    ):
        """
        Creates an mplib.PlanningWorld from a sapien.Scene.

        :param planned_articulations: list of planned articulations.
        """
        super().__init__([])
        self._sim_scene = sapien_scene

        articulations: list[Articulation] = sapien_scene.get_all_articulations()
        actors: list[ActorBase] = sapien_scene.get_all_actors()

        for articulation in articulations:
            urdf_str = export_kinematic_chain_urdf(articulation)
            srdf_str = export_srdf(articulation)

            # Convert all links to FCLObject
            collision_links = [
                fcl_obj
                for link in articulation.get_links()
                if (fcl_obj := self.convert_sapien_actor(link)) is not None
            ]

            articulated_model = ArticulatedModel.create_from_urdf_string(
                urdf_str,
                srdf_str,
                collision_links=collision_links,
                gravity=sapien_scene.get_config().gravity,  # type: ignore
                link_names=[link.name for link in articulation.get_links()],
                joint_names=[j.name for j in articulation.get_active_joints()],
                verbose=True,
            )
            articulated_model.set_qpos(articulation.get_qpos())  # update qpos  # type: ignore
            self.add_articulation(articulated_model)

        for articulation in planned_articulations:
            self.set_articulation_planned(convert_object_name(articulation), True)

        for actor in actors:
            component = actor.get_collision_shapes()
            assert len(component) != 0, (
                f"No collision shape found in {actor.name}: "
                f"{actor.components=}"
            )

            # Convert collision shapes at current global pose
            if (fcl_obj := self.convert_sapien_actor(actor)) is not None:  # type: ignore
                self.add_object(fcl_obj)

    def update_from_simulation(self, *, update_attached_object: bool = True) -> None:
        """
        Updates PlanningWorld's articulations/objects pose with current Scene state.
        Note that shape's local_pose is not updated.
        If those are changed, please recreate a new SapienPlanningWorld instance.

        :param update_attached_object: whether to update the attached pose of
            all attached objects
        """
        for articulation in self._sim_scene.get_all_articulations():
            if art := self.get_articulation(convert_object_name(articulation)):
                # set_qpos to update poses
                art.set_qpos(articulation.qpos)  # type: ignore
            else:
                raise RuntimeError(
                    f"Articulation {articulation.name} not found in PlanningWorld! "
                    "The scene might have changed since last update."
                )

        for ActorBase in self._sim_scene.get_all_actors():
            object_name = convert_object_name(ActorBase)

            # If ActorBase is an attached object
            if attached_body := self.get_attached_object(object_name):
                if update_attached_object:  # update attached pose
                    attached_body.pose = (
                        attached_body.get_attached_link_global_pose().inv()
                        * ActorBase.pose  # type: ignore
                    )
                attached_body.update_pose()
            elif fcl_obj := self.get_object(object_name):
                # Overwrite the object
                self.add_object(
                    FCLObject(
                        object_name,
                        ActorBase.pose,  # type: ignore
                        fcl_obj.shapes,
                        fcl_obj.shape_poses,
                    )
                )
            elif (
                len(
                    ActorBase.get_collision_shapes()  # type: ignore
                )
                > 0
            ):
                raise RuntimeError(
                    f"ActorBase {ActorBase.name} not found in PlanningWorld! "
                    "The scene might have changed since last update."
                )

    def check_collision_between(
        self,
        obj_A: ArticulationBase | ActorBase,
        obj_B: ArticulationBase | ActorBase,
        *,
        acm: AllowedCollisionMatrix = AllowedCollisionMatrix(),  # noqa: B008
    ) -> list[WorldCollisionResult]:
        """
        Check collision between two objects,
        which can either be a ArticulationBase or an ActorBase.

        :param obj_A: object A to check for collision.
        :param obj_B: object B to check for collision.
        :param acm: allowed collision matrix.
        :return: a list of WorldCollisionResult. Empty if there's no collision.
        """
        col_obj_A = self._get_collision_obj(obj_A)
        col_obj_B = self._get_collision_obj(obj_B)

        if isinstance(obj_A, ArticulationBase):  # A is articulation, B is anything
            assert isinstance(col_obj_A, FCLModel), f"Wrong type: {type(col_obj_A)}"
            return col_obj_A.check_collision_with(col_obj_B, acm=acm)
        elif isinstance(obj_B, ArticulationBase):  # A is object, B is articulation
            assert isinstance(col_obj_B, FCLModel), f"Wrong type: {type(col_obj_B)}"
            return col_obj_B.check_collision_with(col_obj_A, acm=acm)
        elif isinstance(obj_B, ActorBase):  # A is object, B is object
            assert isinstance(col_obj_A, FCLObject) and isinstance(
                col_obj_B, FCLObject
            ), f"Wrong type: col_obj_A={type(col_obj_A)}, col_obj_B={type(col_obj_B)}"
            if (
                acm_type := acm.get_allowed_collision(col_obj_A.name, col_obj_B.name)
            ) is None or acm_type == AllowedCollision.NEVER:
                result = collide(col_obj_A, col_obj_B)
                if result.is_collision():
                    return [
                        WorldCollisionResult(
                            result,
                            "object_object",
                            col_obj_A.name,
                            col_obj_B.name,
                            col_obj_A.name,
                            col_obj_B.name,
                        )
                    ]
            return []
        else:
            raise NotImplementedError(f"obj_A={obj_A}, obj_B={obj_B}")

    def distance_between(
        self,
        obj_A: ArticulationBase | ActorBase,
        obj_B: ArticulationBase | ActorBase,
        *,
        acm: AllowedCollisionMatrix = AllowedCollisionMatrix(),  # noqa: B008
        return_distance_only: bool = True,
    ) -> WorldDistanceResult | float:
        """
        Check distance-to-collision between two objects,
        which can either be a ArticulationBase or an ActorBase.

        :param obj_A: object A to check for collision.
        :param obj_B: object B to check for collision.
        :param acm: allowed collision matrix.
        :param return_distance_only: if True, return distance only.
        :return: a WorldDistanceResult or a float if return_distance_only==True.
        """
        col_obj_A = self._get_collision_obj(obj_A)
        col_obj_B = self._get_collision_obj(obj_B)
        ret = WorldDistanceResult()

        if isinstance(obj_A, ArticulationBase):  # A is articulation, B is anything
            assert isinstance(col_obj_A, FCLModel), f"Wrong type: {type(col_obj_A)}"
            ret = col_obj_A.distance_with(col_obj_B, acm=acm)
        elif isinstance(obj_B, ArticulationBase):  # A is object, B is articulation
            assert isinstance(col_obj_B, FCLModel), f"Wrong type: {type(col_obj_B)}"
            ret = col_obj_B.distance_with(col_obj_A, acm=acm)
        elif isinstance(obj_B, ActorBase):  # A is object, B is object
            assert isinstance(col_obj_A, FCLObject) and isinstance(
                col_obj_B, FCLObject
            ), f"Wrong type: col_obj_A={type(col_obj_A)}, col_obj_B={type(col_obj_B)}"
            if (
                acm_type := acm.get_allowed_collision(col_obj_A.name, col_obj_B.name)
            ) is None or acm_type == AllowedCollision.NEVER:
                result = distance(col_obj_A, col_obj_B)
                ret = WorldDistanceResult(
                    result,
                    result.min_distance,
                    "object_object",
                    col_obj_A.name,
                    col_obj_B.name,
                    col_obj_A.name,
                    col_obj_B.name,
                )
        else:
            raise NotImplementedError(f"obj_A={obj_A}, obj_B={obj_B}")

        return ret.min_distance if return_distance_only else ret

    def _get_collision_obj(
        self,
        obj: ArticulationBase | ActorBase,
    ) -> FCLModel | FCLObject | None:
        """Helper function to get mplib collision object from sapien object"""
        if isinstance(obj, ArticulationBase) and (
            articulation := self.get_articulation(convert_object_name(obj))
        ):
            return articulation.get_fcl_model()
        elif isinstance(obj, ActorBase) and (
            fcl_obj := self.get_object(convert_object_name(obj))
        ):
            return fcl_obj
        else:
            raise RuntimeError(
                f"Unknown SAPIEN object type: {type(obj)} or "
                f"Object {obj.name} not found in PlanningWorld "
                "(The scene might have changed since last update)"
            )

    @staticmethod
    def convert_sapien_actor(actor: ActorBase) -> FCLObject | None:
        """
        Converts a SAPIEN Actor Base to an FCLObject.
        All shapes in the returned FCLObject are already set at their world poses.

        :param actor: a SAPIEN ActorBase.
        :return: an FCLObject containing all collision shapes in the Physx component.
            If the component has no collision shapes, return ``None``.
        """
        shapes: list[CollisionObject] = []
        shape_poses: list[Pose] = []
        for shape in actor.get_collision_shapes():
            shape_poses.append(shape.get_local_pose())
            geometry = shape.geometry
            if isinstance(geometry, BoxGeometry):
                geometry = cast(BoxGeometry, geometry)
                c_geom = Box(side=geometry.half_lengths * 2)
            elif isinstance(geometry, CapsuleGeometry):
                geometry = cast(CapsuleGeometry, geometry)
                c_geom = Capsule(radius=geometry.radius, lz=geometry.half_length * 2)
                # NOTE: physx Capsule has x-axis along capsule height
                # FCL Capsule has z-axis along capsule height
                shape_poses[-1] *= Pose(q=euler2quat(0, np.pi / 2, 0))
            elif isinstance(geometry, ConvexMeshGeometry):
                geometry = cast(ConvexMeshGeometry, geometry)
                vertices = geometry.vertices * geometry.scale  # [n, 3]
                faces = geometry.indices.reshape(-1, 3)  # [m * 3]
                c_geom = Convex(vertices=vertices, faces=faces)
            elif isinstance(geometry, PlaneGeometry):
                # PhysxCollisionShapePlane are actually a halfspace
                # https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/Geometry.html#planes
                # PxPlane's Pose determines its normal and offert (normal is +x)
                n = shape_poses[-1].to_transformation_matrix()[:3, 0]
                d = n.dot(shape_poses[-1].p)
                c_geom = Halfspace(n=n, d=d)
                shape_poses[-1] = Pose()
            elif isinstance(geometry, SphereGeometry):
                geometry = cast(SphereGeometry, geometry)
                c_geom = Sphere(radius=geometry.radius)
            elif isinstance(geometry, NonconvexMeshGeometry):
                geometry = cast(NonconvexMeshGeometry, geometry)
                vertices = geometry.vertices * geometry.scale  # [n, 3]
                faces = geometry.indices.reshape(-1, 3)  # [m * 3]
                c_geom = BVHModel()
                c_geom.begin_model()
                c_geom.add_sub_model(vertices=vertices, faces=faces)  # type: ignore
                c_geom.end_model()
            else:
                raise TypeError(f"Unknown shape type: {type(shape)}")
            shapes.append(CollisionObject(c_geom))

        if len(shapes) == 0:
            return None

        return FCLObject(
            actor.name
            if isinstance(actor, LinkBase)
            else convert_object_name(actor),
            actor.pose,  # type: ignore
            shapes,
            shape_poses,
        )


class SapienPlanner(Planner):
    def __init__(
        self,
        sapien_planning_world: SapienPlanningWorld,
        move_group: str,
        *,
        joint_vel_limits: Optional[Sequence[float] | np.ndarray] = None,
        joint_acc_limits: Optional[Sequence[float] | np.ndarray] = None,
    ):
        """
        Creates an mplib.planner.Planner from a SapienPlanningWorld.

        :param sapien_planning_world: a SapienPlanningWorld created from sapien.Scene
        :param move_group: name of the move group (end effector link)
        :param joint_vel_limits: joint velocity limits for time parameterization
        :param joint_acc_limits: joint acceleration limits for time parameterization
        """
        self.planning_world = sapien_planning_world
        self.acm = self.planning_world.get_allowed_collision_matrix()

        if len(planned_arts := self.planning_world.get_planned_articulations()) != 1:
            raise NotImplementedError("Must have exactly one planned articulation")
        self.robot = planned_arts[0]
        self.pinocchio_model = self.robot.get_pinocchio_model()
        self.user_link_names = self.pinocchio_model.get_link_names()
        self.user_joint_names = self.pinocchio_model.get_joint_names()

        self.joint_name_2_idx = {}
        for i, joint in enumerate(self.user_joint_names):
            self.joint_name_2_idx[joint] = i
        self.link_name_2_idx = {}
        for i, link in enumerate(self.user_link_names):
            self.link_name_2_idx[link] = i

        assert (
            move_group in self.user_link_names
        ), f"end-effector not found as one of the links in {self.user_link_names}"
        self.move_group = move_group
        self.robot.set_move_group(self.move_group)
        self.move_group_joint_indices = self.robot.get_move_group_joint_indices()

        self.joint_types = self.pinocchio_model.get_joint_types()
        self.joint_limits = np.concatenate(self.pinocchio_model.get_joint_limits())
        if joint_vel_limits is None:
            joint_vel_limits = np.ones(len(self.move_group_joint_indices))
        if joint_acc_limits is None:
            joint_acc_limits = np.ones(len(self.move_group_joint_indices))
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.move_group_link_id = self.link_name_2_idx[self.move_group]

        assert (
            len(self.joint_vel_limits)
            == len(self.joint_acc_limits)
            == len(self.move_group_joint_indices)
            <= len(self.joint_limits)
        ), (
            "length of joint_vel_limits, joint_acc_limits, and move_group_joint_indices"
            " should equal and be <= number of total joints. "
            f"{len(self.joint_vel_limits)} == {len(self.joint_acc_limits)} "
            f"== {len(self.move_group_joint_indices)} <= {len(self.joint_limits)}"
        )

        # Mask for joints that have equivalent values (revolute joints with range > 2pi)
        self.equiv_joint_mask = [
            t.startswith("JointModelR") for t in self.joint_types
        ] & (self.joint_limits[:, 1] - self.joint_limits[:, 0] > 2 * np.pi)

        self.planner = OMPLPlanner(world=self.planning_world)

    def update_from_simulation(self, *, update_attached_object: bool = True) -> None:
        """
        Updates PlanningWorld's articulations/objects pose with current Scene state.
        Note that shape's local_pose is not updated.
        If those are changed, please recreate a new SapienPlanningWorld instance.

        Directly calls ``SapienPlanningWorld.update_from_simulation()``

        :param update_attached_object: whether to update the attached pose of
            all attached objects
        """
        self.planning_world.update_from_simulation(
            update_attached_object=update_attached_object
        )
