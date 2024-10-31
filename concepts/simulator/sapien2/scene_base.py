import numpy as np
from PIL import Image
from sapien.core import Engine, SapienRenderer, CameraEntity, Pose
from sapien.utils import Viewer

from concepts.simulator.sapien2.camera import get_depth_img, get_rgba_img


class SapienSceneBase(object):
    def __init__(
            self,
            fps: float = 240.0,
            **kwargs
    ):
        self.engine = Engine()
        self.renderer = SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.fps = fps
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / fps)
        ambient_light_color = kwargs.get('ambient_light_color', np.array([0.5, 0.5, 0.5]))
        directional_light_direction = kwargs.get('directional_light_direction', np.array([0, 1, -1]))
        directional_light_color = kwargs.get('directional_light_color', np.array([0.5, 0.5, 0.5]))
        self.scene.set_ambient_light(ambient_light_color)
        self.scene.add_directional_light(directional_light_direction, directional_light_color)
        self.viewer = None

    def step(self):
        self.scene.step()

    def update_render(self):
        self.scene.update_render()

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
            name: str,
            pose: Pose,
            width: int = 768,
            height: int = 768,
            fovy: float = np.deg2rad(60),
            near: float = 0.05,
            far: float = 100
    ) -> CameraEntity:
        camera = self.scene.add_camera(
            name=name,
            width=width,
            height=height,
            fovy=fovy,
            near=near,
            far=far
        )
        camera.set_pose(pose)
        return camera

    def get_picture(
            self,
            camera: CameraEntity,
            get_depth: bool = False,
            debug_viewer: bool = False
    ) -> tuple[Image.Image, CameraEntity] or tuple[Image.Image, np.ndarray, CameraEntity]:
        if debug_viewer:
            if self.viewer is None:
                self.create_viewer()
            while not self.viewer.closed:
                self.update_render()
                self.viewer.render()
        self.update_render()
        image = get_rgba_img(camera=camera)
        image = Image.fromarray(image).convert('RGB')
        if debug_viewer:
            image.show()
        if get_depth:
            depth_img = get_depth_img(camera)
            return image, depth_img, camera
        return image, camera