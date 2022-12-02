import glob
import os
import sys

import carla
import random
import time
import numpy as np
import cv2
from easydict import EasyDict

# Append Carla path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class carla_env():
    def __init__(self):
        # Set the clients and timeouts
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        # Get the world
        self.world = self.client.get_world()

        # Get the blueprint library
        self.blueprint_lib = self.world.get_blueprint_library()

        # Get the ego vehicle blueprint
        self.ego_bp = self.blueprint_lib.find('vehicle.tesla.model3')
        self.ego_bp.set_attribute('role_name', 'ego')

        # Set image parameters
        self.im_width = 640
        self.im_height = 480

        # Get sensors blueprints
        self.cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.cam_bp.set_attribute("image_size_x", f"{self.im_width}")
        self.cam_bp.set_attribute("image_size_y", f"{self.im_height}")
        self.cam_bp.set_attribute("fov", "105")

        # Get collision blueprints
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Set tracking variables
        self.collision_history = []

        # Keep track of all actors to close them later
        self.actors = []


    def reset(self):
        # Reset the ego_vehicle
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle = self.world.spawn_actor(self.ego_bp, transform)
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # time.sleep(4)
        self.actors.append(self.ego_vehicle)

        # Reset the camera image
        transform = carla.Transform(carla.Location(2,0,1), carla.Rotation(0,0,0))
        self.front_camera = self.world.spawn_actor(self.cam_bp, transform, attach_to=self.ego_vehicle,
                                                   attachment_type=carla.AttachmentType.Rigid)
        self.actors.append(self.front_camera)

        # Reset the collision handler
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, transform, attach_to=self.ego_vehicle,
                                                       attachment_tupe=carla.AttachmentType.Rigid)
        self.actors.append(self.collision_sensor)

        # Start listening to the sensor data
        self.front_camera.listen(lambda data: self._img_callback(data))
        self.collision_sensor.listen(lambda  data: self._collision_callback(data))

        state = self.front_camera
        return state, None

    def step(self, action):

        next_state = None
        reward = None
        done = None
        info = None
        return next_state, reward, done, info, None

    def reward_function(self, action):
        reward = None
        return reward

    def _img_callback(self, image):
        img = np.array(image.raw_data.tolist()).reshape((self.im_height, self.im_width, 4))
        return img/255.0

    def _collision_callback(self, event):
        self.collision_history.append(event)



if __name__ == '__main__':
    pass
