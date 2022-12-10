import glob
import os
import sys

import carla
import random
import time
import numpy as np
import cv2
from collections import deque
from easydict import EasyDict

# Append Carla path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class Carla_Actions():
    def __init__(self):
        super(Carla_Actions, self).__init__()
        self.shape = (3,)

    def sample(self):
        throttle = np.random.uniform(-1,1)
        brake = np.random.uniform(-1,1)
        steer = np.random.uniform(-1,1)
        return np.array([throttle, steer, brake])

class carla_env():
    def __init__(self, motion_batch = 1):
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

        # Keep track of all actors to close them later
        self.actors = []

        # Keep track of steps taken so far
        self.ep_step = 0
        self.max_allowed_steps = 1000

        # Define some constants
        self.action_space = Carla_Actions()
        self.observation_space = np.zeros((480, 640, 3))

        self.curr_image = None
        self.motion_batch = motion_batch
        self.state = deque()
        self._max_episode_steps = 1000

    def reset(self):
        self.close_agents()
        # Reset step counter
        self.ep_step = 0

        # Reset all actors
        self.actors = []

        # Reset the ego_vehicle
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle = self.world.spawn_actor(self.ego_bp, transform)
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
        time.sleep(4)
        self.actors.append(self.ego_vehicle)

        # Reset the camera image
        transform = carla.Transform(carla.Location(2,0,1), carla.Rotation(0,0,0))
        self.front_camera = self.world.spawn_actor(self.cam_bp, transform, attach_to=self.ego_vehicle,
                                                   attachment_type=carla.AttachmentType.Rigid)
        self.actors.append(self.front_camera)

        # Reset the collision handler
        self.collision_history = []
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, transform, attach_to=self.ego_vehicle,
                                                       attachment_type=carla.AttachmentType.Rigid)
        self.actors.append(self.collision_sensor)

        # Start listening to the sensor data
        self.front_camera.listen(lambda data: self._img_callback(data))
        self.collision_sensor.listen(lambda  data: self._collision_callback(data))

        while len(self.state)!=self.motion_batch:
            time.sleep(0.001)

        return np.array(self.state), None

    def step(self, action):
        # Rescale actions
        action[0] = (action[0] + 1)/2
        action[2] = (action[2] + 1)/2
        action = np.float64(action)
        # Update step counter
        self.ep_step += 1
        # Apply control to the vehicle based on the action
        try:
            # print(type(action))
            # print(action.shape)
            # print(action)
            # print(action.dtype)
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2]))
        except:
            print(type(action))
            print(action.shape)
            print(action)
            print(action.dtype)
            raise Exception

        # get the speed of the vehicle
        v = self.ego_vehicle.get_velocity()
        v = 3.6*np.sqrt(v.x**2 + v.y**2 + v.z**2)

        done = False
        if len(self.collision_history) != 0:
            self.close_agents()
            done = True
            reward = -1
        else:
            reward = -1 + v * 2/100

        if self.ep_step >= self.max_allowed_steps:
            done = True

        return np.array(self.state), reward, done, None, None

    def close_agents(self):
        for actor in self.actors:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

        return None

    def _img_callback(self, image):
        # img = np.array(image.raw_data.tolist()).reshape((self.im_height, self.im_width, 4))
        img = np.array(image.raw_data).reshape((self.im_height, self.im_width, 4))[:,:,:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img/255.0).astype(np.float32)
        self._state_callback(img)

    def _state_callback(self,img):
        while len(self.state)!=self.motion_batch:
            self.state.append(img)

        self.state.popleft()
        self.state.append(img)

    def _collision_callback(self, event):
        # print("Collisioon appended here")
        self.collision_history.append(event)



if __name__ == '__main__':
    pass
