import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import cv2
import math

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # while self.front_camera is None:
        #     time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None



# IM_WIDTH = 640
# IM_HEIGHT = 480
#
# def process_img(image):
#     # print("Raw dta: ",dir(image.raw_data))
#     # time.sleep(2000)
#     img = np.array(image.raw_data.tolist()).reshape((IM_HEIGHT, IM_WIDTH, 4))
#     # print("Reaching here for the ith time")
#     # cv2.imshow("Images", img[:,:,3])
#     # cv2.waitKey(1)
#     cv2.imwrite(f"imgs/{image.frame}.jpg", img)
#     return img/255.0
#
#
# actor_list = []
# try:
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(2.0)
#
#     world = client.get_world()
#
#     ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
#     ego_bp.set_attribute('role_name', 'ego')
#
#     spawn_point = random.choice(world.get_map().get_spawn_points())
#     ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
#
#     # Check if everything is working correctly
#     ego_vehicle.set_autopilot(True)
#
#     # ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
#     actor_list.append(ego_vehicle)
#
#     cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
#     cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
#     cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
#     cam_bp.set_attribute("fov", "105")
#     spawn_point = carla.Transform(carla.Location(2,0,1), carla.Rotation(0,0,0))
#
#     sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=ego_vehicle,
#                                attachment_type=carla.AttachmentType.Rigid)
#     actor_list.append(sensor)
#     sensor.listen(lambda data: process_img(data))
#     # sensor.listen(lambda image: image.save_to_disk('tutorial/output/%.6d.jpg' % image.frame))
#
#     time.sleep(5)
#
#
# finally:
#     for actor in actor_list:
#         actor.destroy()
#     print(f"All actors cleaned up!")
#
# if __name__ == '__main__':
#     pass
