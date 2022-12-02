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

IM_WIDTH = 120
IM_HEIGHT = 90

def process_img(image):
    # print("Raw dta: ",dir(image.raw_data))
    # time.sleep(2000)
    img = np.array(image.raw_data.tolist()).reshape((IM_HEIGHT, IM_WIDTH, 4))
    # print("Reaching here for the ith time")
    # cv2.imshow("Images", img[:,:,3])
    # cv2.waitKey(1)
    cv2.imwrite(f"{image.frame}.jpg", img)
    return img/255.0


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name', 'ego')

    spawn_point = random.choice(world.get_map().get_spawn_points())
    ego_vehicle = world.spawn_actor(ego_bp, spawn_point)

    # Check if everything is working correctly
    ego_vehicle.set_autopilot(True)

    # ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(ego_vehicle)

    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "105")
    spawn_point = carla.Transform(carla.Location(2,0,1), carla.Rotation(0,0,0))

    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=ego_vehicle,
                               attachment_type=carla.AttachmentType.Rigid)
    actor_list.append(sensor)
    sensor.listen(lambda data: process_img(data))
    # sensor.listen(lambda image: image.save_to_disk('tutorial/output/%.6d.jpg' % image.frame))

    time.sleep(5)


finally:
    for actor in actor_list:
        actor.destroy()
    print(f"All actors cleaned up!")

if __name__ == '__main__':
    pass
