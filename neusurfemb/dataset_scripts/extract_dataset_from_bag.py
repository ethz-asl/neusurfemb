import argparse
import cv2
import numpy as np
import os
import tqdm

from cv_bridge import CvBridge
import rosbag

parser = argparse.ArgumentParser()
parser.add_argument('--bag-path', required=True, type=str)
parser.add_argument('--output-folder', required=True, type=str)
parser.add_argument('--freq', type=int, default=10)
args = parser.parse_args()

bag_path = args.bag_path
output_folder = args.output_folder

bag = rosbag.Bag(bag_path)

os.makedirs(output_folder, exist_ok=False)
rgb_output_folder = os.path.join(output_folder, "rgb")
raw_rgb_output_folder = os.path.join(output_folder, "raw_rgb")

os.makedirs(rgb_output_folder, exist_ok=False)
os.makedirs(raw_rgb_output_folder, exist_ok=False)

bridge = CvBridge()

img_count = {'rect': 0, 'raw': 0}

image_rect_topic_name = '/output/image'
camera_info_topic_name = '/output/camera_info'

frame_idx = -1
num_messages = {
    'rect': bag.get_message_count(image_rect_topic_name),
    'raw': bag.get_message_count('/camera/image_raw')
}

assert (abs(num_messages['raw'] - num_messages['rect']) <= 2)

# Read images.
for topic, msg, t in tqdm.tqdm(
        bag.read_messages(topics=[image_rect_topic_name, '/camera/image_raw']),
        total=num_messages['rect'] * 2):
    if (topic == '/camera/image_raw'):
        frame_idx += 1
    if (frame_idx % args.freq != 0):
        continue

    rgb_path = os.path.join(rgb_output_folder, f"{img_count['rect']:06d}.png")
    raw_rgb_path = os.path.join(raw_rgb_output_folder,
                                f"{img_count['raw']:06d}.png")

    if (topic == image_rect_topic_name):
        rgb = bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imwrite(filename=rgb_path, img=rgb)
        img_count['rect'] += 1
    elif (topic == '/camera/image_raw'):
        raw_rgb = bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imwrite(filename=raw_rgb_path, img=raw_rgb)
        img_count['raw'] += 1

# Read intrinsics.
for intrinsics_msg in bag.read_messages(topics=[camera_info_topic_name]):
    break
K = np.array(intrinsics_msg.message.K).reshape(3, 3)
D = np.array(intrinsics_msg.message.D)
np.savetxt(fname=os.path.join(output_folder, "intrinsics.txt"), X=K)
np.savetxt(fname=os.path.join(output_folder, "distortion_parameters.txt"), X=D)

bag.close()
