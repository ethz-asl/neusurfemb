import cv2
import copy
import os
import numpy as np
import rosbag
import rospy
import yaml

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo


class V2BConverter:

    def __init__(self, input_path, output_file, **kwargs):
        self.bag = None
        self.input_path = input_path
        self.output_file = output_file
        self.output_dir = kwargs.get('output_dir', './')
        self.output_dir_images = os.path.join(self.output_dir, "images")
        self.sleep_rate = float(kwargs.get('sleep_rate', 0.01))
        self.div_num = int(kwargs.get('div_num', 2))
        self.calibration_file = kwargs.get('calibration_file')
        self.save_images = kwargs.get('save_images')
        self.verbose = kwargs.get('verbose')

        if (self.calibration_file is not None):
            # Read camera info from calibration file.
            with open(self.calibration_file, "r") as f:
                calib_data = yaml.load(f, Loader=yaml.SafeLoader)
            self._camera_info_template_msg = CameraInfo()
            W, H = calib_data["cam0"]["resolution"]
            self._camera_info_template_msg.width = W
            self._camera_info_template_msg.height = H

            K = np.eye(3)
            intrinsics = calib_data["cam0"]["intrinsics"]
            K[0, 0] = intrinsics[0]
            K[1, 1] = intrinsics[1]
            K[0, 2] = intrinsics[2]
            K[1, 2] = intrinsics[3]
            self._camera_info_template_msg.K = K.reshape(-1).tolist()
            self._camera_info_template_msg.D = calib_data["cam0"][
                "distortion_coeffs"]
            self._camera_info_template_msg.distortion_model = calib_data[
                "cam0"]["distortion_model"]
            self._camera_info_template_msg.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            P = np.zeros([3, 4])
            P[:3, :3] = K.copy()
            self._camera_info_template_msg.P = P.reshape(-1).tolist()

        self._image_idx = 0

    @staticmethod
    def open_output_dir(output_dir, verbose):
        try:
            os.makedirs(output_dir)
            if (verbose):
                print("Directory ", output_dir, " Created")
        except FileExistsError:
            print("Directory ", output_dir, " already exists")

    def open_bag_file(self):
        try:
            self.bag = rosbag.Bag(
                os.path.join(self.output_dir, self.output_file), 'w')
        except Exception as e:
            print(e)

    def write_image(self, image, time):
        bridge = CvBridge()

        try:
            # Add image message.
            image_message = bridge.cv2_to_imgmsg(image, encoding="bgr8")
            if (self.save_images):
                cv2.imwrite(
                    os.path.join(self.output_dir_images,
                                 f"{self._image_idx:06d}.png"), image)
            image_message.header.stamp = time
            self.bag.write('/camera/image_raw', image_message, t=time)

            if (self.calibration_file is not None):
                # Add camera info message.
                camera_info_message = copy.deepcopy(
                    self._camera_info_template_msg)
                camera_info_message.header.stamp = time
                self.bag.write('/camera/camera_info',
                               camera_info_message,
                               t=time)
        except Exception as e:
            print(e)

        self._image_idx += 1

    def convert(self):
        cap = cv2.VideoCapture(self.input_path)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0.0)

        orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))

        assert (orientation in [0, 90, 180, 270])

        self.open_output_dir(self.output_dir, verbose=self.verbose)
        if (self.save_images):
            self.open_output_dir(self.output_dir_images, verbose=self.verbose)
        self.open_bag_file()

        i, count = 0, 0

        curr_time = rospy.Time()

        while cap.isOpened():
            ret, frame = cap.read()

            if (ret is False):
                break

            if (orientation == 90):
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif (orientation == 180):
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif (orientation == 270):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if (i % self.div_num == 0):
                self.write_image(frame, time=curr_time)
                if (self.verbose and self.save_images):
                    print("Wrote extracted_frame_" + str(count) + '.jpg' +
                          "\n")
                count += 1
                curr_time = rospy.Time.from_sec(curr_time.to_sec() +
                                                self.sleep_rate)
            i += 1

        print("Total {} of frames were extracted.".format(count))
        cap.release()
        cv2.destroyAllWindows()
