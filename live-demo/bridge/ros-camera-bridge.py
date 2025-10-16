import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class ROSCameraBridge(Node):
    """Listens to RealSense topics and provides latest RGB + Depth frames as numpy arrays."""

    def __init__(self):
        super().__init__('ros_camera_bridge')
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        self.create_subscription(Image, '/camera/color/image_raw', self._rgb_callback, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self._depth_callback, 10)

    def _rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_image = depth.astype(np.float32)

    def get_latest_frames(self):
        return self.rgb_image, self.depth_image
