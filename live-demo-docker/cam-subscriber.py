import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

TOPIC = '/intel_realsense/d435/depth/image_rect_raw'

class DepthImageSubscriber(Node):
    def __init__(self):
        super().__init__('depth_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            TOPIC,
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info('Depth image subscriber started.')

    def listener_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image (16UC1 expected for depth)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Optional: normalize and display (for visualization only)
            normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = np.uint8(normalized)
            cv2.imshow("Depth Image", depth_display)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = DepthImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
