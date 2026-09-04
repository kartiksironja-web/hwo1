import rclpy
from rclpy.node import Node

import numpy as np

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


# ============================================================
# Rotation matrices
# ============================================================

def rot_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rot_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


# ============================================================
# Rotation matrix to quaternion
# ROS quaternion order: x, y, z, w
# ============================================================

def matrix_to_quaternion(R):

    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2

        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:

        S = np.sqrt(
            1.0 + R[0, 0] - R[1, 1] - R[2, 2]
        ) * 2

        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S

    elif R[1, 1] > R[2, 2]:

        S = np.sqrt(
            1.0 + R[1, 1] - R[0, 0] - R[2, 2]
        ) * 2

        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S

    else:

        S = np.sqrt(
            1.0 + R[2, 2] - R[0, 0] - R[1, 1]
        ) * 2

        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return qx, qy, qz, qw


# ============================================================
# ROS NODE
# ============================================================

class RotationDemo(Node):

    def __init__(self):

        super().__init__('rotation_demo')

        self.br = TransformBroadcaster(self)

        # Toggle parameter
        self.declare_parameter('rotation_mode', 'body')

        # Initial orientation
        self.R = np.eye(3)

        self.step = 0

        # Update every 0.05 seconds
        self.timer = self.create_timer(
            0.05,
            self.update
        )

        self.get_logger().info(
            'Rotation demo started!'
        )

        self.get_logger().info(
            'Mode can be: body or fixed'
        )


    def update(self):

        mode = self.get_parameter(
            'rotation_mode'
        ).value


        # Small rotation angle
        angle = np.deg2rad(3)


        # Alternate between X and Y rotations
        if self.step % 2 == 0:

            R_increment = rot_x(angle)

            axis_name = "X"

        else:

            R_increment = rot_y(angle)

            axis_name = "Y"


        # ====================================================
        # BODY / CURRENT FRAME
        #
        # R_new = R_old @ R_increment
        # ====================================================

        if mode == 'body':

            self.R = self.R @ R_increment


        # ====================================================
        # FIXED / SPACE FRAME
        #
        # R_new = R_increment @ R_old
        # ====================================================

        elif mode == 'fixed':

            self.R = R_increment @ self.R


        else:

            self.get_logger().warn(
                'Unknown mode. Use body or fixed.'
            )

            return


        # Convert to quaternion
        qx, qy, qz, qw = matrix_to_quaternion(
            self.R
        )


        # ====================================================
        # Broadcast world -> body
        # ====================================================

        t = TransformStamped()

        t.header.stamp = (
            self.get_clock().now().to_msg()
        )

        t.header.frame_id = 'world'

        t.child_frame_id = 'body'


        # Position
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0


        # Orientation
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw


        self.br.sendTransform(t)


        # Print status occasionally
        if self.step % 40 == 0:

            self.get_logger().info(
                f'Mode: {mode} | Rotation about {axis_name}'
            )


        self.step += 1


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = RotationDemo()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
