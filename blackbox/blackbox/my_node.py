import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan, Imu, MagneticField
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.encoders import ScalarEncoder
from htm.bindings.engine_internal import Network


class HTMNode(Node):
    def __init__(self):
        super().__init__('htm_node')

        # Create HTM network
        self.network = Network()

        accel_encoder_params = {"n": 500, "w": 21, "minval": -200, "maxval": 200, "periodic": False}
        self.accel_encoders = [self.network.addRegion("accel_x_encoder", "ScalarEncoderRegion", str(accel_encoder_params)),
                              self.network.addRegion("accel_y_encoder", "ScalarEncoderRegion", str(accel_encoder_params)),
                              self.network.addRegion("accel_z_encoder", "ScalarEncoderRegion", str(accel_encoder_params))]

        gyro_encoder_params = {"resolution": 0.01, "minval": -1, "maxval": 1, "periodic": False, "activeBits": 2}
        self.gyro_encoders = [self.network.addRegion("gyro_x_encoder", "ScalarEncoderRegion", str(gyro_encoder_params)),
                              self.network.addRegion("gyro_y_encoder", "ScalarEncoderRegion", str(gyro_encoder_params)),
                              self.network.addRegion("gyro_z_encoder", "ScalarEncoderRegion", str(gyro_encoder_params))]

        joint_encoder_params = {"resolution": 0.1, "minval": -3.14159, "maxval": 3.14159, "periodic": False, "activeBits": 2}
        self.joint_encoders = [self.network.addRegion("joint_A_encoder", "ScalarEncoderRegion", str(joint_encoder_params)),
                               self.network.addRegion("joint_B_encoder", "ScalarEncoderRegion", str(joint_encoder_params))]

        range_encoder_params = {"resolution": 1, "minval": 34, "maxval": 8191, "periodic": False, "activeBits": 10}
        self.range_encoders = [self.network.addRegion("range_r_encoder", "ScalarEncoderRegion", str(range_encoder_params)),
                               self.network.addRegion("rnge_l_encoder", "ScalarEncoderRegion", str(range_encoder_params))]

        # # Spatial Pooler
        # enc_output_length = self.joint_encoders[0].getWidth() + self.joint_encoders[1].getWidth() +\
        #     self.accel_encoders[0].getWidth() + self.accel_encoders[1].getWidth() + self.accel_encoders[2].getWidth() +\
        #     self.gyro_encoders[0].getWidth() + self.gyro_encoders[1].getWidth() + self.gyro_encoders[2].getWidth() +\
        #     self.range_encoders[0].getWidth() + self.range_encoders[1].getWidth()
        
        sp_params = {
            "inputDimensions": (100,),
            "columnDimensions": (2048,),
            "potentialRadius": 50,
            "numActiveColumnsPerInhArea": 40,
            "globalInhibition": True,
            "synPermInactiveDec": 0.008,
            "synPermActiveInc": 0.05,
            "synPermConnected": 0.1,
            "boostStrength": 3.0,
            "seed": 1960
        }

        self.sp_region = self.network.addRegion("sp", "SPRegion", str(sp_params))
        self.network.link("accel_x_encoder", "sp", "UniformLink", "")
        self.network.link("accel_y_encoder", "sp", "UniformLink", "")
        self.network.link("accel_z_encoder", "sp", "UniformLink", "")

        # Temporal Memory
        tm_params = {
            "columnCount": 2048,
            "cellsPerColumn": 32,
            "inputDimensions": (2048,),
            "maxSynapsesPerSegment": 32,
            "maxSegmentsPerCell": 128,
            "initialPerm": 0.21,
            "permanenceInc": 0.1,
            "permanenceDec": 0.1,
            "globalDecay": 0.0,
            "maxAge": 0,
            "minThreshold": 9,
            "activationThreshold": 12,
            "outputType": "normal",
            "pamLength": 1,
        }
        self.tm_region = self.network.addRegion("tm", "TMRegion", str(tm_params))
        self.network.link("sp", "tm", "UniformLink", "")

        # ROS2 Subscribers and callbacks
        self.jointstate_sub = self.create_subscription(
            JointState, 'joint_states', self.jointstate_callback, 10)
        self.laserscan_sub = self.create_subscription(
            LaserScan, 'scan', self.laserscan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu', self.imu_callback, 10)
        self.magnet_sub = self.create_subscription(
            MagneticField, 'magnetic_field', self.magnet_callback, 10)

    def jointstate_callback(self, msg):
        pos_sdr = self.joint_encoders[0].encode(msg.position[0])
        pos2_sdr = self.joint_encoders[1].encode(msg.position[1])

        # Encode the joint state data to an SDR
        joint_sdr = SDR( self.joint_encoders[0].getWidth() + self.joint_encoders[1].getWidth())
        joint_sdr.combine(pos_sdr)
        joint_sdr.combine(pos2_sdr)

    def laserscan_callback(self, msg):
        # Get the distance from the laser scan
        dist = msg.ranges[0]

        # Encode the distance to an SDR
        range_sdr = SDR(self.range_encoders[0].getWidth())
        range_sdr = self.range_encoders[0].encode(dist)

    def imu_callback(self, msg):
        sdr_x = self.accel_encoders[0].encode(msg.linear_acceleration.x)
        sdr_y = self.accel_encoders[1].encode(msg.linear_acceleration.y)
        sdr_z = self.accel_encoders[2].encode(msg.linear_acceleration.z)

        gyro_x_sdr = self.gyro_encoders[0].encode(msg.angular_velocity.x)
        gyro_y_sdr = self.gyro_encoders[1].encode(msg.angular_velocity.y)
        gyro_z_sdr = self.gyro_encoders[2].encode(msg.angular_velocity.z)

        # Combine the encoded values into a single SDR
        imu_sdr = SDR(sdr_x.size + sdr_y.size + sdr_z.size + gyro_x_sdr.size + gyro_y_sdr.size + gyro_z_sdr.size)
        imu_sdr.combine(sdr_x)
        imu_sdr.combine(sdr_y)
        imu_sdr.combine(sdr_z)
        imu_sdr.combine(gyro_x_sdr)
        imu_sdr.combine(gyro_y_sdr)
        imu_sdr.combine(gyro_z_sdr)

    def magnet_callback(self, msg):
        magnet_sdr = self.accel_encoders[0].encode(msg.magnetic_field.x)

def main(args=None):
    rclpy.init(args=args)
    htm_node = HTMNode()
    rclpy.spin(htm_node)

if __name__ == "__main__":
    main()
