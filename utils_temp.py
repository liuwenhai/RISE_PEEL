import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as Rot 

from utils.transformation import rotation_transform,xyz_rot_transform


import rospy
from sensor_msgs.msg import PointCloud2, JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped,WrenchStamped,Vector3
from std_msgs.msg import Header

from inverse_kinematics.pinocchio_model import RobotModel


L515_2_BASE = np.array([[1,0,0,0],
                        [0,-np.sin(70 / 180 * np.pi),np.cos(70 / 180 * np.pi),0],
                        [0,-np.cos(70 / 180 * np.pi),-np.sin(70 / 180 * np.pi),0.59],
                        [0,0,0,1]])

class RealData():
    def __init__(self, path):
        self.path = path
        self.depth_scale = np.loadtxt(os.path.join(path, "depth_scale.txt")).item()
        self.intrinsics = np.loadtxt(os.path.join(path, "intrinsics.txt"))
        cx, cy, fx, fy = self.intrinsics
        self.k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    def __len__(self):
        return len(os.listdir(os.path.join(self.path, 'color')))
    
    def __getitem__(self, idx):
        color = cv2.imread(os.path.join(self.path, 'color', f'{str(idx).zfill(16)}.png'), cv2.IMREAD_COLOR).astype(np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(self.path, 'depth', f'{str(idx).zfill(16)}.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) # (H, W), uint16
        depth = depth * self.depth_scale
        return color, depth
    
    def get_item(self, idx):
        if idx>=0 and idx<len(self):
            return self[idx]
        else:
            return None
    @property
    def ready_rot_6d(self):
        return np.array([-1, 0, 0, 0, 1, 0])


class Agent():
    def __init__(self):
        rospy.init_node("agent")
        self.pc_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

        self.wrench_pub = rospy.Publisher('/wrench', WrenchStamped, queue_size=10)

        self.wrench_stamped_msg = WrenchStamped()
        self.wrench_stamped_msg.header.frame_id = "ft_peel"
        
        self.joint_state = JointState()
        self.joint_state.header = Header()
        self.joint_state.name = ['joint1','joint2', 'joint3', 'joint4', 
                                 'joint5', 'joint6', 'joint7', 'left_finger_joint', 
                                 'right_finger_joint', 'ft_joint1','ft_joint2', 'ft_joint3', 
                                 'ft_joint4', 'ft_joint5', 'ft_joint6', 'ft_joint7']
        
        self.init_ros_pose()
        self.init_ik_robot()
    
    def init_ros_pose(self,base_x = 1.,base_y = 0.4):
        base_x = 0.85
        base_y = 0.22
        # pub l515 coord pose
        self.br_world2l515 = TransformBroadcaster()
        self.l515_pose = TransformStamped()
        self.l515_pose.header.frame_id = 'world'
        self.l515_pose.child_frame_id = 'l515'
        quat = Rot.from_matrix(L515_2_BASE[:3,:3]).as_quat()
        self.l515_pose.transform.translation.x = L515_2_BASE[0,3]
        self.l515_pose.transform.translation.y = L515_2_BASE[1,3]
        self.l515_pose.transform.translation.z = L515_2_BASE[2,3]
        self.l515_pose.transform.rotation.x = quat[0]
        self.l515_pose.transform.rotation.y = quat[1]
        self.l515_pose.transform.rotation.z = quat[2]
        self.l515_pose.transform.rotation.w = quat[3]
        
        # pub robot base coord pose
        self.br_world2robot = TransformBroadcaster()
        self.robot_pose = TransformStamped()
        self.robot_pose.header.frame_id = 'world'
        self.robot_pose.child_frame_id = 'ft_base_link'
        self.robot_pose.transform.translation.x = -base_x
        self.robot_pose.transform.translation.y = base_y
        self.robot_pose.transform.translation.z = 0
        self.robot_pose.transform.rotation.x = 0
        self.robot_pose.transform.rotation.y = 0
        self.robot_pose.transform.rotation.z = 0
        self.robot_pose.transform.rotation.w = 1
        
        self.WORLD2FT_BASE = np.eye(4)
        self.WORLD2FT_BASE[0,3] = -base_x
        self.WORLD2FT_BASE[1,3] = base_y
        self.WORLD2FT_BASE[2,3] = 0
        
        self.WORLD2GRIPPER_BASE = np.eye(4)
        self.WORLD2GRIPPER_BASE[0,3] = base_x
        self.WORLD2GRIPPER_BASE[1,3] = base_y
        self.WORLD2GRIPPER_BASE[2,3] = 0
        self.WORLD2GRIPPER_BASE[:3,:3] = Rot.from_euler('XYZ',[0,0,np.pi]).as_matrix()
        
        
    
    def init_ik_robot(self):
        ft_robot_ik_path = '/home/wenhai/my_code/ft_collector/models/flexiv/rizon4_ft.urdf'
        gripper_robot_ik_path = '/home/wenhai/my_code/ft_collector/models/flexiv/rizon4_gripper_ik.urdf'
        self.ft_robot = RobotModel(ft_robot_ik_path, "ft_peel")
        self.gripper_robot = RobotModel(gripper_robot_ik_path, "tcp")
        
        self.ft_rest_pose = np.array([0,0,0.,np.pi/2,0,0,0])
        self.gripper_rest_pose = np.array([0,0,0.,np.pi/2,0,0,0])
    
    def update_robot_state(self, step_action, cloud):
        gripper_width = step_action[18]
        ft_pose = np.identity(4)
        ft_pose[:3,3] = step_action[:3]
        # ft_rot = step_action[3:9]
        # ft_pose[:3,:3] = rotation_transform(ft_rot,from_rep = "rotation_6d",to_rep = "matrix")
        ft_pose = xyz_rot_transform(step_action[:9], from_rep = "rotation_6d", to_rep = "matrix")
        # ft_pose[:3,:3] = Rot.from_quat(step_action[3:7]).as_matrix()
        gri_pose = np.identity(4)
        # gri_pose[:3,3] = step_action[7:10]
        # gri_rot = step_action[12:18]
        gri_pose[:3, 3] = step_action[9:12]
        gri_rot = step_action[12:18]
        gri_pose[:3,:3] = rotation_transform(gri_rot,from_rep = "rotation_6d",to_rep = "matrix")
        # gri_pose[:3,:3] = Rot.from_quat(step_action[10:14]).as_matrix()

        ft_pose_in_robot = np.linalg.inv(self.WORLD2FT_BASE) @ L515_2_BASE @ ft_pose
        gripper_pose_in_robot = np.linalg.inv(self.WORLD2GRIPPER_BASE) @ L515_2_BASE @ gri_pose
        
        ft_joint = get_ik_joint(self.ft_robot, ft_pose_in_robot, self.ft_rest_pose)
        gripper_joint = get_ik_joint(self.gripper_robot, gripper_pose_in_robot, self.gripper_rest_pose)
        self.ft_rest_pose = ft_joint
        self.gripper_rest_pose = gripper_joint
        
        gripper_percent = get_joint_percent(gripper_joint, self.gripper_robot.join_limits_low, self.gripper_robot.join_limits_high)
        ft_percent = get_joint_percent(ft_joint, self.ft_robot.join_limits_low, self.ft_robot.join_limits_high)
        print('gripper percent: ', np.round(gripper_percent, 3).tolist())
        print('ft percent: ', np.round(ft_percent, 3).tolist())
        print('....................................')
        
        now = rospy.Time.now()
        self.l515_pose.header.stamp = now
        self.br_world2l515.sendTransform(self.l515_pose)
        self.robot_pose.header.stamp = now
        self.br_world2robot.sendTransform(self.robot_pose)
        
        
        self.joint_state.position = gripper_joint.tolist() + [gripper_width/2,gripper_width/2] + ft_joint.tolist()
        self.joint_state.velocity = []
        self.joint_state.effort = []
        self.joint_state.header.stamp = now
        self.joint_pub.publish(self.joint_state)
        
        obs_pc_xyz_obs, obs_pc_rgb_obs = cloud[:,:3], cloud[:,3:]
        base_pc = obs_pc_xyz_obs @ L515_2_BASE[:3,:3].T + L515_2_BASE[:3,3]
        index = (base_pc[:,0] < 0.47) & (base_pc[:,0] > -0.47) & (base_pc[:,1] < 0.8) & (base_pc[:,2] < 0.4) & (base_pc[:,2] > -0.01)
        index = np.random.choice(np.where(index)[0],int(index.sum()*0.5),replace=False)
        
        cloud_msg = create_point_cloud_msg(np.concatenate([obs_pc_xyz_obs[index], obs_pc_rgb_obs[index]*255], axis=-1), now, frame_id='l515')
        self.pc_pub.publish(cloud_msg)

        if len(step_action) > 19:
            wrench = step_action[19:]
            self.wrench_stamped_msg.wrench.force = Vector3(wrench[0], wrench[1], wrench[2])
            self.wrench_stamped_msg.wrench.torque = Vector3(wrench[3], wrench[4], wrench[5])
            self.wrench_stamped_msg.header.stamp = now
            self.wrench_pub.publish(self.wrench_stamped_msg)
        
        

def get_joint_percent(joint, min_joint, max_joint):
    return (joint - min_joint) / (max_joint - min_joint)

def get_ik_joint(robotmodel, pose, rest_joint):
    link_postion = pose[:3,3:].astype(np.float64)
    link_rotation = pose[:3,:3].astype(np.float64)
    link_rotation = Rot.from_matrix(link_rotation).as_quat()
    link_rotation = np.array([link_rotation[3], link_rotation[0], link_rotation[1], link_rotation[2]])
    ik_joint = robotmodel.inverse_kinematics(link_postion, link_rotation.astype(np.float64)[:,np.newaxis], rest_pose=rest_joint.astype(np.float64)[:,np.newaxis])
    return ik_joint
        
        
        
        

def create_point_cloud_msg(points,now,frame_id='l515'):
    import rospy
    import numpy as np
    import struct  # 用于RGB颜色的打包
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    """
    将 n x 6 点云数据转换为 sensor_msgs/PointCloud2 消息
    :param points: np.array, n x 6 点云数据，前3列是(x, y, z)，后3列是(r, g, b)
    :return: sensor_msgs/PointCloud2 消息
    """
    # 创建消息头
    header = Header()
    header.stamp = now
    header.frame_id = frame_id  # 修改为你的坐标系框架ID

    # 定义 PointField
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]

    # 构建 PointCloud2 的数据部分
    # point_cloud_data = []

    # for point in points:
    #     x, y, z = point[:3]
    #     r, g, b = point[3:6].astype(np.uint8)  # 转换为uint8类型
    #     rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]  # RGB颜色打包为一个整数
    #     point_cloud_data.append([x, y, z, rgb])
    
    point_cloud_data = np.zeros((len(points), 4), dtype=np.float32)
    point_cloud_data[:, 0:3] = points[:, 0:3]
    rgb_int = np.zeros((points.shape[0],), dtype=np.uint32)
    rgb_int = (points[:, 3].astype(np.uint32) << 16) | (points[:, 4].astype(np.uint32) << 8) | points[:, 5].astype(np.uint32)
    point_cloud_data[:, 3] = rgb_int.view(np.float32)  # 将其作为 float32 存储

    # 使用 PointCloud2 构建点云消息
    point_cloud_msg = PointCloud2()
    point_cloud_msg.header = header
    point_cloud_msg.height = 1  # 表示这是一个无序点云
    point_cloud_msg.width = len(point_cloud_data)
    point_cloud_msg.is_dense = True  # 无NaN点
    point_cloud_msg.is_bigendian = False
    point_cloud_msg.fields = fields
    point_cloud_msg.point_step = 16  # 每个点的字节数 (4个float32，每个4字节)
    point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
    # point_cloud_msg.data = np.array(point_cloud_data, dtype=np.float32).tobytes()
    point_cloud_msg.data = point_cloud_data.tobytes()

    return point_cloud_msg
    
        


if __name__ == "__main__":
    dataset = RealData("/mnt/data/data/peel_data/peel_data_1/data/000")
    for i in range(len(dataset)):
        color, depth = dataset[i]
        print(color.shape, depth.shape)