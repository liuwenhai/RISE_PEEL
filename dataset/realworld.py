import os
import json
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs

import h5py
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import *
from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform, \
    xyz_rot_to_mat, mat_to_xyz_rot


class RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20, 
        voxel_size = 0.005,
        cam_ids = ['750612070851'],
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.calib_path = os.path.join(path, "calib")
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        
        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.projectors = {}
        
        for i in range(self.num_demos):
            demo_path = os.path.join(self.data_path, self.all_demos[i])
            for cam_id in cam_ids:
                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # metadata
                with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                    meta = json.load(f)
                # get frame ids
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                ]
                # get calib timestamps
                with open(os.path.join(demo_path, "timestamp.txt"), "r") as f:
                    calib_timestamp = f.readline().rstrip()
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []
                padding_mask_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1)
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)
                
                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamp += [calib_timestamp] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
        
    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        for cloud in clouds:
            cloud = apply_mat_to_pcd(cloud, aug_mat)
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        return clouds, tcps

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list

    def load_point_cloud(self, colors, depths, cam_id):
        h, w = depths.shape
        fx, fy = INTRINSICS[cam_id][0, 0], INTRINSICS[cam_id][1, 1]
        cx, cy = INTRINSICS[cam_id][0, 2], INTRINSICS[cam_id][1, 2]
        scale = 1000. if 'f' not in cam_id else 4000.
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        return points.astype(np.float32), colors.astype(np.float32)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamp[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')
        depth_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'depth')
        tcp_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'tcp')
        gripper_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'gripper_command')

        # load camera projector by calib timestamp
        timestamp_path = os.path.join(data_path, 'timestamp.txt')
        with open(timestamp_path, 'r') as f:
            timestamp = f.readline().rstrip()
        if timestamp not in self.projectors:
            # create projector cache
            self.projectors[timestamp] = Projector(os.path.join(self.calib_path, timestamp))
        projector = self.projectors[timestamp]

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        # load colors and depths
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_list.append(colors)
            depths_list.append(
                np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)

        # point clouds
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            points, colors = self.load_point_cloud(colors_list[i], depths_list[i], cam_id)
            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
            # apply imagenet normalization
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)

        # actions
        action_tcps = []
        action_grippers = []
        for frame_id in action_frame_ids:
            tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            action_tcps.append(projected_tcp)
            action_grippers.append(gripper_width)
        action_tcps = np.stack(action_tcps)
        action_grippers = np.stack(action_grippers)

        # point augmentations
        if self.split == 'train' and self.aug:
            clouds, action_tcps = self._augmentation(clouds, action_tcps)
        
        # rotation transformation (to 6d)
        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        actions = np.concatenate((action_tcps, action_grippers[..., np.newaxis]), axis = -1)

        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'action': actions,
            'action_normalized': actions_normalized
        }
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        return ret_dict
        
class PeelingDatasetHDF5(Dataset):
    """_summary_
    peeling cucumber skin dataset
    
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20,
        voxel_size = 0.05,
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        with_action_wrench = False,
        with_obs_wrench = False,
        hdf5_use_swmr = True,
        use_action_wrench = False
        ):
        assert split in ['train', 'val', 'all']
        self.path = path
        self.split = split
        
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.with_action_wrench = with_action_wrench
        self.with_obs_wrench = with_obs_wrench
        
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_path = os.path.join(self.path, 'cucumber_peel_0-9_10000.hdf5')
        # self.hdf5_path = os.path.join(self.path, 'cucumber_peel_10-11_test_10000.hdf5')
        self._hdf5_file = None
        self.action_dim = 19
        self.use_action_wrench = use_action_wrench
        if use_action_wrench:
            self.action_dim += 6
        
        self.load_demo_info()
        
        
    
    def load_demo_info(self):
        self.demos = list(self.hdf5_file["data"].keys())
        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]#[:30] # here to select how many training data to use

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.demos:
            demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length

            num_sequences = demo_length
            
            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1
        
        # self.close_and_delete_hdf5_handle()
    
    
    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None
    
    # def __del__(self):
    #     self.close_and_delete_hdf5_handle()
    
    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            # self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file
    
    def __len__(self):
        return self.total_num_sequences
    
    def __getitem__(self, index):
        return self.get_item(index)
    
    def get_item(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
        index_in_demo = index - demo_start_index
        end_index_in_demo = demo_length
        
        pointcloud,_ = self.get_sequence_from_demo(demo_id, index_in_demo, 
                                    key='obs/pointcloud', 
                                    num_frames_to_stack=self.num_obs-1, 
                                    seq_length=1)
        clouds = []
        for i in range(len(pointcloud)):
            pc = pointcloud[i]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pc[:,:3])
            cloud.colors = o3d.utility.Vector3dVector(pc[:,3:])
            cloud = cloud.voxel_down_sample(self.voxel_size)
            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([cloud, axis])
            points = np.array(cloud.points)
            colors = np.array(cloud.colors)
            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)
        
        action_with_wrench, _ = self.get_sequence_from_demo(demo_id, index_in_demo,
                                    key='action_with_wrench',
                                    num_frames_to_stack=self.num_obs-1, 
                                    seq_length=self.num_action)
        gripper_pose_action = np.concatenate([action_with_wrench[:, :3], action_with_wrench[:, 6:10]], axis=1)
        ft_pose_action = np.concatenate([action_with_wrench[:,3:6], action_with_wrench[:, 10:14]], axis=1)
        gripper_width_action = action_with_wrench[:,14:15]
        ft_pose_action[:, [3, 4, 5, 6]] = ft_pose_action[:, [6, 3, 4, 5]]
        gripper_pose_action[:, [3, 4, 5, 6]] = gripper_pose_action[:, [6, 3, 4, 5]]
        if self.split == 'train' and self.aug:
            clouds, gripper_pose_action, ft_pose_action = self._augmentation_new(clouds, gripper_pose_action, ft_pose_action)
        ft_pose_action = xyz_rot_transform(ft_pose_action, from_rep = "quaternion", to_rep = "rotation_6d")
        gripper_pose_action = xyz_rot_transform(gripper_pose_action, from_rep = "quaternion", to_rep = "rotation_6d")
        gripper_width_action = gripper_width_action.reshape(-1, 1)
        
        actions = np.concatenate((ft_pose_action, gripper_pose_action, gripper_width_action), axis = -1)
        if self.use_action_wrench:
            actions = np.concatenate((actions, action_with_wrench[:,15:]), axis = -1)
        actions_normalized = self._normalize_tcp(actions.copy())
        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))
        
        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'action': actions,
            'action_normalized': actions_normalized
        }
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        return ret_dict
        
    
    def get_sequence_from_demo(self, demo_id, index_in_demo, key, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            key: key to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # fetch observation from the dataset file
        hd5key = "data/{}/{}".format(demo_id, key)
        data = self.hdf5_file[hd5key]
        data = data[seq_begin_index: seq_end_index]
        # if seq_begin_pad > 0 or seq_end_pad > 0:
        #     import pdb;pdb.set_trace()
        data = np.concatenate([data[0:1].repeat(seq_begin_pad, axis=0),data,data[-1:].repeat(seq_end_pad, axis=0)], axis=0)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return data, pad_mask
        
    
    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, 9:12] = (tcp_list[:, 9:12] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, 18] = tcp_list[:, 18] / MAX_GRIPPER_WIDTH * 2 - 1
        if self.use_action_wrench:
            tcp_list[:, 19:] = (tcp_list[:, 19:] - WRENCH_MIN) / (WRENCH_MAX - WRENCH_MIN) * 2 - 1
        return tcp_list
    
    def _augmentation(self, clouds, tcps, tcps_1):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        for cloud in clouds:
            cloud = apply_mat_to_pcd(cloud, aug_mat)
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        tcps_1 = apply_mat_to_pose(tcps_1, aug_mat, rotation_rep = "quaternion")
        return clouds, tcps, tcps_1

    def _augmentation_new(self, clouds, tcps, tcps_1,x_min = -0.05, x_max = 0.05, y_min = -0.03, y_max = 0.03, z_min = -0.03, z_max = 0.05, z_theta_min = -np.pi / 4, z_theta_max = np.pi / 4):
        # aug_trans_max = np.array([0.05,0.03,0.05])
        # aug_trans_min = np.array([-0.05,-0.03,-0.03])
        # translation_offsets = np.random.rand(3) * (aug_trans_max - aug_trans_min) + aug_trans_min
        # rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        # rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        x_trans = np.random.uniform(x_min, x_max)
        y_trans = np.random.uniform(y_min, y_max)
        z_trans = np.random.uniform(z_min, z_max)
        z_theta = np.random.uniform(z_theta_min, z_theta_max)
        rotation_angles = np.array([0, 0, z_theta])
        translation_offsets = np.array([x_trans, y_trans, z_trans])
        pose = rot_trans_mat(translation_offsets, rotation_angles)
        # import pdb;pdb.set_trace()
        L515_2_BASE = np.array([[1, 0, 0, 0],
                                [0, -np.sin(70 / 180 * np.pi), np.cos(70 / 180 * np.pi), 0],
                                [0, -np.cos(70 / 180 * np.pi), -np.sin(70 / 180 * np.pi), 0.59],
                                [0, 0, 0, 1]])

        l5152base = L515_2_BASE
        l5152base[1, 3] = -0.25

        transform_pose = np.linalg.inv(l5152base) @ pose @ l5152base
        for cloud in clouds:
            cloud = apply_mat_to_pcd(cloud, transform_pose)
        tcps = apply_mat_to_pose(tcps, transform_pose, rotation_rep = "quaternion")
        tcps_1 = apply_mat_to_pose(tcps_1, transform_pose, rotation_rep = "quaternion")

        return clouds, tcps, tcps_1
        
    def load_point_cloud(self, colors, depths):
        h, w = depths.shape
        cx, cy, fx, fy = L515_INTRINSICS
        # scale = 1000. if 'f' not in cam_id else 4000.
        scale = 4000.
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        return points.astype(np.float32), colors.astype(np.float32)

class PeelingDataset(Dataset):
    """_summary_
    peeling cucumber skin dataset
    
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20,
        voxel_size = 0.05,
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        with_action_wrench = False,
        with_obs_wrench = False
        ):
        assert split in ['train', 'val', 'all']
        self.path = path
        self.split = split
        
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.with_action_wrench = with_action_wrench
        self.with_obs_wrench = with_obs_wrench
        
        self.all_demos = sorted(os.listdir(self.path))
        self.num_demos = len(self.all_demos)
        
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.data_paths = []
        
        for i in range(self.num_demos):
            demo_path = os.path.join(self.path, self.all_demos[i])
            ft_pose = np.load(os.path.join(demo_path, "ft_pose.npy"))
            gripper_pose = np.load(os.path.join(demo_path, "gripper_pose.npy"))
            gripper_width = np.load(os.path.join(demo_path, "gripper_width.npy"))
            wrench = np.load(os.path.join(demo_path, "wrench.npy"))
            # intrinsics = np.loadtxt(os.path.join(demo_path, "intrinsics.txt"))
            # timestamp = np.load(os.path.join(demo_path, "timestamp.npy"))
            # depth_scale = np.loadtxt(os.path.join(demo_path, "depth_scale.txt")).item()
            frame_ids = [x
                for x in sorted(os.listdir(os.path.join(demo_path, 'color')),
                                key=lambda x: int(x.split('.')[0]))
            ]
            
            obs_frame_ids_list = []
            action_frame_ids_list = []
            
            for cur_idx in range(len(frame_ids) - 1):
                obs_pad_before = max(0, num_obs - cur_idx - 1)
                action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - num_obs + 1)
                frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                # action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                
                ft_pose_action = ft_pose[cur_idx + 1: frame_end] + ft_pose[-1:] * action_pad_after
                gripper_pose_action = gripper_pose[cur_idx + 1: frame_end] + gripper_pose[-1:] * action_pad_after
                gripper_width_action = gripper_width[cur_idx + 1: frame_end] + gripper_width[-1:] * action_pad_after
                wrench_action = wrench[cur_idx + 1: frame_end] + wrench[-1:] * action_pad_after
                
                ft_pose_action = mat_to_xyz_rot(mat=ft_pose_action,rotation_rep="quaternion")
                gripper_pose_action = mat_to_xyz_rot(mat=gripper_pose_action,rotation_rep="quaternion")
                
                obs_frame_ids_list.append(obs_frame_ids)
                action_frame_ids_list.append([ft_pose_action, gripper_pose_action, gripper_width_action, wrench_action])
            
            self.data_paths += [demo_path] * len(obs_frame_ids_list)
            self.obs_frame_ids += obs_frame_ids_list
            self.action_frame_ids += action_frame_ids_list
    
    def __len__(self):
        return len(self.obs_frame_ids) 
    
    def __getitem__(self, index):
        data_path = self.data_paths[index]
        obs_frame_ids = self.obs_frame_ids[index]  
        action_frame_ids = self.action_frame_ids[index]  
        
        color_dir = os.path.join(data_path,'color')
        depth_dir = os.path.join(data_path, 'depth')
        
        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)
            
        # load colors and depths
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, frame_id))
            depths = np.array(Image.open(os.path.join(depth_dir, frame_id)), dtype = np.float32)
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_list.append(colors)
            depths_list.append(depths)
        
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)
        
        # point clouds
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            points, colors = self.load_point_cloud(colors_list[i], depths_list[i])
            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
            # apply imagenet normalization
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)
        ft_pose_action, gripper_pose_action, gripper_width_action, wrench_action = action_frame_ids
        if self.split == 'train' and self.aug:
            clouds, ft_pose_action, gripper_pose_action = self._augmentation(clouds, ft_pose_action, gripper_pose_action)
        
        ft_pose_action = xyz_rot_transform(ft_pose_action, from_rep = "quaternion", to_rep = "rotation_6d")
        gripper_pose_action = xyz_rot_transform(gripper_pose_action, from_rep = "quaternion", to_rep = "rotation_6d")
        gripper_width_action = gripper_width_action.reshape(-1, 1)
        actions = np.concatenate((ft_pose_action, gripper_pose_action, gripper_width_action), axis = -1)
        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())
        
        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))
        
        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'action': actions,
            'action_normalized': actions_normalized
        }
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        return ret_dict
    
    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, 9:12] = (tcp_list[:, 9:12] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list
    
    def _augmentation(self, clouds, tcps, tcps_1):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        for cloud in clouds:
            cloud = apply_mat_to_pcd(cloud, aug_mat)
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        tcps_1 = apply_mat_to_pose(tcps_1, aug_mat, rotation_rep = "quaternion")
        return clouds, tcps, tcps_1   
        
    def load_point_cloud(self, colors, depths):
        h, w = depths.shape
        cx, cy, fx, fy = L515_INTRINSICS
        # scale = 1000. if 'f' not in cam_id else 4000.
        scale = 4000.
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        return points.astype(np.float32), colors.astype(np.float32)

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095
