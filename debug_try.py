from dataset.realworld import RealWorldDataset, PeelingDataset, PeelingDatasetHDF5
import torch
from tqdm import tqdm
from utils_temp import Agent, PeelDataRos
path = 'data/collect_cups'
# path = 'data/peel_data_1'
path = 'data/peel_data_1'
num_action = 20
voxel_size = 0.005
aug = False
aug_jitter = False

dataset = PeelingDatasetHDF5(
        path = path,
        split = 'train',
        num_obs = 1,
        num_action = num_action,
        voxel_size = voxel_size,
        aug = aug,
        aug_jitter = aug_jitter, 
        with_cloud = False,
        use_action_wrench=True
    )
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        num_workers = 0,
    )
dataloader = tqdm(dataset)
# vis dataset, not including robot
agent_data = PeelDataRos()
# vis robot
# agent_robot = Agent()
import time
from utils.constants import *
def unnormalize_action(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., 9:12] = (action[..., 9:12] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., 18] = (action[..., 18] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    if action.shape[-1]>19:
        action[..., 19:] = (action[..., 19:] + 1) / 2.0 * (WRENCH_MAX - WRENCH_MIN) + WRENCH_MIN
    # tcp_list[:, 19:] = (tcp_list[:, 19:] - WRENCH_MIN) / (WRENCH_MAX - WRENCH_MIN) * 2 - 1
    return action
for data in dataloader:
    # import pdb;pdb.set_trace()
    # data['input_coords_list']
    cloud = data['input_feats_list'][0]
    action = data['action'].numpy()[0]
    action_norm = data['action_normalized'].numpy()[0]
    cloud[:, 3:] = cloud[:, 3:] * IMG_STD + IMG_MEAN
    cloud[:, 3:] *= 255
    action_norm_ = unnormalize_action(action_norm)
    # agent_robot.update_robot_state(action, cloud)
    # import pdb;pdb.set_trace()
    if aug:
        for act in data['action_normalized'].numpy():
            act = unnormalize_action(act)
            agent_data.update_all(act, cloud)
            time.sleep(0.01)
    else:
        # for act in data['action_normalized'].numpy():
        #     act = unnormalize_action(act)
        #     agent_data.update_all(act, cloud)
        #     time.sleep(0.1)
        agent_data.update_all(action, cloud)
        time.sleep(0.1)
    print('wrench: ', action[-6:])
    # data['action_normalized']

    