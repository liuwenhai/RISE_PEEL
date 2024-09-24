import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# tcp normalization and gripper width normalization
# TRANS_MIN, TRANS_MAX = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7])
# new tcp normalization
TRANS_MIN, TRANS_MAX = np.array([-0.60, -0.6, 0]), np.array([0.6, 0.35, 0.9])
MAX_GRIPPER_WIDTH = 0.11 # meter
FORCE_MIN, FORCE_MAX = np.array([-5, -5, -15]), np.array([3, 15, 5])
TORQUE_MIN, TORQUE_MAX = np.array([-2, -2, -2]), np.array([2, 2, 2])
WRENCH_MIN, WRENCH_MAX = np.array([-5, -5, -15, -2, -2, -2]), np.array([3, 15, 5, 2, 2, 2])

# workspace in camera coordinate
# WORKSPACE_MIN = np.array([-0.6, -0.5, 0])
# WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])

WORKSPACE_MIN = np.array([-0.7, -0.7, 0])
WORKSPACE_MAX = np.array([0.7, 0.7, 1.0])

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0])
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4])

# gripper threshold (to avoid gripper action too frequently)
GRIPPER_THRESHOLD = 0.02 # meter
