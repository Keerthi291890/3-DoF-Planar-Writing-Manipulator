"""
simulation.py
Sets up the PyBullet environment and exposes the robot/pen for main.py
"""

import pybullet as p
import pybullet_data
import numpy as np
from kinematics import Kinematics

# -------------------- Simulation Setup --------------------
if not p.isConnected():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.resetSimulation()
p.setGravity(0, 0, -9.8)

# Load plane
plane_id = p.loadURDF("plane.urdf")

# Load the Updated Industrial Robot
# Lift base (z=0.1) so the pedestal sits on the floor
# useFixedBase=True is CRITICAL so the robot doesn't fall over
robot_id = p.loadURDF("planar_3dof.urdf", basePosition=[0, 0, 0.1], useFixedBase=True)

# Pen Tip (The "Ink" Droplet Logic Object)
pen_tip = p.createVisualShape(p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0, 0, 1])
pen_tip = p.createMultiBody(baseMass=0, baseVisualShapeIndex=pen_tip)

# -------------------- Robot Parameters --------------------
L1, L2, L3 = 1.0, 1.0, 1.0
kin = Kinematics(L1, L2, L3)
L_total = L1 + L2 + L3

# Z heights for Drawing
Z_DOWN = 0.02 
Z_UP = 0.2

# -------------------- Helper Functions --------------------

def move_joints(joint_angles):
    """
    Moves the robot joints to the specified angles (in degrees).
    High force is used to ensure the heavier links move snappy.
    """
    for i in range(3):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id, 
            jointIndex=i, 
            controlMode=p.POSITION_CONTROL, 
            targetPosition=np.deg2rad(joint_angles[i]),
            force=800,    # High torque for industrial look
            positionGain=0.1,
            velocityGain=1.0
        )

def is_point_reachable(x, y):
    D = np.sqrt(x**2 + y**2)
    return D <= L_total