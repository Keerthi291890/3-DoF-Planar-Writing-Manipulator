"""
kinematics.py
Member 1: Kinematic Modeling & Mathematical Foundations

Implements Forward and Inverse Kinematics for a 3-DOF Planar RRR Manipulator.
-   FK: (theta1, theta2, theta3) -> (x, y, phi)
-   IK: (x, y, phi) -> (theta1, theta2, theta3)

Note: All angle inputs/outputs are in **degrees**.
Internal calculations use radians.
"""

import numpy as np
import matplotlib.pyplot as plt

class Kinematics:
    """
    Handles FK and IK calculations for a 3-DOF Planar RRR arm.
    Link parameters L1, L2, L3 are defined on initialization.
    """
    def __init__(self, L1: float, L2: float, L3: float):
        """
        Initializes the manipulator with specific link lengths.
        
        Args:
            L1 (float): Length of Link 1 (base to elbow)
            L2 (float): Length of Link 2 (elbow to wrist)
            L3 (float): Length of Link 3 (wrist to end-effector)
        """
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L_total = L1 + L2 + L3

    def forward(self, a1: float, a2: float, a3: float) -> tuple[float, float, float]:
        """
        Calculates Forward Kinematics (FK).
        Converts joint angles (theta1, theta2, theta3) to 
        end-effector position (x, y) and orientation (phi).

        Args:
            a1 (float): Joint 1 angle (degrees)
            a2 (float): Joint 2 angle (degrees)
            a3 (float): Joint 3 angle (degrees)

        Returns:
            tuple[float, float, float]: (x, y, phi) where phi is 
                                        end-effector orientation in degrees.
        """
        # Convert degrees to radians for numpy calculations
        t1 = np.deg2rad(a1)
        t2 = np.deg2rad(a2)
        t3 = np.deg2rad(a3)

        # Cumulative angles
        t12 = t1 + t2
        t123 = t1 + t2 + t3

        # Calculate x, y position
        # x = L1*cos(t1) + L2*cos(t1+t2) + L3*cos(t1+t2+t3)
        # y = L1*sin(t1) + L2*sin(t1+t2) + L3*sin(t1+t2+t3)
        x = self.L1 * np.cos(t1) + self.L2 * np.cos(t12) + self.L3 * np.cos(t123)
        y = self.L1 * np.sin(t1) + self.L2 * np.sin(t12) + self.L3 * np.sin(t123)
        
        # Calculate final orientation (phi)
        # This is the absolute angle of the final link
        phi_rad = t123
        
        # Convert orientation back to degrees
        phi = np.rad2deg(phi_rad)
        
        # As per prompt, z is 0 for a planar arm
        # We return (x, y, phi) as (x, y, z) is contradictory
        return (x, y, phi)

    def inverse(self, x: float, y: float, phi: float) -> list[tuple[float, float, float]]:
        """
        Calculates Inverse Kinematics (IK).
        Converts end-effector pose (x, y, phi) to joint angles 
        (theta1, theta2, theta3).
        
        This method returns two solutions corresponding to "elbow up"
        and "elbow down" configurations.

        Args:
            x (float): Target x-position
            y (float): Target y-position
            phi (float): Target orientation (degrees)

        Returns:
            list[tuple[float, float, float]]: A list of valid solutions.
                Each solution is (a1, a2, a3) in degrees.
                Returns an empty list [] if the position is unreachable.
        """
        solutions = []
        
        # Convert target orientation to radians
        phi_rad = np.deg2rad(phi)
        
        # --- Step 1: Find the Wrist Center (W) ---
        # We find the (x, y) position of the wrist (joint between L2 and L3)
        # by "walking back" from the end-effector (x, y) along L3.
        Wx = x - self.L3 * np.cos(phi_rad)
        Wy = y - self.L3 * np.sin(phi_rad)

        # --- Step 2: Solve 2-DOF IK for the (Wx, Wy) target ---
        # This is the standard 2-link IK solution for links L1 and L2
        
        # Distance from origin to wrist
        D_sq = Wx**2 + Wy**2
        D = np.sqrt(D_sq)

        # Check for reachability
        # If D is greater than L1+L2, it's too far
        # If D is less than |L1-L2|, it's too close (assuming L1 != L2)
        if D > (self.L1 + self.L2) or D < abs(self.L1 - self.L2):
            # Target is unreachable
            return []

        # Use Law of Cosines to find angle a2 (theta2)
        # D^2 = L1^2 + L2^2 - 2*L1*L2*cos(pi - a2)
        # cos(pi - a2) = -cos(a2)
        # D^2 = L1^2 + L2^2 + 2*L1*L2*cos(a2)
        # cos(a2) = (D^2 - L1^2 - L2^2) / (2 * L1 * L2)
        
        cos_a2 = (D_sq - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        
        # Clamp value to [-1, 1] to avoid domain errors from float precision
        cos_a2 = np.clip(cos_a2, -1.0, 1.0)
        
        # Calculate the two possible solutions for a2 (elbow up/down)
        a2_rad_up = -np.arccos(cos_a2) # Elbow "up"
        a2_rad_down = np.arccos(cos_a2) # Elbow "down"

        for a2_rad in [a2_rad_up, a2_rad_down]:
            # --- Step 3: Solve for a1 (theta1) ---
            # Use the atan2-based geometric solution
            # k1 = L1 + L2*cos(a2)
            # k2 = L2*sin(a2)
            # a1 = atan2(Wy, Wx) - atan2(k2, k1)
            
            k1 = self.L1 + self.L2 * np.cos(a2_rad)
            k2 = self.L2 * np.sin(a2_rad)
            
            a1_rad = np.arctan2(Wy, Wx) - np.arctan2(k2, k1)
            
            # --- Step 4: Solve for a3 (theta3) ---
            # We know phi = a1 + a2 + a3
            # So, a3 = phi - a1 - a2
            a3_rad = phi_rad - a1_rad - a2_rad
            
            # --- Step 5: Normalize and convert to degrees ---
            # Normalize all angles to [-180, 180] for consistency
            a1_deg = self.normalize_angle(np.rad2deg(a1_rad))
            a2_deg = self.normalize_angle(np.rad2deg(a2_rad))
            a3_deg = self.normalize_angle(np.rad2deg(a3_rad))
            
            solutions.append((a1_deg, a2_deg, a3_deg))

        return solutions

    @staticmethod
    def normalize_angle(angle_deg: float) -> float:
        """Normalizes an angle to the range [-180, 180]."""
        return (angle_deg + 180) % 360 - 180

# ===================================================================
# --- VERIFICATION & PLOTTING (Deliverables) ---
# ===================================================================

def test_kinematics(kin: Kinematics):
    """
    Performs a round-trip test (FK -> IK -> FK) to validate
    the kinematic solutions.
    """
    print("--- Kinematics Round-Trip Test ---")
    
    # Define sample joint angles
    # Test multiple configurations
    test_angles = [
        (30, 45, 10),
        (90, 0, 0),
        (45, -30, 60),
        (0, 90, 0)
    ]
    
    for i, angles_in in enumerate(test_angles):
        print(f"\nTest Case {i+1}:")
        a1_in, a2_in, a3_in = angles_in
        print(f"  Input Angles:\t (a1={a1_in:.2f}, a2={a2_in:.2f}, a3={a3_in:.2f})")
        
        # 1. Forward Kinematics
        x, y, phi = kin.forward(a1_in, a2_in, a3_in)
        print(f"  -> FK Result:\t (x={x:.2f}, y={y:.2f}, phi={phi:.2f})")
        
        # 2. Inverse Kinematics
        ik_solutions = kin.inverse(x, y, phi)
        print(f"  -> IK Solutions: {len(ik_solutions)} found")
        
        if not ik_solutions:
            print("  ERROR: No IK solution found!")
            continue

        # 3. Check solutions
        found = False
        for sol in ik_solutions:
            a1_out, a2_out, a3_out = sol
            
            # 4. FK on the IK result
            x_out, y_out, phi_out = kin.forward(a1_out, a2_out, a3_out)
            
            # Check if the FK(IK) output matches the original FK output
            if (np.isclose(x, x_out) and 
                np.isclose(y, y_out) and 
                np.isclose(phi, kin.normalize_angle(phi_out))):
                print(f"     Solution {sol} VERIFIED.")
                found = True
            
            # Also check if the IK solution matches the input angles
            # Note: Multiple joint configs can lead to the same pose
            if (np.isclose(kin.normalize_angle(a1_in), a1_out) and
                np.isclose(kin.normalize_angle(a2_in), a2_out) and
                np.isclose(kin.normalize_angle(a3_in), a3_out)):
                print(f"     -> Matched original input angles.")

        if not found:
            print("  ERROR: IK solution did not verify via FK round-trip.")
    print("------------------------------------")


def plot_workspace(kin: Kinematics, num_points: int = 5000):
    """
    Generates a scatter plot of the reachable (x, y) workspace
    using a Monte Carlo method.
    """
    print("\nGenerating workspace plot...")
    points_x = []
    points_y = []
    
    # Generate random joint angles
    # Assuming full 360-degree rotation for all joints
    rand_a1 = np.random.uniform(-180, 180, num_points)
    rand_a2 = np.random.uniform(-180, 180, num_points)
    rand_a3 = np.random.uniform(-180, 180, num_points)
    
    for i in range(num_points):
        x, y, _ = kin.forward(rand_a1[i], rand_a2[i], rand_a3[i])
        points_x.append(x)
        points_y.append(y)

    plt.figure(figsize=(8, 8))
    plt.scatter(points_x, points_y, s=1, alpha=0.3, c='blue')
    
    # Draw circles for max and min reach
    max_reach = kin.L1 + kin.L2 + kin.L3
    # Inner reach is complex, depends on L1, L2, L3 config
    # L1 - L2 - L3, L2 - L1 - L3, etc.
    min_reach = 0 # Can likely reach origin if L1 < L2+L3, etc.
    
    ax = plt.gca()
    ax.add_patch(plt.Circle((0, 0), max_reach, fill=False, 
                           linestyle='--', color='red', label=f'Max Reach (L1+L2+L3 = {max_reach:.1f})'))
    
    plt.title(f"Reachable Workspace (L1={kin.L1}, L2={kin.L2}, L3={kin.L3})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    print("Workspace plot generated.")


def plot_trajectory(kin: Kinematics):
    """
    Generates verification plots for a sample trajectory (a circle).
    1. Plots Desired vs. Achieved Cartesian path.
    2. Plots the corresponding joint angle trajectories.
    """
    print("\nGenerating trajectory verification plots...")
    
    # --- 1. Define a desired path ---
    # A circle centered at (12, 0) with radius 4
    # Hold orientation (phi) constant at 90 degrees
    t = np.linspace(0, 2 * np.pi, 150)
    path_x = 12 + 4 * np.cos(t)
    path_y = 0 + 4 * np.sin(t)
    path_phi = np.full_like(t, 90.0) # Hold orientation at 90 deg

    # --- 2. Calculate IK and FK for each point ---
    joint_traj_a1 = []
    joint_traj_a2 = []
    joint_traj_a3 = []
    achieved_x = []
    achieved_y = []

    for i in range(len(t)):
        x_d, y_d, phi_d = path_x[i], path_y[i], path_phi[i]
        
        # Calculate IK
        solutions = kin.inverse(x_d, y_d, phi_d)
        
        if solutions:
            # Choose the first solution (e.g., "elbow up")
            # For real robotics, you'd choose the one closest
            # to the previous joint state to ensure continuity.
            sol = solutions[0] 
            
            joint_traj_a1.append(sol[0])
            joint_traj_a2.append(sol[1])
            joint_traj_a3.append(sol[2])
            
            # Verify with FK
            x_a, y_a, _ = kin.forward(sol[0], sol[1], sol[2])
            achieved_x.append(x_a)
            achieved_y.append(y_a)
        else:
            # Point was unreachable
            pass # Skip plotting this point

    # --- 3. Create Plots ---
    plt.figure(figsize=(12, 6))

    # Plot 1: Cartesian Path
    plt.subplot(1, 2, 1)
    plt.plot(path_x, path_y, 'r--', linewidth=2, label='Desired Path')
    plt.plot(achieved_x, achieved_y, 'b-', alpha=0.7, label='Achieved Path (FK(IK))')
    plt.title("Path Verification")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':')

    # Plot 2: Joint Trajectories
    plt.subplot(1, 2, 2)
    plt.plot(joint_traj_a1, label=r'$\theta_1$ (Base)')
    plt.plot(joint_traj_a2, label=r'$\theta_2$ (Elbow)')
    plt.plot(joint_traj_a3, label=r'$\theta_3$ (Wrist)')
    plt.title("Joint Trajectories")
    plt.xlabel("Path Sample Point")
    plt.ylabel("Joint Angle (degrees)")
    plt.legend()
    plt.grid(True, linestyle=':')
    
    plt.tight_layout()
    print("Trajectory plots generated.")


# ===================================================================
# --- MAIN EXECUTION ---
# ===================================================================
if __name__ == "__main__":
    
    # ===== INCREASED LINK LENGTHS FOR BIG WORKSPACE =====
    L1 = 1.0   # Base to elbow
    L2 = 1.0   # Elbow to wrist
    L3 = 1.0   # Wrist to end-effector
    
    manipulator = Kinematics(L1, L2, L3)
    print(f"Initialized 3-DOF RRR Manipulator (L1={L1}, L2={L2}, L3={L3})")

    # 2. Run IK/FK validation test
    test_kinematics(manipulator)
    
    # 3. Generate workspace plot (increased points)
    plot_workspace(manipulator, num_points=30000)
    
    # 4. Generate trajectory verification plots
    plot_trajectory(manipulator)
    
    # 5. Show plots
    print("\nDisplaying plots...")
    plt.show()
