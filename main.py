"""
main.py
Member 4 – Integration, Testing & Validation
3-DOF Planar Alphabet Writing Manipulator using PyBullet
Updated: 
- Fixed Self-Collision: Implemented 'Smart IK Selection' to prioritize open arm configurations.
- Retains all previous features (Dynamic Wrapping, Green Safety Mode, Gap Filling).
"""

import time
import sys
import numpy as np
from simulation import p, move_joints, is_point_reachable, Z_DOWN, Z_UP, pen_tip, robot_id, L1, L2, L3
from kinematics import Kinematics
from trajectory import generate_letter, interpolate_line, add_pen_state

# Initialize Kinematics
kin = Kinematics(L1, L2, L3)

# ==============================================================
# --- CONFIGURATION ---
# ==============================================================
# BASE settings (Reduced significantly to prevent cutoff)
BASE_SCALE = 0.15 

# Workspace Boundaries (The "Paper" dimensions)
PAPER_LEFT_X = -0.9
PAPER_RIGHT_X = 0.9
PAPER_TOP_Y = -0.5  # Adjusted Y slightly to be closer to comfortable reach

# SPEED CONTROL
DRAWING_SPEED = 0.005 

# SAFETY THRESHOLD
ROTATION_THRESHOLD = 15.0 

# Colors
COLOR_INK = [1, 0, 0, 1]   # Red (Writing)
COLOR_SAFE = [0, 1, 0, 1]  # Green (Rotating / Air Move)

def get_best_solution(solutions, current_joints):
    """
    Selects the best IK solution to prevent self-collision and ensure smoothness.
    
    Logic:
    1. Penalize solutions where the arm folds back on itself (High angles for Link 2/3).
    2. Penalize solutions that are far from the current joint configuration.
    3. Choose the solution with the lowest cost.
    """
    if not solutions:
        return None

    best_sol = None
    min_cost = float('inf')

    # Weighting factors
    W_TRAVEL = 1.0    # Cost for moving joints
    W_FOLDING = 5.0   # High Cost for cramped angles (prevents collision)

    for sol in solutions:
        a1, a2, a3 = sol
        cur_a1, cur_a2, cur_a3 = current_joints

        # --- 1. Calculate Travel Cost (Continuity) ---
        # How much do we have to move to get to this solution?
        # We handle angle wrapping difference (simplistic approach for -180/180)
        diff_a1 = abs(a1 - cur_a1)
        diff_a2 = abs(a2 - cur_a2)
        diff_a3 = abs(a3 - cur_a3)
        travel_cost = (diff_a1 + diff_a2 + diff_a3) * W_TRAVEL

        # --- 2. Calculate Folding Cost (Collision Avoidance) ---
        # The collision in the image happens when Link 3 folds back onto Link 1.
        # This occurs when a2 and a3 are very sharp (large absolute values).
        # We discourage configurations where |a2| > 120 degrees.
        folding_cost = 0
        if abs(a2) > 120: folding_cost += 100
        if abs(a3) > 120: folding_cost += 100
        
        # Multiply by weight
        total_folding_penalty = folding_cost * W_FOLDING

        # --- 3. Total Cost ---
        total_cost = travel_cost + total_folding_penalty

        if total_cost < min_cost:
            min_cost = total_cost
            best_sol = sol

    return best_sol

def main():
    print("🦾 Starting 3-DOF Planar Manipulator Simulation...")
    
    p.resetDebugVisualizerCamera(cameraDistance=3.5, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0, -1.0, 0])

    # Initial Joints (Home) - Open configuration
    current_joints = [10, 10, 10]
    move_joints(current_joints)
    for _ in range(50): p.stepSimulation()

    print("\n" + "="*40)
    print("   ROBOTIC CALLIGRAPHY SYSTEM   ")
    print("="*40)
    
    while True:
        print("\n" + "-"*40)
        print("Type 'EXIT' or 'QUIT' to stop the simulation.")
        user_input = input("✍️  Enter a Word: ").strip().upper()
        
        if user_input in ["EXIT", "QUIT"]:
            print("👋 Shutting down...")
            break
            
        if not user_input: continue

        print("🧹 Clearing canvas for new job...")
        p.removeAllUserDebugItems()
        
        total_start = time.time()

        # --- DYNAMIC LAYOUT CALCULATION ---
        # Scales reduced to fit "N" and other tall letters within reach
        if len(user_input) > 15:
            current_scale = 0.07  # Very small for long sentences
        elif len(user_input) > 6:
            current_scale = 0.10  # Medium small
        else:
            current_scale = BASE_SCALE # 0.15 (Standard)

        # 2. Calculate Spacing
        char_spacing = 1.5 * current_scale
        line_height = 3.0 * current_scale 

        # --- CENTER ALIGNMENT CALCULATION ---
        # Instead of starting at PAPER_LEFT_X, we calculate the total width
        # and offset the start position so the text is centered at X=0
        total_text_width = len(user_input) * char_spacing
        
        # Start X is (Center - Half_Width)
        # We assume the robot center is x=0
        cursor_x = 0.0 - (total_text_width / 2.0)
        
        # Starting Y position
        cursor_y = PAPER_TOP_Y
        
        # Track previous end pos for Green Transitions
        previous_end_pos = None

        print(f"🚀 Commencing drawing: '{user_input}' (Scale: {current_scale})")
        print(f"   -> Centering logic: Start X set to {cursor_x:.2f}")

        for ch in user_input:
            
            # --- WORD WRAP CHECK ---
            # If the text is somehow wider than the paper, wrap it (Fallback logic)
            if cursor_x + char_spacing > PAPER_RIGHT_X:
                print("   [Carriage Return] Moving to next line...")
                # Recalculate center for remaining text? 
                # For simplicity in wrapping, we just return to the calculated left start
                cursor_x = 0.0 - (total_text_width / 2.0) 
                cursor_y -= line_height 

            current_offset = (cursor_x, cursor_y)

            try:
                # 1. Generate Letter Path
                letter_path = generate_letter(ch, scale=current_scale, offset=current_offset)
            except ValueError:
                print(f"⚠️  Letter '{ch}' not implemented. Skipping.")
                cursor_x += char_spacing
                continue

            # 2. Generate Transition Path
            if previous_end_pos is not None:
                start_of_new_letter = letter_path[0][:2]
                transition_points = interpolate_line(previous_end_pos, start_of_new_letter, n_points=50)
                transition_path = add_pen_state(transition_points, pen_down=False)
                full_path = np.vstack((transition_path, letter_path))
            else:
                full_path = letter_path

            print(f"   > Drawing '{ch}' at ({cursor_x:.2f}, {cursor_y:.2f})...")
            
            last_point = None
            
            for pt in full_path:
                target_x, target_y, target_pen_state = pt
                
                # Check Reachability
                if not is_point_reachable(target_x, target_y):
                    last_point = None
                    continue

                # Solve IK
                try:
                    # FIX: We use a fixed orientation (0.0) here. 
                    solutions = kin.inverse(target_x, target_y, 0.0)
                    
                    # --- NEW SELECTION LOGIC ---
                    best_solution = get_best_solution(solutions, current_joints)
                    
                    if not best_solution: 
                        last_point = None
                        continue
                        
                    new_a1, new_a2, new_a3 = best_solution
                    new_joints = [new_a1, new_a2, new_a3]
                    
                except Exception:
                    last_point = None
                    continue

                # --- ROTATION CHECK ---
                diffs = [abs(new - old) for new, old in zip(new_joints, current_joints)]
                max_rotation = max(diffs)
                is_rotating_heavily = max_rotation > ROTATION_THRESHOLD
                
                if target_pen_state == 1 and not is_rotating_heavily:
                    visual_color = COLOR_INK
                    z_height = Z_DOWN
                    is_writing_now = True
                else:
                    visual_color = COLOR_SAFE 
                    z_height = Z_UP
                    is_writing_now = False

                p.changeVisualShape(pen_tip, -1, rgbaColor=visual_color)

                move_joints(new_joints)
                for _ in range(5): p.stepSimulation()
                time.sleep(DRAWING_SPEED)

                x_fk, y_fk, _ = kin.forward(new_a1, new_a2, new_a3)
                pen_pos = [x_fk, y_fk, z_height]
                p.resetBasePositionAndOrientation(pen_tip, pen_pos, [0, 0, 0, 1])

                # --- INK LOGIC ---
                if is_writing_now and last_point is not None:
                    p.addUserDebugLine(last_point, pen_pos, [0, 0, 0], 3, 0)

                if target_pen_state == 0:
                    last_point = None 
                elif is_writing_now:
                    last_point = pen_pos 

                current_joints = new_joints
            
            # Advance Cursor & Track End Position
            cursor_x += char_spacing
            end_x, end_y, _ = kin.forward(current_joints[0], current_joints[1], current_joints[2])
            previous_end_pos = (end_x, end_y)

        print(f"✅ Finished '{user_input}' in {time.time() - total_start:.2f}s.")

    p.disconnect()

if __name__ == "__main__":
    main()