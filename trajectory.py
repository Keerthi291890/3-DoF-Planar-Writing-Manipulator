"""
trajectory.py
Member 2 – Letter Path Generation & Trajectory Planning

Generates smooth (x, y, pen_state) paths for uppercase letters.
Updated: 
- Full Alphabet (A-Z) support.
- 'O' is now drawn as a HEXAGON to differentiate it from 'D'.
"""

import numpy as np

# ================================================================
# --- Helper Functions ---
# ================================================================

def interpolate_line(p1, p2, n_points=100): 
    """
    Linearly interpolates between two points.
    Used for both Ink strokes (High Res) and Air strokes.
    """
    x_vals = np.linspace(p1[0], p2[0], n_points)
    y_vals = np.linspace(p1[1], p2[1], n_points)
    return np.column_stack((x_vals, y_vals))

def add_pen_state(points, pen_down=True):
    """Adds pen state (1=down, 0=up) to each point."""
    pen_state = np.full((len(points), 1), int(pen_down))
    return np.hstack((points, pen_state))

def combine_strokes(strokes):
    """
    Combine multiple strokes. 
    Interpolates the "Air Move" (Green) between strokes.
    """
    path = []
    last_end_point = None

    for i, stroke in enumerate(strokes):
        if i > 0:
            # --- AIR MOVE GENERATION ---
            start_of_new_stroke = stroke[0]
            air_path = interpolate_line(last_end_point, start_of_new_stroke, n_points=40)
            path.append(add_pen_state(air_path, pen_down=False))

        # --- INK MOVE GENERATION ---
        path.append(add_pen_state(stroke, pen_down=True))
        last_end_point = stroke[-1]

    return np.vstack(path)


# ================================================================
# --- Letter Definitions (A-Z) ---
# ================================================================

def letter_A(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0 + s/2, y0 + 2*s)),
        interpolate_line((x0 + s, y0), (x0 + s/2, y0 + 2*s)),
        interpolate_line((x0 + 0.25*s, y0 + s), (x0 + 0.75*s, y0 + s))
    ]
    return combine_strokes(strokes)

def letter_B(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0 + 2*s)), # Vertical
        interpolate_line((x0, y0 + 2*s), (x0 + 0.7*s, y0 + 2*s)), # Top Top
        interpolate_line((x0 + 0.7*s, y0 + 2*s), (x0 + 0.7*s, y0 + s)), # Top Right
        interpolate_line((x0 + 0.7*s, y0 + s), (x0, y0 + s)), # Top Bottom
        interpolate_line((x0, y0 + s), (x0 + s, y0 + s)), # Bottom Top
        interpolate_line((x0 + s, y0 + s), (x0 + s, y0)), # Bottom Right
        interpolate_line((x0 + s, y0), (x0, y0)) # Bottom Bottom
    ]
    return combine_strokes(strokes)

def letter_C(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0 + s, y0 + 2*s), (x0, y0 + 2*s)),
        interpolate_line((x0, y0 + 2*s), (x0, y0)),
        interpolate_line((x0, y0), (x0 + s, y0))
    ]
    return combine_strokes(strokes)

def letter_D(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0 + 2*s)), # Vertical
        interpolate_line((x0, y0 + 2*s), (x0 + 0.8*s, y0 + 2*s)), # Top
        interpolate_line((x0 + 0.8*s, y0 + 2*s), (x0 + 0.8*s, y0)), # Right
        interpolate_line((x0 + 0.8*s, y0), (x0, y0)) # Bottom
    ]
    return combine_strokes(strokes)

def letter_E(scale=1, offset=(0, 0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0 + 2*s)),
        interpolate_line((x0, y0 + 2*s), (x0 + s, y0 + 2*s)),
        interpolate_line((x0, y0 + s), (x0 + 0.7*s, y0 + s)),
        interpolate_line((x0, y0), (x0 + s, y0)),
    ]
    return combine_strokes(strokes)

def letter_F(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0 + 2*s)),
        interpolate_line((x0, y0 + 2*s), (x0 + s, y0 + 2*s)),
        interpolate_line((x0, y0 + s), (x0 + 0.7*s, y0 + s)),
    ]
    return combine_strokes(strokes)

def letter_G(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0 + s, y0 + 2*s), (x0, y0 + 2*s)), # Top
        interpolate_line((x0, y0 + 2*s), (x0, y0)), # Left
        interpolate_line((x0, y0), (x0 + s, y0)), # Bottom
        interpolate_line((x0 + s, y0), (x0 + s, y0 + s)), # Right up
        interpolate_line((x0 + s, y0 + s), (x0 + 0.5*s, y0 + s)) # Inward
    ]
    return combine_strokes(strokes)

def letter_H(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0+2*s)),
        interpolate_line((x0+s, y0), (x0+s, y0+2*s)),
        interpolate_line( (x0, y0+s), (x0+s, y0+s))
    ]
    return combine_strokes(strokes)

def letter_I(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0+s/2, y0), (x0+s/2, y0+2*s))
    ]
    return combine_strokes(strokes)

def letter_J(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0 + s, y0 + 2*s), (x0 + s, y0)), # Right down
        interpolate_line((x0 + s, y0), (x0, y0)), # Bottom hook
        interpolate_line((x0, y0), (x0, y0 + 0.5*s)) # Up hook
    ]
    return combine_strokes(strokes)

def letter_K(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0+2*s)),
        interpolate_line((x0, y0+s), (x0+s, y0+2*s)),
        interpolate_line((x0, y0+s), (x0+s, y0))
    ]
    return combine_strokes(strokes)

def letter_L(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0+2*s), (x0, y0)),
        interpolate_line((x0, y0), (x0+s, y0))
    ]
    return combine_strokes(strokes)

def letter_M(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0+2*s)),
        interpolate_line((x0, y0+2*s), (x0+s/2, y0+s)),
        interpolate_line((x0+s/2, y0+s), (x0+s, y0+2*s)),
        interpolate_line((x0+s, y0+2*s), (x0+s, y0))
    ]
    return combine_strokes(strokes)

def letter_N(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0+2*s)),
        interpolate_line((x0, y0+2*s), (x0+s, y0)),
        interpolate_line((x0+s, y0), (x0+s, y0+2*s))
    ]
    return combine_strokes(strokes)

def letter_O(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    # Updated to OCTAGON shape (8 sides) for a rounder look
    strokes = [
        interpolate_line((x0 + 0.3*s, y0 + 2*s), (x0 + 0.7*s, y0 + 2*s)),   # Top Horizontal
        interpolate_line((x0 + 0.7*s, y0 + 2*s), (x0 + s, y0 + 1.6*s)),     # Top-Right Corner
        interpolate_line((x0 + s, y0 + 1.6*s), (x0 + s, y0 + 0.4*s)),       # Right Vertical
        interpolate_line((x0 + s, y0 + 0.4*s), (x0 + 0.7*s, y0)),           # Bottom-Right Corner
        interpolate_line((x0 + 0.7*s, y0), (x0 + 0.3*s, y0)),               # Bottom Horizontal
        interpolate_line((x0 + 0.3*s, y0), (x0, y0 + 0.4*s)),               # Bottom-Left Corner
        interpolate_line((x0, y0 + 0.4*s), (x0, y0 + 1.6*s)),               # Left Vertical
        interpolate_line((x0, y0 + 1.6*s), (x0 + 0.3*s, y0 + 2*s))          # Top-Left Corner
    ]
    return combine_strokes(strokes)

def letter_P(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0 + 2*s)), # Vertical
        interpolate_line((x0, y0 + 2*s), (x0 + s, y0 + 2*s)), # Top
        interpolate_line((x0 + s, y0 + 2*s), (x0 + s, y0 + s)), # Right
        interpolate_line((x0 + s, y0 + s), (x0, y0 + s)) # Mid
    ]
    return combine_strokes(strokes)

def letter_Q(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0 + 2*s), (x0 + s, y0 + 2*s)), # Top
        interpolate_line((x0 + s, y0 + 2*s), (x0 + s, y0)), # Right
        interpolate_line((x0 + s, y0), (x0, y0)), # Bottom
        interpolate_line((x0, y0), (x0, y0 + 2*s)), # Left
        interpolate_line((x0 + 0.5*s, y0 + 0.5*s), (x0 + s, y0 - 0.2*s)) # Tail
    ]
    return combine_strokes(strokes)

def letter_R(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0, y0 + 2*s)), # Vertical
        interpolate_line((x0, y0 + 2*s), (x0 + s, y0 + 2*s)), # Top
        interpolate_line((x0 + s, y0 + 2*s), (x0 + s, y0 + s)), # Right
        interpolate_line((x0 + s, y0 + s), (x0, y0 + s)), # Mid
        interpolate_line((x0 + 0.2*s, y0 + s), (x0 + s, y0)) # Leg
    ]
    return combine_strokes(strokes)

def letter_S(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0 + s, y0 + 2*s), (x0, y0 + 2*s)), # Top
        interpolate_line((x0, y0 + 2*s), (x0, y0 + s)), # Left top
        interpolate_line((x0, y0 + s), (x0 + s, y0 + s)), # Mid
        interpolate_line((x0 + s, y0 + s), (x0 + s, y0)), # Right bot
        interpolate_line((x0 + s, y0), (x0, y0)) # Bottom
    ]
    return combine_strokes(strokes)

def letter_T(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0+2*s), (x0+s, y0+2*s)),
        interpolate_line((x0+s/2, y0+2*s), (x0+s/2, y0))
    ]
    return combine_strokes(strokes)

def letter_U(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0 + 2*s), (x0, y0)), # Left
        interpolate_line((x0, y0), (x0 + s, y0)), # Bottom
        interpolate_line((x0 + s, y0), (x0 + s, y0 + 2*s)) # Right
    ]
    return combine_strokes(strokes)

def letter_V(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0+2*s), (x0+s/2, y0)),
        interpolate_line((x0+s/2, y0), (x0+s, y0+2*s))
    ]
    return combine_strokes(strokes)

def letter_W(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0+2*s), (x0+0.25*s, y0)),
        interpolate_line((x0 +0.25*s, y0), (x0+s*0.5, y0+2*s)),
        interpolate_line((x0+s*0.5, y0+2*s), (x0+0.75*s, y0)),
        interpolate_line((x0+0.75*s, y0), (x0+s, y0+2*s)),
    ]
    return combine_strokes(strokes)

def letter_X(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0), (x0+s, y0+2*s)),
        interpolate_line((x0, y0+2*s), (x0+s, y0)),
    ]
    return combine_strokes(strokes)

def letter_Y(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0+2*s), (x0+s/2, y0+s)),
        interpolate_line((x0+s, y0+2*s), (x0+s/2, y0+s)),
        interpolate_line((x0+s/2, y0+s), (x0+s/2, y0))
    ]
    return combine_strokes(strokes)

def letter_Z(scale=1, offset=(0,0)):
    x0, y0 = offset; s = scale
    strokes = [
        interpolate_line((x0, y0+2*s), (x0+s, y0+2*s)),
        interpolate_line((x0+s, y0+2*s), (x0, y0)),
        interpolate_line((x0, y0), (x0+s, y0))
    ]
    return combine_strokes(strokes)

# ================================================================
# --- Dispatcher Function ---
# ================================================================

def generate_letter(letter, scale=1, offset=(0,0)):
    """Returns (x, y, pen_state) for a given letter."""
    letter = letter.upper()
    letters = {
        "A": letter_A, "B": letter_B, "C": letter_C, "D": letter_D,
        "E": letter_E, "F": letter_F, "G": letter_G, "H": letter_H,
        "I": letter_I, "J": letter_J, "K": letter_K, "L": letter_L,
        "M": letter_M, "N": letter_N, "O": letter_O, "P": letter_P,
        "Q": letter_Q, "R": letter_R, "S": letter_S, "T": letter_T,
        "U": letter_U, "V": letter_V, "W": letter_W, "X": letter_X,
        "Y": letter_Y, "Z": letter_Z
    }
    if letter not in letters:
        raise ValueError(f"Letter '{letter}' not implemented.")
    return letters[letter](scale, offset)