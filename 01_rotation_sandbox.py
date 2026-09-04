import mujoco
import mujoco.viewer
import numpy as np
import time


# ============================================================
# ROTATION MATRICES
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
# MATRIX TO MUJOCO QUATERNION
# MuJoCo quaternion format: [w, x, y, z]
# ============================================================

def matrix_to_quaternion(R):

    q = np.zeros(4)

    mujoco.mju_mat2Quat(q, R.flatten())

    return q


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(
    "model/asymmetric_body.xml"
)

data = mujoco.MjData(model)


body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "asymmetric_body"
)


# ============================================================
# DEFINE ROTATIONS
# ============================================================

angle = np.pi / 2

Rx = rot_x(angle)
Ry = rot_y(angle)


# ============================================================
# CASE A
# First X, then Y
#
# For fixed/space frame rotations:
#
# R_new = R_increment @ R_old
#
# Therefore:
#
# R_A = Ry @ Rx
# ============================================================

R_A_step0 = np.eye(3)

R_A_step1 = Rx

R_A_final = Ry @ Rx


# ============================================================
# CASE B
# First Y, then X
#
# R_B = Rx @ Ry
# ============================================================

R_B_step0 = np.eye(3)

R_B_step1 = Ry

R_B_final = Rx @ Ry


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("====================================================")
print("       PROBLEM 7: ROTATION NON-COMMUTATIVITY")
print("====================================================")

print("\nRotations are about the FIXED (SPACE) FRAME.")

print("\nCASE A:")
print("Step 1: Rotate +90 degrees about X axis")
print("Step 2: Rotate +90 degrees about Y axis")

print("\nFinal rotation matrix A = Ry @ Rx")

print(R_A_final)


print("\n----------------------------------------------------")


print("\nCASE B:")
print("Step 1: Rotate +90 degrees about Y axis")
print("Step 2: Rotate +90 degrees about X axis")

print("\nFinal rotation matrix B = Rx @ Ry")

print(R_B_final)


print("\n----------------------------------------------------")

print("\nMatrix difference:")

print(R_A_final - R_B_final)


print("\nAre final orientations equal?")

print(np.allclose(R_A_final, R_B_final))


print("\nConclusion:")

print("Ry @ Rx != Rx @ Ry")

print("Therefore, 3D rotations are non-commutative.")

print("====================================================\n")


# ============================================================
# FUNCTION TO SET BODY ORIENTATION
# ============================================================

def set_orientation(R):

    q = matrix_to_quaternion(R)

    model.body_quat[body_id] = q

    mujoco.mj_forward(model, data)


# ============================================================
# VIEWER
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("MuJoCo viewer started.")
    print("Each stage will be displayed for 2 seconds.\n")


    while viewer.is_running():

        # ====================================================
        # CASE A
        # ====================================================

        print("\n====================================")
        print("CASE A")
        print("Initial orientation")
        print("====================================")

        set_orientation(R_A_step0)

        start = time.time()

        while time.time() - start < 2 and viewer.is_running():

            viewer.sync()

            time.sleep(0.01)


        print("\nCASE A")
        print("Step 1: Rotate about FIXED X axis")

        set_orientation(R_A_step1)

        start = time.time()

        while time.time() - start < 2 and viewer.is_running():

            viewer.sync()

            time.sleep(0.01)


        print("\nCASE A")
        print("Step 2: Rotate about FIXED Y axis")
        print("FINAL ORIENTATION A")

        set_orientation(R_A_final)

        start = time.time()

        while time.time() - start < 2 and viewer.is_running():

            viewer.sync()

            time.sleep(0.01)


        # ====================================================
        # CASE B
        # ====================================================

        print("\n====================================")
        print("CASE B")
        print("Initial orientation")
        print("====================================")

        set_orientation(R_B_step0)

        start = time.time()

        while time.time() - start < 2 and viewer.is_running():

            viewer.sync()

            time.sleep(0.01)


        print("\nCASE B")
        print("Step 1: Rotate about FIXED Y axis")

        set_orientation(R_B_step1)

        start = time.time()

        while time.time() - start < 2 and viewer.is_running():

            viewer.sync()

            time.sleep(0.01)


        print("\nCASE B")
        print("Step 2: Rotate about FIXED X axis")
        print("FINAL ORIENTATION B")

        set_orientation(R_B_final)

        start = time.time()

        while time.time() - start < 2 and viewer.is_running():

            viewer.sync()

            time.sleep(0.01)
