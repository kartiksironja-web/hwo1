import mujoco
import mujoco.viewer
import numpy as np
import time


# -------------------------------------------------
# Hat (skew-symmetric) operator
# -------------------------------------------------
def hat(w):
    wx, wy, wz = w

    return np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])


# -------------------------------------------------
# Time-varying angular velocity
# -------------------------------------------------
def angular_velocity(t):
    return np.array([
        np.sin(t),
        np.cos(t),
        0.5 * np.sin(2 * t)
    ])


# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
model = mujoco.MjModel.from_xml_path(
    "model/asymmetric_body.xml"
)

data = mujoco.MjData(model)

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "asymmetric_body"
)


print("\n==============================================")
print("PROBLEM 8: VERIFYING SKEW-SYMMETRIC IDENTITIES")
print("==============================================")

print("\nIdentity 1:")
print("R(v x w) = (Rv) x (Rw)")

print("\nIdentity 2:")
print("hat(R omega) = R hat(omega) R^T\n")


# -------------------------------------------------
# Start viewer
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:

    start_time = time.time()
    last_print = -1

    while viewer.is_running():

        t = time.time() - start_time

        # Current time-varying angular velocity
        omega = angular_velocity(t)

        # Give the body angular velocity so it rotates
        data.qvel[:] = 0 if data.qvel.size > 0 else data.qvel

        # Current rotation matrix
        mujoco.mj_forward(model, data)

        R = data.xmat[body_id].reshape(3, 3).copy()

        # Random vectors
        v = np.random.randn(3)
        w = np.random.randn(3)

        # =============================================
        # IDENTITY 1
        # R(v x w) = (Rv) x (Rw)
        # =============================================

        lhs1 = R @ np.cross(v, w)

        rhs1 = np.cross(
            R @ v,
            R @ w
        )

        residual1 = np.linalg.norm(lhs1 - rhs1)


        # =============================================
        # IDENTITY 2
        # hat(R omega) = R hat(omega) R^T
        # =============================================

        lhs2 = hat(R @ omega)

        rhs2 = R @ hat(omega) @ R.T

        residual2 = np.linalg.norm(lhs2 - rhs2)


        # Print once per second
        current_second = int(t)

        if current_second != last_print:

            print("----------------------------------------------")
            print(f"Time = {t:.2f} seconds")
            print(f"omega = {omega}")
            print(f"Identity 1 residual = {residual1:.3e}")
            print(f"Identity 2 residual = {residual2:.3e}")

            last_print = current_second


        viewer.sync()
        time.sleep(0.01)
