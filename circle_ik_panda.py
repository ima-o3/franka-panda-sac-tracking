from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt


# Assumes this structure:
# mujoco_projects/
#   mujoco_menagerie/
#     franka_emika_panda/
#       scene.xml
#   panda_tracking_rl/
#     circle_ik_panda.py
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)


def get_joint_limits(model, n_arm_joints=7):
    """
    Get joint limits for the first 7 Panda joints.
    If the XML does not provide useful limits, use approximate Franka Panda limits.
    """

    fallback_lower = np.array([
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
    ])

    fallback_upper = np.array([
        2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
    ])

    try:
        lower = model.jnt_range[:n_arm_joints, 0].copy()
        upper = model.jnt_range[:n_arm_joints, 1].copy()

        if np.all(upper > lower):
            return lower, upper

    except Exception:
        pass

    return fallback_lower, fallback_upper


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print("Loaded model:")
    print(MODEL_PATH)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)

    # Find the Panda hand body.
    ee_body_name = "hand"
    ee_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ee_body_name,
    )

    if ee_body_id == -1:
        print("\nCould not find body called 'hand'. Available bodies are:")
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(i, name)
        raise ValueError("End-effector body not found.")

    # Home pose for the 7 Panda arm joints.
    q_home = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 1.6, 0.7])

    # Set both actual joint positions and actuator targets.
    data.qpos[:7] = q_home
    data.ctrl[:7] = q_home

    mujoco.mj_forward(model, data)

    # Circular trajectory settings.
    radius = 0.08        # metres
    omega = 0.5          # rad/s
    run_time = 30.0      # seconds

    # Controller settings.
    kp = 4.0             # Cartesian position correction gain
    damping = 0.05       # Damped least-squares IK damping
    max_ee_speed = 0.35  # m/s
    max_joint_speed = 1.0  # rad/s

    dt = model.opt.timestep

    lower_limits, upper_limits = get_joint_limits(model)

    # Start the circle from the current end-effector position.
    ee_start = data.xpos[ee_body_id].copy()

    # This makes the target start exactly at the current hand position.
    # The circle centre is offset by -radius in x.
    circle_center = ee_start + np.array([-radius, 0.0, 0.0])

    q_cmd = q_home.copy()

    actual_positions = []
    target_positions = []
    tracking_errors = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall_time = time.time()

        while viewer.is_running():
            sim_t = data.time

            if sim_t > run_time:
                break

            # Desired circular target in the horizontal XY plane.
            target_pos = circle_center + np.array([
                radius * np.cos(omega * sim_t),
                radius * np.sin(omega * sim_t),
                0.0,
            ])

            # Feedforward target velocity of the circular trajectory.
            target_vel = np.array([
                -radius * omega * np.sin(omega * sim_t),
                radius * omega * np.cos(omega * sim_t),
                0.0,
            ])

            # Current end-effector position.
            ee_pos = data.xpos[ee_body_id].copy()

            # Cartesian tracking error.
            error = target_pos - ee_pos

            # Desired end-effector velocity = feedforward velocity + feedback correction.
            v_des = target_vel + kp * error

            # Limit end-effector speed for stability.
            speed = np.linalg.norm(v_des)
            if speed > max_ee_speed:
                v_des = v_des / speed * max_ee_speed

            # Compute translational Jacobian for the hand body.
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))

            mujoco.mj_jacBody(
                model,
                data,
                jacp,
                jacr,
                ee_body_id,
            )

            # Use only the first 7 arm joints, not the gripper.
            J = jacp[:, :7]

            # Damped least-squares inverse kinematics:
            # dq = J.T @ inv(J J.T + λ²I) @ v_des
            A = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, v_des)

            # Limit joint speed.
            dq = np.clip(dq, -max_joint_speed, max_joint_speed)

            # Integrate joint velocity command into a joint position target.
            q_cmd = q_cmd + dq * dt

            # Keep commanded joint targets inside limits.
            q_cmd = np.clip(q_cmd, lower_limits, upper_limits)

            # Send position targets to Panda actuators.
            data.ctrl[:7] = q_cmd

            # Step physics.
            mujoco.mj_step(model, data)

            # Store data for plotting.
            actual_positions.append(ee_pos)
            target_positions.append(target_pos)
            tracking_errors.append(np.linalg.norm(error))

            if len(tracking_errors) % 250 == 0:
                print(
                    f"t={sim_t:5.2f}s | "
                    f"tracking error={tracking_errors[-1]:.4f} m | "
                    f"target={target_pos.round(3)} | "
                    f"ee={ee_pos.round(3)}"
                )

            viewer.sync()

            # Rough real-time pacing.
            elapsed_wall = time.time() - start_wall_time
            if sim_t > elapsed_wall:
                time.sleep(sim_t - elapsed_wall)

    actual_positions = np.array(actual_positions)
    target_positions = np.array(target_positions)
    tracking_errors = np.array(tracking_errors)

    print("\nFinished circular tracking.")
    print(f"Mean tracking error: {tracking_errors.mean():.4f} m")
    print(f"Final tracking error: {tracking_errors[-1]:.4f} m")

    # Plot top-down XY trajectory.
    plt.figure()
    plt.plot(target_positions[:, 0], target_positions[:, 1], label="Target circle")
    plt.plot(actual_positions[:, 0], actual_positions[:, 1], label="Actual end-effector")
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("Panda End-Effector Circular Tracking")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)

    # Plot tracking error over time.
    plt.figure()
    plt.plot(tracking_errors)
    plt.xlabel("Simulation step")
    plt.ylabel("Tracking error [m]")
    plt.title("Tracking Error Over Time")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()