from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = Path("../mujoco_menagerie/franka_emika_panda/scene.xml")


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print("Model loaded successfully.")
    print("Number of positions nq:", model.nq)
    print("Number of velocities nv:", model.nv)
    print("Number of actuators nu:", model.nu)

    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(i, name)

    # A reasonable starting pose for the Panda arm.
    home_ctrl = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 1.6, 0.7])

    if model.nu >= 7:
        data.ctrl[:7] = home_ctrl

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        while viewer.is_running() and time.time() - start < 20:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()