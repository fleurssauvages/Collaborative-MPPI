"""Two Panda arms facing each other in one MuJoCo scene.

The left arm stands at x = -base_distance/2 and faces +x; the right arm stands
at x = +base_distance/2 and faces -x, so their workspaces overlap around x = 0.
Every arm joint has the Menagerie position actuator. The hand is replaced by a
stick with a flat disc tip (franka_emika_panda/panda_stick.xml, as in Franka_MPPI) whose tip centre is
the end-effector. A scene XML (reaching or air hockey) supplies everything else.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent / 'assets'
PANDA_XML = ASSETS / 'franka_emika_panda' / 'panda_stick.xml'
SCENE_XML = ASSETS / 'dual_panda_scene.xml'
ARMS = ('left', 'right')
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])
EE_OFFSET = 0.086   # stick body origin to the tip centre, metres
TIP_R, TIP_HALF_H = 0.02, 0.01  # flat disc at the stick tip, metres
# Bodies whose origins approximate each arm for arm-arm clearance.
PROXY_BODIES = ('link3', 'link4', 'link5', 'link6', 'link7', 'stick')


def build_model(base_distance: float = 1.5, scene_xml: Path = SCENE_XML) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    for arm, x, yaw in (('left', -base_distance / 2, 0.0), ('right', base_distance / 2, np.pi)):
        panda = mujoco.MjSpec.from_file(str(PANDA_XML))
        panda.body('stick').add_site(name='ee', pos=[0, 0, EE_OFFSET], size=[0.004, 0, 0], rgba=[0.1, 1, 0.1, 0.35])
        frame = spec.worldbody.add_frame(pos=[x, 0, 0], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        frame.attach_body(panda.body('link0'), f'{arm}_', '')
    # Rollout outputs: tip position, then the clearance proxies, per arm.
    for arm in ARMS:
        spec.add_sensor(name=f'{arm}_ee_pos', type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f'{arm}_ee')
        for body in PROXY_BODIES:
            spec.add_sensor(name=f'{arm}_{body}_pos', type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                            objtype=mujoco.mjtObj.mjOBJ_XBODY, objname=f'{arm}_{body}')
    # Tip linear velocity, after the contiguous per-arm position blocks.
    for arm in ARMS:
        spec.add_sensor(name=f'{arm}_ee_vel', type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f'{arm}_ee')
    return spec.compile()


class DualFranka:
    def __init__(self, base_distance: float = 1.5, scene_xml: Path = SCENE_XML):
        self.model = build_model(base_distance, scene_xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_qadr, self.joint_dadr, self.arm_act, self.ee_site = {}, {}, {}, {}
        for arm in ARMS:
            joints = [m.joint(f'{arm}_joint{k}') for k in range(1, 8)]
            self.joint_qadr[arm] = np.array([j.qposadr[0] for j in joints])
            self.joint_dadr[arm] = np.array([j.dofadr[0] for j in joints])
            self.arm_act[arm] = np.array([m.actuator(f'{arm}_actuator{k}').id for k in range(1, 8)])
            self.ee_site[arm] = m.site(f'{arm}_ee').id
        self.target_mocap = {arm: m.body(f'{arm}_target').mocapid[0] for arm in ARMS}
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        # Each arm's sensors are contiguous: tip first, then the clearance proxies.
        self.sensor_ee = {arm: self.sensor_slice(f'{arm}_ee_pos') for arm in ARMS}
        self.sensor_ee_vel = {arm: self.sensor_slice(f'{arm}_ee_vel') for arm in ARMS}
        self.sensor_arm = {arm: slice(self.sensor_ee[arm].start,
                                      self.sensor_slice(f'{arm}_{PROXY_BODIES[-1]}_pos').stop) for arm in ARMS}
        self.reset()

    def reset(self, q_home: np.ndarray = HOME_Q):
        mujoco.mj_resetData(self.model, self.data)
        for arm in ARMS:
            self.data.qpos[self.joint_qadr[arm]] = q_home
            self.data.ctrl[self.arm_act[arm]] = q_home
        mujoco.mj_forward(self.model, self.data)

    def sensor_slice(self, name: str) -> slice:
        """Columns of a named sensor in sensordata (and in rollout sensor output)."""
        sensor = self.model.sensor(name)
        return slice(int(sensor.adr[0]), int(sensor.adr[0] + sensor.dim[0]))

    @property
    def nu(self) -> int:
        return self.model.nu

    def q(self, arm, data=None):
        return (self.data if data is None else data).qpos[self.joint_qadr[arm]].copy()

    def dq(self, arm, data=None):
        return (self.data if data is None else data).qvel[self.joint_dadr[arm]].copy()

    def ee(self, arm, data=None):
        return (self.data if data is None else data).site_xpos[self.ee_site[arm]].copy()

    def jacobian(self, arm, data=None):
        """Translational tip Jacobian (3 x 7) over the arm's joints."""
        d = self.data if data is None else data
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, jacp, None, self.ee_site[arm])
        return jacp[:, self.joint_dadr[arm]]

    def jacobian_upright(self, arm, data=None):
        """Tip translational Jacobian stacked with the x and y rows of the rotational
        one (5 x 7), and the tool z axis: rotation about the vertical stays free."""
        d = self.data if data is None else data
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, jacp, jacr, self.ee_site[arm])
        dofs = self.joint_dadr[arm]
        return np.vstack([jacp[:, dofs], jacr[:2, dofs]]), d.site_xmat[self.ee_site[arm]].reshape(3, 3)[:, 2].copy()

    def set_target_marker(self, arm, xyz):
        self.data.mocap_pos[self.target_mocap[arm]] = xyz

    def state(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        out = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, d, out, spec)
        return out
