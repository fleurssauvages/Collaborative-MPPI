"""Two Panda arms in one MuJoCo scene.

By default the arms face each other: the left arm stands at x = -base_distance/2
and faces +x; the right arm stands at x = +base_distance/2 and faces -x, so their
workspaces overlap around x = 0. ``bases`` places them otherwise ({arm: (xyz, yaw)}).
Every arm joint has the Menagerie position actuator. With tool='stick' the hand is
replaced by a stick with a flat disc tip (franka_emika_panda/panda_stick.xml, as in
Franka_MPPI) whose tip centre is the end-effector. With tool='gripper' the arm keeps
the Franka hand; the end-effector sits between the fingertip pads and the gripper
servo is stiffened (see GRIP_STIFFNESS). A scene XML supplies everything else.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent / 'assets'
PANDA_XML = ASSETS / 'franka_emika_panda' / 'panda_stick.xml'
PANDA_GRIPPER_XML = ASSETS / 'franka_emika_panda' / 'panda.xml'
SCENE_XML = ASSETS / 'dual_panda_scene.xml'
ARMS = ('left', 'right')
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])
EE_OFFSET = 0.086   # stick body origin to the tip centre, metres
TIP_R, TIP_HALF_H = 0.02, 0.01  # flat disc at the stick tip, metres
# Bodies whose origins approximate each arm for arm-arm clearance.
PROXY_BODIES = ('link3', 'link4', 'link5', 'link6', 'link7', 'stick')
GRIPPER_PROXY_BODIES = ('link3', 'link4', 'link5', 'link6', 'link7', 'hand')
TCP_OFFSET = 0.103  # hand origin to the centre between the fingertip pads, metres
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0  # gripper commands (0.08 m and 0 m between the fingers)
# The Menagerie servo (100 N/m) squeezes a 4 cm object with only 2 N; this holds it with 15 N,
# so a held object does not creep out while the arm moves. MuJoCo contacts soften with the
# object's mass: much more grip drives the fingers through a light (5 g) object.
GRIP_STIFFNESS, GRIP_DAMPING = 750.0, 40.0


def facing_bases(base_distance: float = 1.5) -> dict:
    return {'left': ((-base_distance / 2, 0.0, 0.0), 0.0), 'right': ((base_distance / 2, 0.0, 0.0), np.pi)}


def build_model(base_distance: float = 1.5, scene_xml: Path = SCENE_XML, tool: str = 'stick',
                bases: dict | None = None) -> mujoco.MjModel:
    if tool not in ('stick', 'gripper'):
        raise ValueError("tool must be 'stick' or 'gripper'")
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    for arm, (pos, yaw) in (bases or facing_bases(base_distance)).items():
        if tool == 'stick':
            panda = mujoco.MjSpec.from_file(str(PANDA_XML))
            panda.body('stick').add_site(name='ee', pos=[0, 0, EE_OFFSET], size=[0.004, 0, 0], rgba=[0.1, 1, 0.1, 0.35])
        else:
            panda = mujoco.MjSpec.from_file(str(PANDA_GRIPPER_XML))
            panda.body('hand').add_site(name='ee', pos=[0, 0, TCP_OFFSET], size=[0.004, 0, 0], rgba=[0.1, 1, 0.1, 0.35])
            grip = panda.actuator('actuator8')
            grip.gainprm[0] = 0.04 * GRIP_STIFFNESS / 255
            grip.biasprm[1], grip.biasprm[2] = -GRIP_STIFFNESS, -GRIP_DAMPING
            for key in list(panda.keys):
                panda.delete(key)
        frame = spec.worldbody.add_frame(pos=list(pos), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        frame.attach_body(panda.body('link0'), f'{arm}_', '')
    # Rollout outputs: tip position, then the clearance proxies, per arm.
    proxies = PROXY_BODIES if tool == 'stick' else GRIPPER_PROXY_BODIES
    for arm in ARMS:
        spec.add_sensor(name=f'{arm}_ee_pos', type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f'{arm}_ee')
        for body in proxies:
            spec.add_sensor(name=f'{arm}_{body}_pos', type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                            objtype=mujoco.mjtObj.mjOBJ_XBODY, objname=f'{arm}_{body}')
    # Tip linear velocity, after the contiguous per-arm position blocks.
    for arm in ARMS:
        spec.add_sensor(name=f'{arm}_ee_vel', type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f'{arm}_ee')
    return spec.compile()


class DualArm:
    """Shared accessors of a two-arm robot. Subclasses set model, data, plan_model,
    home_q, joint_qadr, joint_dadr, arm_act and ee_site."""

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
        """Translational tip Jacobian (3 x ndof) over the arm's joints."""
        d = self.data if data is None else data
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, jacp, None, self.ee_site[arm])
        return jacp[:, self.joint_dadr[arm]]

    def jacobian_rot(self, arm, data=None):
        """Tip translational and rotational Jacobians (3 x ndof each) and the tool rotation matrix."""
        d = self.data if data is None else data
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, jacp, jacr, self.ee_site[arm])
        dofs = self.joint_dadr[arm]
        return jacp[:, dofs], jacr[:, dofs], self.tool_rot(arm, d)

    def tool_rot(self, arm, data=None):
        return (self.data if data is None else data).site_xmat[self.ee_site[arm]].reshape(3, 3).copy()

    def state(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        out = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, d, out, spec)
        return out


class DualFranka(DualArm):
    def __init__(self, base_distance: float = 1.5, scene_xml: Path = SCENE_XML, tool: str = 'stick',
                 bases: dict | None = None):
        self.model = build_model(base_distance, scene_xml, tool, bases)
        self.data = mujoco.MjData(self.model)
        # Model that MPPI rolls out; a task may set a cheaper copy with the same state layout.
        self.plan_model = self.model
        self.tool = tool
        self.home_q = HOME_Q  # posture the MPPI null space is pulled towards
        self.proxy_bodies = PROXY_BODIES if tool == 'stick' else GRIPPER_PROXY_BODIES
        m = self.model
        self.joint_qadr, self.joint_dadr, self.arm_act, self.ee_site = {}, {}, {}, {}
        for arm in ARMS:
            joints = [m.joint(f'{arm}_joint{k}') for k in range(1, 8)]
            self.joint_qadr[arm] = np.array([j.qposadr[0] for j in joints])
            self.joint_dadr[arm] = np.array([j.dofadr[0] for j in joints])
            self.arm_act[arm] = np.array([m.actuator(f'{arm}_actuator{k}').id for k in range(1, 8)])
            self.ee_site[arm] = m.site(f'{arm}_ee').id
        self.grip_act = {arm: m.actuator(f'{arm}_actuator8').id for arm in ARMS} if tool == 'gripper' else {}
        self.finger_qadr = {arm: np.array([m.joint(f'{arm}_finger_joint{k}').qposadr[0] for k in (1, 2)])
                            for arm in self.grip_act}
        self.target_mocap = {arm: m.body(f'{arm}_target').mocapid[0] for arm in ARMS}
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        # Each arm's sensors are contiguous: tip first, then the clearance proxies.
        self.sensor_ee = {arm: self.sensor_slice(f'{arm}_ee_pos') for arm in ARMS}
        self.sensor_ee_vel = {arm: self.sensor_slice(f'{arm}_ee_vel') for arm in ARMS}
        self.sensor_arm = {arm: slice(self.sensor_ee[arm].start,
                                      self.sensor_slice(f'{arm}_{self.proxy_bodies[-1]}_pos').stop) for arm in ARMS}
        self.reset()

    def reset(self, q_home: np.ndarray = HOME_Q):
        mujoco.mj_resetData(self.model, self.data)
        for arm in ARMS:
            self.data.qpos[self.joint_qadr[arm]] = q_home
            self.data.ctrl[self.arm_act[arm]] = q_home
        for arm, act in self.grip_act.items():  # open
            self.data.qpos[self.finger_qadr[arm]] = 0.04
            self.data.ctrl[act] = GRIP_OPEN
        mujoco.mj_forward(self.model, self.data)

    def set_target_marker(self, arm, xyz):
        self.data.mocap_pos[self.target_mocap[arm]] = xyz
