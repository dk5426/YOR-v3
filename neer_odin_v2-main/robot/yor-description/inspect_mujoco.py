import mujoco
m = mujoco.MjModel.from_xml_path('/Users/dhawalkabra/Documents/GitHub/YOR/robot/yor-description/robot_mujoco.xml')
d = mujoco.MjData(m)
mujoco.mj_kinematics(m, d)

for i in range(m.nbody):
    name = m.body(i).name
    pos = d.xpos[i]
    print(f"Body: {name}, Pos: {pos}")
