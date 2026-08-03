
from dynamixel_sdk import COMM_SUCCESS # Uses Dynamixel SDK library
from robot.arm.dxl import DXL

MY_DXL = 'X_SERIES'
BAUDRATE = 1000000 # 115200

# https://emanual.robotis.com/docs/en/dxl/protocol2/
PROTOCOL_VERSION = 2.0

# Factory default ID of all DYNAMIXEL is 1
DXL_ID = 3

# Use the actual port assigned to the U2D2.
DEVICENAME = '/dev/ttyUSB0'

MIN_TOTAL_LOOPS = 15


class Gripper:
    def __init__(self, baudrate=BAUDRATE, device_name=DEVICENAME, protocol_version=PROTOCOL_VERSION, dxl_id=DXL_ID):
        self.dxl = DXL(device_name, protocol_version, baudrate, dxl_id)

        self.dxl.set_return_delay_time(0)
        self.dxl.enable_torque()
        self.dxl.set_pos_d_gain(0)
        self.dxl.set_profile_acceleration(50)
        self.dxl.set_profile_velocity(1000)

    def move_to_pos(self, goal_position):

        self.dxl.move_to_nonblocking(goal_position)

        # if abs(self.dxl.get_present_position() - goal_position) < 10:
        #     pass

        # loops = 0
        # prev_position = float('-inf')
        # tot_loops = 0
        
        # while True:
        #     dxl_present_position = self.dxl.get_present_position()

        #     if abs(dxl_present_position - goal_position) < 10:
        #         break

        #     if abs(dxl_present_position - prev_position) < 5 and tot_loops > MIN_TOTAL_LOOPS:
        #         loops += 1
                
        #     if loops > 5:
        #         self.dxl.move_to(dxl_present_position)
        #         break
        #     prev_position = dxl_present_position
            
        #     tot_loops += 1

    #    dxl_present_position = self.dxl.get_present_position()
        
     #   print("Desired position:%03d  Final position:%03d" % (goal_position, dxl_present_position))
        
    def disable(self):
        self.dxl.disable()

if __name__ == "__main__":
    left_gripper = Gripper(baudrate=BAUDRATE, dxl_id=3)
