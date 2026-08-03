from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
import time


ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_GOAL_PWM = 100
ADDR_PRESENT_PWM = 124
ADDR_PRESENT_CURR = 126
ADDR_PRESENT_POSITION = 132
ADDR_MOVING_STATUS = 122
ADDR_OPERATING_MODE = 11
ADDR_RETURN_DELAY_TIME = 9
ADDR_PWM_LIMIT = 36

XL430_ADDR_POS_D_GAIN = 80
XL430_ADDR_PROFILE_ACCELERATION = 108
XL430_ADDR_PROFILE_VELOCITY = 112

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

class DXL:
    def __init__(self, device_name, protocol_version, baudrate, dxl_id):
        # Initialize PortHandler instance
        self.portHandler = PortHandler(device_name)

        # Initialize PacketHandler instance
        self.packetHandler = PacketHandler(protocol_version)
        
        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(baudrate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()
            
        self.dxl_id = dxl_id

        self.status_return_level = 2  # Default Value
        self.set_status_return_level(self.status_return_level)

    
    def set_pwm_limit(self, pwm_limit_value):
        '''
        Establishes a limit to the PWM for PWM mode or Position modes
        '''
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.dxl_id, ADDR_PWM_LIMIT, pwm_limit_value)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            print("Failed to set PWM Limit.")
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print("Failed to set PWM Limit")


    def move_pwm(self,pwm_value):
        '''
        USE WHEN OPERATING PWM MODE
        Inputting a value 0-885 will cause the motor to spin counter-clockwise.
        Inputting a value of 0-(-885) will cause the motor to spin clockwise.
        '''
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.dxl_id, ADDR_GOAL_PWM, pwm_value)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            print("Failed to engage PWM.")
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print("Failed to engage PWM")

    def calibrate_motor(self):
        '''
        For motor calibration, uses PWM from an open position to establish an open and closed position value.
        Returns two values, open_gripper_value and close_gripper_value.
        '''
        # Close Gripper Calibration Block
        self.disabled_torque()
        self.set_pwm_limit(300)
        self.set_operating_mode(16)
        self.enable_torque()
        self.move_pwm(300)  # Sets the speed value to negative in order to go CCL
        time.sleep(.5)
        while(self.check_is_moving()):
            self.get_present_load()
            time.sleep(.01)
        self.move_pwm(0)
        self.disabled_torque()
        self.set_operating_mode(4)
        self.enable_torque()
        close_gripper_value = self.get_present_position()

        #Open Gripper Calibration Block
        self.disabled_torque()
        self.set_operating_mode(16)
        self.enable_torque()
        self.move_pwm(-300)
        counter = 0
        time.sleep(1)
        while True:
    
    
            if counter > 100:
                print("Stall! Gripper Opened!")
                break
            counter += 1
            time.sleep(0.01)
        
        self.move_pwm(0)
        self.disabled_torque()
        self.set_operating_mode(4)
        self.enable_torque()
        open_gripper_value = self.get_present_position()

        return open_gripper_value, close_gripper_value

    def set_operating_mode(self, operating_mode=3):
        '''
        Within Dynamixel motors using Protocol 2, sets the operating mode to Extended Position Mode via the 11 address.
        Operating_mode inputs are as follows:
        1. Velocity Control Mode (1)
        3. Position Control Mode (Default)(3)
        4. Extended Control Mode (4)
        16. PWM Control Mode (16)
        '''
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.dxl_id, ADDR_OPERATING_MODE, operating_mode)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else: 
            print("Succeeded to set operating mode.")
       
    # Set return delay time     
    def set_return_delay_time(self, return_delay_time):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.dxl_id, ADDR_RETURN_DELAY_TIME, int(return_delay_time))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
    
    # Enable Dynamixel Torque
    def enable_torque(self):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel has been successfully connected")
    
    # Disable Dynamixel Torque
    def disabled_torque(self):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            
    # Set D Gain
    def set_pos_d_gain(self, d_gain):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.dxl_id, XL430_ADDR_POS_D_GAIN, int(d_gain))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            
    # Set Profile Acceleration
    def set_profile_acceleration(self, acceleration):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.dxl_id, XL430_ADDR_PROFILE_ACCELERATION, int(acceleration))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
    
    # Set Profile Velocity
    def set_profile_velocity(self, velocity):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.dxl_id, XL430_ADDR_PROFILE_VELOCITY, int(velocity))
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
    
    # Read present position
    def get_present_position(self) -> int:
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, self.dxl_id, ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            print("Failed to get position.")
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print("Failed to get position)")
            
        return dxl_present_position
    
    def get_present_load(self):
        dxl_present_cur, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, self.dxl_id, ADDR_PRESENT_CURR)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            print("Failed to get load.")
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print("Failed to get load)")
        # print("Current Load is:", dxl_present_cur, "mA")
        return dxl_present_cur
    
    def check_is_moving(self):
        '''
        Enables transmission for one byte to the motion address, returning the value of 0 or 1
        '''
        dxl_moving, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, self.dxl_id, ADDR_MOVING_STATUS)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            print("Failed to check movement.")
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))  
            print("Failed to check movement.")      
        return dxl_moving
    
    def move_to(self, goal_position):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.dxl_id, ADDR_GOAL_POSITION, goal_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def set_status_return_level(self, level=1):
        """
        Set status return level:
        0 - Do not respond to any instructions
        1 - Respond only to READ instructions
        2 - Respond to all instructions (default)
        """
        ADDR_STATUS_RETURN_LEVEL = 68
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.dxl_id, ADDR_STATUS_RETURN_LEVEL, level)
        self.status_return_level = level


    def move_to_nonblocking(self, goal_position):
        if self.status_return_level != 1:
            self.set_status_return_level(1)

        self.packetHandler.write4ByteTxOnly(self.portHandler, self.dxl_id, ADDR_GOAL_POSITION, goal_position)

            
    def disable(self):
        self.disabled_torque()

        # Close port
        self.portHandler.closePort()
