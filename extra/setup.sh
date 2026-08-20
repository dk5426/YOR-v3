#!/bin/bash

# echo "Enter the name of the can interface (can0, e.g.):"
# read interface

interface=can0

# Check if the interface exists
if ! ip link show "$interface" &> /dev/null; then
    echo "Interface $interface does not exist."
    echo "If you have a physical CAN adapter, make sure it's connected."
    echo "If you want to create a virtual CAN interface for testing, use:"
    echo "sudo ip link add dev $interface type vcan"
    exit 1
fi

# Try to set the interface type and bitrate
if sudo ip link set $interface type can bitrate 1000000; then
    echo "Set $interface type to CAN with 1Mbps bitrate."
else
    echo "Failed to set $interface type and bitrate."
    echo "If this is a virtual CAN interface, you can skip this step."
fi

# Bring up the interface
if sudo ip link set $interface up; then
    echo "Brought up $interface."
else
    echo "Failed to bring up $interface."
    exit 1
fi

# Set txqueuelen
if sudo ip link set $interface txqueuelen 1000; then
    echo "Set txqueuelen to 1000."
else
    echo "Failed to set txqueuelen."
    exit 1
fi

echo "CAN interface $interface setup complete."

sleep 0.1

sudo ip link set can1 down
sudo ip link set can1 name can_left
sudo ip link set up can_left type can bitrate 1000000
sleep 0.1

sudo ip link set can2 down
sudo ip link set can2 name can_right
sudo ip link set up can_right type can bitrate 1000000
sleep 0.1


#python3 home_gripper.py
sleep 0.1

#python3 piper_reset.py
