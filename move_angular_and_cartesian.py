#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
 
def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def example_cartesian_action_movement(base, base_cyclic, movement = {'x': 0,
                                                                    'y': 0,
                                                                    'z': 0,
                                                                    'theta_x': 0,
                                                                    'theta_y': 0,
                                                                    'theta_z': 0}):
    print(movement)
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x + movement['x']        # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y + movement['y']    # (meters)
    cartesian_pose.z = feedback.base.tool_pose_z + movement['z']    # (meters)
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x + movement['theta_x'] # (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y + movement['theta_y'] # (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z + movement['theta_z'] # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished


def ExampleSendGripperCommands(base, move = 0.1):

    # Create the GripperCommand we will send
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()

    # Close the gripper with position increments
    print("Performing gripper test in position...")
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    position = 0.00
    finger.finger_identifier = 1
    while position < move:
        finger.value = position
        print("Going to position {:0.2f}...".format(finger.value))
        base.SendGripperCommand(gripper_command)
        position += 0.1
        time.sleep(1)

    # Set speed to open gripper
    # print ("Opening gripper using speed command...")
    # gripper_command.mode = Base_pb2.GRIPPER_SPEED
    # finger.value = 0.1
    # self.base.SendGripperCommand(gripper_command)
    gripper_request = Base_pb2.GripperRequest()

    # Wait for reported position to be opened
    # gripper_request.mode = Base_pb2.GRIPPER_POSITION
    # while True:
    #     gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
    #     if len (gripper_measure.finger):
    #         print("Current position is : {0}".format(gripper_measure.finger[0].value))
    #         if gripper_measure.finger[0].value < 0.01:
    #             break
    #     else: # Else, no finger present in answer, end loop
    #         break

    # Set speed to close gripper
    # print ("Closing gripper using speed command...")
    # gripper_command.mode = Base_pb2.GRIPPER_SPEED
    # finger.value = -0.1
    # self.base.SendGripperCommand(gripper_command)

    # Wait for reported speed to be 0
    gripper_request.mode = Base_pb2.GRIPPER_SPEED
    while True:
        gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
        if len (gripper_measure.finger):
            print("Current speed is : {0}".format(gripper_measure.finger[0].value))
            if gripper_measure.finger[0].value == 0.0:
                break
        else: # Else, no finger present in answer, end loop
            break
    return True

def main():
    
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Example core
        success = True

        tilting_gripper = {'x': 0,
                    'y': 0,
                    'z': 0,
                    'theta_x': 90,
                    'theta_y': 0,
                    'theta_z': 0}

        movement = {'x': 0.1,
                    'y': 0.54,
                    'z': -0.3,
                    'theta_x': 0,
                    'theta_y': 0,
                    'theta_z': 0}
        reverse_movement = {'x': 0.07,
            'y': -0.47,
            'z': 0.3,
            'theta_x': 0,
            'theta_y': 0,
            'theta_z': 0}

        move_grip_down = {'x': 0,
                    'y': 0,
                    'z': -0.1,
                    'theta_x': 0,
                    'theta_y': 0,
                    'theta_z': 0}
        move_grip_up = {'x': 0,
                    'y': 0,
                    'z': 0.1,
                    'theta_x': 0,
                    'theta_y': 0,
                    'theta_z': 0}

        second_move = {'x': -0.39,
                        'y': 0.39,
                        'z': -0.25,
                        'theta_x': 0,
                        'theta_y': 0,
                        'theta_z': 0}

        success &= ExampleSendGripperCommands(base,0.1)
        success &= example_move_to_home_position(base)
        success &= example_cartesian_action_movement(base, base_cyclic, tilting_gripper)
        # Move gripper to location of object
        # success &= example_cartesian_action_movement(base, base_cyclic, movement)
        # success &= example_cartesian_action_movement(base, base_cyclic, move_grip_down)

        # # Gripper grip object
        # success &= ExampleSendGripperCommands(base, 1)
        # success &= example_cartesian_action_movement(base, base_cyclic, move_grip_up)

        # success &= example_cartesian_action_movement(base, base_cyclic, second_move)
        # success &= example_cartesian_action_movement(base, base_cyclic, move_grip_down)

        # Gripper release object
        # success &= ExampleSendGripperCommands(base, 0.1)

        # You can also refer to the 110-Waypoints examples if you want to execute
        # a trajectory defined by a series of waypoints in joint space or in Cartesian space

        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
