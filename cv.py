import cv2
import numpy as np
import multiprocessing
import time
from move_angular_and_cartesian import *

shared_white = multiprocessing.Array('f', [0, 0])
my_shared_variable_white = tuple(shared_white)
shared_blue = multiprocessing.Array('f', [0, 0])
my_shared_variable_blue = tuple(shared_blue)
# white = (0,0)
# blue = (0,0)

def cv():
    # Define the lower and upper bounds for the white and yellow color ranges in the HSV color space
    lower_white = np.array([100, 0, 160])
    upper_white = np.array([130, 90, 256])
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])


    # Create a video capture object for the camera
    cap = cv2.VideoCapture(0)

    transform = 0

    def transform_pixel_to_cm(pixel_x, pixel_y, transform):
        return pixel_x*transform, pixel_y*transform


    # Continuously read frames from the camera
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV frame to detect white and yellow color ranges
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        # print(mask_white)
        mask_yellow = cv2.inRange(hsv, lower_blue, upper_blue)

        # Combine the white and yellow binary masks using the bitwise OR operation
        mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

        # Find contours in the white binary mask
        contours_white, hierarchy = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find contours in the yellow binary mask
        contours_yellow, hierarchy = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        yellow_centers = []
        # Filter contours based on shape and area in the yellow binary mask
        for cnt in contours_yellow:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            if len(approx) == 4 and area > 900:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.putText(frame, 'Object', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                yellow_centers.append((int(x+w/2), int(y+h/2)))

                transform = 5.5/w

                # print("Blue Top-left corner: ", top_left)
                # print("Blue Bottom-right corner: ",  bottom_right)

                # print("Blue Top-left corner: ", transform_pixel_to_cm(top_left[0], top_left[1], transform))
                # print("Blue Bottom-right corner: ",  transform_pixel_to_cm(bottom_right[0], bottom_right[1], transform))

        # Filter contours based on shape and area in the white binary mask
        white_centers = []

        for cnt in contours_white:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            if len(approx) == 4 and area > 900:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
                cv2.putText(frame, 'Target Area', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                white_centers.append((int(x+w/2), int(y+h/2)))

                # print("White Top-left corner: ", top_left)
                # print("White Bottom-right corner: ", bottom_right)
        

        if len(white_centers) > 0 and len(yellow_centers) > 0:
            white_center = white_centers[0]
            yellow_center = yellow_centers[0]
            cv2.line(frame, white_center, yellow_center, (0, 0, 255), 2)

            # Calculate the Euclidean distance between the center points of the white and yellow bounding boxes
            distance = cv2.norm(np.array(white_center)-np.array(yellow_center))
            cv2.putText(frame, "Distance: {:.2f}".format(distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            global my_shared_variable_white
            global my_shared_variable_blue
            with shared_white.get_lock():
                shared_white[0] = white_center[0]
                shared_white[1] = white_center[1]
            my_shared_variable_white = tuple(shared_white)
            with shared_blue.get_lock():
                shared_blue[0] = yellow_center[0]
                shared_blue[1] = yellow_center[1]
            my_shared_variable_blue = tuple(shared_blue)
            print(my_shared_variable_white, my_shared_variable_blue)
            # return white_center,yellow_center


        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(cv2.waitKey(1), f'break ********* {0xFF=}')
            break

        #do_stuff()

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

from typing import Tuple,List

class Pixel_to_cm:
    def __init__(self):
        self.transform_matrix = None

    def fit(self, 
                 origin_pixel: Tuple[float], 
                 ref_points_pixel:List[Tuple[float]],
                 ref_points_cm:List[Tuple[float]]):

        self.dim = len(origin_pixel)
        # create a matrix [point1.T point2.T point3.T]
        ref_matrix_pixel = np.array(ref_points_pixel).T  
        # create a matrix [point1.T point2.T point3.T]
        ref_matrix_cm = np.array(ref_points_cm).T
        # create origin point as column vector (n, 1)
        self.origin_pixel = np.array(origin_pixel)

        assert ref_matrix_cm.shape == ref_matrix_pixel.shape == (self.dim, self.dim),\
            f"The number of reference points {len(ref_matrix_pixel)=}  {len(ref_matrix_pixel)=} should match the number of dimensions {self.dim}."
        try:
            self.transform_matrix = ref_matrix_cm @ np.linalg.inv(ref_matrix_pixel - self.origin_pixel.reshape(-1,1))
        except np.linalg.LinAlgError as e:
            print(
                'The vectors connecting reference points and origin point should be linear independent.'
            )
            raise e
        
        return self

    def transform(self, point_pixel: Tuple[float]) -> Tuple[float]:
        assert self.transform_matrix is not None, 'You need to fit the transformer first.'
        assert len(point_pixel) == self.dim, 'The input point should have the same dimensions.'

        return tuple(self.transform_matrix @ (np.array(point_pixel) - self.origin_pixel))

    def inverse_transform(self, point_cm: Tuple[float]) -> Tuple[int]:
        assert self.transform_matrix is not None, 'You need to fit the transformer first.'
        assert len(point_cm) == self.dim, 'The input point should have the same dimensions.'
        temp = np.round(np.linalg.inv(self.transform_matrix) @ point_cm + self.origin_pixel\
                        ).astype(int)
        return tuple(temp)
        

origin_pixel = ( 200 , 135 )
ref_p1_pixel, ref_p2_pixel = ( 757 , 115), ( 788 , 628)
ref_p1_cm, ref_p2_cm = (40,0), (40 , 40)
transformer = Pixel_to_cm().fit(origin_pixel, [ref_p1_pixel, ref_p2_pixel], [ref_p1_cm, ref_p2_cm])

def do_stuff(base, base_cyclic):
    global my_shared_variable_white
    global my_shared_variable_blue
    global transformer
    print(my_shared_variable_white, my_shared_variable_blue)

    print(transformer.transform(my_shared_variable_white))
    print(transformer.transform(my_shared_variable_blue))

    blue = transformer.transform(my_shared_variable_blue)
    white = transformer.transform(my_shared_variable_white)
    # sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    # import utilities

    # # Parse arguments
    # args = utilities.parseConnectionArguments()
    
    # # Create connection to the device and get the router
    # with utilities.DeviceConnection.createTcpConnection(args) as router:

    #     # Create required services
    #     base = BaseClient(router)
    #     base_cyclic = BaseCyclicClient(router)

        # Example core
    success = True

    tilting_gripper = {'x': 0,
                'y': 0,
                'z': 0,
                'theta_x': 90,
                'theta_y': 0,
                'theta_z': 0}

    movement = {'x': -(0.01 * blue[1] - 0.09),
                'y': (0.54 - 0.01 *blue[0]),
                'z': 0,
                'theta_x': 0,
                'theta_y': 0,
                'theta_z': 0}

    move_grip_down = {'x': 0,
                'y': 0,
                'z': -0.3,
                'theta_x': 0,
                'theta_y': 0,
                'theta_z': 0}
    move_grip_down_two = {'x': 0,
                'y': 0,
                'z': -0.08,
                'theta_x': 0,
                'theta_y': 0,
                'theta_z': 0}
    move_grip_up = {'x': 0,
                'y': 0,
                'z': 0.15,
                'theta_x': 0,
                'theta_y': 0,
                'theta_z': 0}

    second_move = {'x': -0.01*(white[1]-blue[1]),
                    'y': -0.01*(white[0]- blue[0]),
                    'z': 0,
                    'theta_x': 0,
                    'theta_y': 0,
                    'theta_z': 0}

    success &= ExampleSendGripperCommands(base,0.1)
    success &= example_move_to_home_position(base)
    success &= example_cartesian_action_movement(base, base_cyclic, tilting_gripper)
    # Move gripper to location of object
    success &= example_cartesian_action_movement(base, base_cyclic, move_grip_down)
    success &= example_cartesian_action_movement(base, base_cyclic, movement)
    success &= example_cartesian_action_movement(base, base_cyclic, move_grip_down_two)

    # Gripper grip object
    success &= ExampleSendGripperCommands(base, 0.5)
    example_cartesian_action_movement(base, base_cyclic, move_grip_up)
    example_cartesian_action_movement(base, base_cyclic, move_grip_up)
    example_cartesian_action_movement(base, base_cyclic, second_move)
    example_cartesian_action_movement(base, base_cyclic, move_grip_down_two)
    ExampleSendGripperCommands(base, 0.1)

        # success &= example_cartesian_action_movement(base, base_cyclic, move_grip_up)
        # success &= example_cartesian_action_movement(base, base_cyclic, reverse_movement)

        # success &= example_cartesian_action_movement(base, base_cyclic, second_move)
        # success &= example_cartesian_action_movement(base, base_cyclic, move_grip_down)

        # Gripper release object
        # success &= ExampleSendGripperCommands(base, 0.1)

        # You can also refer to the 110-Waypoints examples if you want to execute
        # a trajectory defined by a series of waypoints in joint space or in Cartesian space

    return 0 if success else 1

if __name__=='__main__':
    # print(transformer.transform(( 757 , 115)))
    # exit(0)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    while user_input:= input("Would you like arm to continue? Y for Yes, Anything else for No ") == "Y":
        # cv()
        # thread1 = multiprocessing.Process(target=cv)
        # thread2 = multiprocessing.Process(target=do_stuff)
        # thread1.start()
        # time.sleep(3)
        # thread2.start()
        
        # Create connection to the device and get the router
        with utilities.DeviceConnection.createTcpConnection(args) as router:

            # Create required services
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)

            # Example core
            success = True
            success &= example_move_to_home_position(base)
            cv()

            do_stuff(base, base_cyclic)

        # wait for threads to finish
        # thread1.join()
        # thread2.join()

        # global my_shared_variable_blue
        # print(my_shared_variable_blue)

    # print(transformer.transform(( 405 , 338 )))
    
    # print(transformer.inverse_transform((0, 0)))
    # print(transformer.transform((235, 34)))
