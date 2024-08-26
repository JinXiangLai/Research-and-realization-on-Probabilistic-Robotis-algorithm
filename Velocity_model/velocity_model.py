import random
from math import pi, sin, cos
import copy
from matplotlib import pyplot as plt
import numpy as np
RADIUS = 1.0
ANGLE2RAD = pi / 180
RAD2DEGREE = 180 / pi
SAMPLE_TIME = 1.0
INITIAL_POSE = [90 * ANGLE2RAD, RADIUS, 0]
ANGEL_VELOCITY = [1.0 * ANGLE2RAD, 2.0 * ANGLE2RAD, 3.0 * ANGLE2RAD]

def GenerateSensorData():
    total_angle = 0.0
    angel_vel= []
    linear_vel =[]
    while total_angle < 360.0: 
        id = random.randint(0, 2)
        w = ANGEL_VELOCITY[id]
        v = w * RADIUS
        angel_vel.append(w)
        linear_vel.append(v)
        total_angle += abs(w * SAMPLE_TIME * RAD2DEGREE)
    return angel_vel, linear_vel

def UpdateCurrentPose(w, v, pose):
        # use velocity model introduced by 《Probabilistic Robotics》
        # r = abs(v / w)
        # for the direction of the v must point to the circle center
        # and this direction is decided by the direction of the w
        # for example, the positive v with a positive w and the positive v with a negative w
        # will point to the different circle center
        r = v / w 
        x_c = pose[1] - r * sin(pose[0])
        y_c = pose[2] + r * cos(pose[0])
        # update orientation
        pose[0] += w * SAMPLE_TIME
        # update position
        pose[1] = x_c + r * sin(pose[0])
        pose[2] = y_c - r * cos(pose[0])

def main():
    w_data, v_data = GenerateSensorData()
    # [θ, x, y]
    pose = copy.deepcopy(INITIAL_POSE) 
    x_data = [pose[1]]
    y_data = [pose[2]]
    for i in range(len(w_data)):
        w = w_data[i]
        v = v_data[i]
        UpdateCurrentPose(w, v, pose)
        x_data.append(pose[1])
        y_data.append(pose[2])
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    plt.plot(x_data, y_data)
    plt.xlim((-RADIUS, RADIUS))
    plt.ylim((-RADIUS, RADIUS))
    plt.legend("Velocity model")
    plt.show()

if __name__ == '__main__':
    main()
