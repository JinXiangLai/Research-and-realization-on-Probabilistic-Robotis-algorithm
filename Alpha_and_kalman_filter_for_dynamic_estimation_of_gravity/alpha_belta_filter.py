import numpy as np
import math

_EPS = np.finfo(float).eps * 4.0

data_folder = ''
data_file_name = data_folder + 'vel_imu_simulate.csv'


def str_list2float_list(all_data):
    result = []
    for item in all_data:
        if item[0][0] == '#':
            continue
        data = item.split(' ')  # throw \n and split number
        data_type = data[0]
        data = list(map(float, data[1:]))  # str to float
        result.append((data_type, data))
    return result


def quaternion_matrix(quaternion):
    # change [qx, qy, qz, qw] to [qw, qx, qy, qz]
    q = quaternion
    q = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64, copy=True)
    # q = np.array(quaternion, dtype=np.float64, copy=True)
    # 归一化
    q_norm = np.linalg.norm(q, ord=2)
    q = q / q_norm
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
         [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
         [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])


def hat(v: np.ndarray) -> np.ndarray:
    # yapf: disable
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])
    # yapf: enable


def angle_velocity2rotation(w, delta_t):
    wx = w[0] * delta_t
    wy = w[1] * delta_t
    wz = w[2] * delta_t

    # Rodrigues formula
    d2 = wx * wx + wy * wy + wz * wz
    d = np.sqrt(d2)

    v = np.array([wx, wy, wz])
    W = hat(v)
    eps = 1e-4
    if (d < eps):
        deltaR = np.eye(3) + W
    else:
        deltaR = np.eye(3) + W * np.sin(d) / d + W.dot(W) * (1.0 -
                                                             np.cos(d)) / d2
    return deltaR

def main():
    vel_imu_data = []
    print('\n********** Reading data file **********\n')
    with open(data_file_name) as file:
        vel_imu_data = file.readlines()
    vel_imu_data = str_list2float_list(vel_imu_data)  # OK

    print(vel_imu_data[0])

    v_k0 = 0
    R_k0 = 0
    g_k0 = np.array([0, 0, -9.8])
    t_k0 = 0
    acc = 0

    idx = -1
    alpha = 0.1
    belta = 0.1

    acc_noise = 0.1
    gyro_noise = 0.01

    # initialization
    while (idx < len(vel_imu_data) - 1):
        idx += 1
        if (vel_imu_data[idx][0] == 'Novetal'):
            data = vel_imu_data[idx][1]
            t_k0 = data[0]
            v_k0 = np.array(data[1:4])
            R_k0 = quaternion_matrix(np.array(data[4:]))
            break

	# process data one by one
    v_update = 0
    while (idx < len(vel_imu_data) - 1):
        idx += 1
        data = vel_imu_data[idx][1]
        delta_t = data[0] - t_k0
        t_k0 = data[0]

        # predict
        if (vel_imu_data[idx][0] == 'IMU'):
            if 'simulate' in data_file_name:
                gyro = np.array(data[1:4]) + np.array(
                    np.random.normal(loc=0, scale=gyro_noise, size=3))
                acc = np.array(data[4:]) + np.array(
                    np.random.normal(loc=0, scale=acc_noise, size=3))
            else:
                gyro = np.array(data[1:4])
                acc = np.array(data[4:])

            # predict velocity
            v_k0 = v_k0 + R_k0.dot(acc) * delta_t + g_k0 * delta_t
            # predict rotarion
            R_k0 = R_k0.dot(angle_velocity2rotation(gyro, delta_t))
            # g_k0 = g_k0

        # update
        elif (vel_imu_data[idx][0] == 'Novetal'):
            v0 = v_k0
            R0 = R_k0

            if delta_t > 0:
                v_k0 = v_k0 + R_k0.dot(
                    acc) * delta_t + g_k0 * delta_t
            v_z = np.array(data[1:4]) 
            diff = v_z - v_k0
            v_update = v_k0 + alpha * diff

            v_diff = v_update - v0
            g_k0 = (1 - belta) * g_k0 + belta * (v_diff / delta_t -
                                                 R0.dot(acc))
            # normalization no need actually
            # g_k0 = g_k0 / np.linalg.norm(g_k0) * 9.8

            v_k0 = v_update
            R_k0 = quaternion_matrix(np.array(data[4:]))
            print('g:', g_k0.transpose(), np.linalg.norm(g_k0))

if __name__ == '__main__':
    main()