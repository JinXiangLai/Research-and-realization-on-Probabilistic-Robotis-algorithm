import numpy as np
import math
from gravity_evaluate import quaternion_matrix, angle_velocity2rotation, hat

data_folder = ''
data_file_name = data_folder + 'vel_imu_simulate.csv'


def content_gyro_acc(t, gyro, acc) -> str:
    content = 'IMU ' + str(t)
    for i in gyro:
        content += ' ' + str(i)
    for i in acc:
        content += ' ' + str(i)
    return content + '\n'


def content_vel_quat(t, v, quat) -> str:
    content = 'Novetal ' + str(t)
    for i in v:
        content += ' ' + str(i)
    for i in quat:
        content += ' ' + str(i)
    return content + '\n'


def quaternion_from_matrix(matrix, isprecise=False):
    '''
        return [qx, qy, qz, qw]
    '''
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    q = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64, copy=True)
    q = q/np.linalg.norm(q)
    return q


ax = 1
ay = 0.001
az = 9.8
sigma_ax = 0.1
sigma_ay = 0.01
sigma_az = 0.001

wx = 0.0001
wy = 0.001
wz = 0.1
sigma_wx = 0.0001
sigma_wy = 0.001
sigma_wz = 0.01


def generate_acc_data():
    a0 = ax + np.random.normal(loc=0, scale=sigma_ax)
    a1 = ay + np.random.normal(loc=0, scale=sigma_ay)
    a2 = az + np.random.normal(loc=0, scale=sigma_az)
    return np.array([a0, a1, a2])


def generate_gyro_data():
    w0 = wx + np.random.normal(loc=0, scale=sigma_wx)
    w1 = wy + np.random.normal(loc=0, scale=sigma_wy)
    w2 = wz + np.random.normal(loc=0, scale=sigma_wz)
    return np.array([w0, w1, w2])


def calculate_vel_acc(v, q, gyro, acc, delta_t, g):
    R = quaternion_matrix(q)
    v = v + R.dot(acc) * delta_t + g * delta_t
    R = R.dot(angle_velocity2rotation(gyro, delta_t))
    H = np.eye(4)
    H[:3, :3] = R
    q = quaternion_from_matrix(H)
    return v, q


def main():
    # Macro parameters
    sample_time = 0.01  # 10 ms
    vel = np.array([0., 0., 0.])

    quat = np.array([0., 0., 0., 1.])
    g = np.array([0, 0, -9.8])
    generate_data_num = 10000
    timestamp = 0

    with open(data_file_name, 'w') as file:
        initial_content = '#' + 'type ' + 'timestamp ' + 'content\n'
        file.write(initial_content)
        w = np.array([0., 0., 0.])
        a = -1.0 * g
        sample_freq = 3
        for i in range(generate_data_num):
            if i % sample_freq:
                # sample data
                w = generate_gyro_data()
                a = generate_acc_data()
                content = content_gyro_acc(timestamp, w, a)
                file.write(content)

            # calculate velocity and rotation
            vel, quat = calculate_vel_acc(vel, quat, w, a, sample_time, g)
            if i % sample_freq == 0:
                content = content_vel_quat(timestamp, vel, quat)
                file.write(content)

            timestamp += sample_time


if __name__ == '__main__':
    main()
