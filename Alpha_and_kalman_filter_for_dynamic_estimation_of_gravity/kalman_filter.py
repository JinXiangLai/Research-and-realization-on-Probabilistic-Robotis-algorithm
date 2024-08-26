import numpy as np
import math
from gravity_evaluate import str_list2float_list, quaternion_matrix, angle_velocity2rotation

_EPS = np.finfo(float).eps * 4.0

data_folder = ''
data_file_name = data_folder + 'vel_imu_simulate.csv'


class KalmanFilter(object):
    '''
        initial:
            v0
            g0
            cov0
            Q
            R
        predict:
                 v_k+1
        	x_k+1          = F * x_k + B * u
                 g_k+1
            P_k+1 = F * P_k * F.transpose() + B * Q * B.transpose()
            # also use Q_=B * Q * B.transpose()
        update:
        	v_z_theory = h(v_k+1) -- here liner observation function H = I ---- observation function
            
			K = P_k+1 * H.transpose() * (H * P_k+1 * H.transpose() + V).inverse()
			x_k+1` = x_k+1 + K(v_z_measurement - h(v_k+1))
			P_k+1` = (I - K*H) * P_k+1
	'''

    def __init__(self, vel_imu_data: list):
        # The most important step -- set covariance etc.
        # initial state variables covariance, relationship between state variables is important
        self.P_k0_ = np.eye(6)
        self.P_k0_[:3, :3] *= 0.1
        self.P_k0_[3:, 3:] *= 0.01

        # movement input covariance
        self.acc_noise_ = 0.1#0.1
        self.gyro_noise_ = 0.001#0.01 
        self.Q_ = np.eye(3)
        self.Q_ *= self.acc_noise_

        # observation covariance
        obs_vel_noise = 0.01
        self.R_ = np.eye(6)
        self.R_ *= obs_vel_noise

        # define initial state variables
        self.v_k0_ = np.zeros(3)
        self.R_k0_ = np.eye(3)
        self.g_k0_ = np.array([0, 0, -9.0])

        self.x_k0_ = np.zeros(6)
        self.x_k0_[:3] = self.v_k0_
        self.x_k0_[3:] = self.g_k0_

        # define movement matrix
        self.A_ = np.eye(6)

        # define movement input matrix
        self.B_ = np.zeros((6, 3))
        self.B_[:3, :3] = np.eye(3)

    def run(self, vel_imu_data):
        # some control variables
        last_time = 0
        self.acc_ = np.zeros(3)
        self.gyro_ = np.zeros(3)
        idx = -1

        # initialization
        while (idx < len(vel_imu_data) - 1):
            idx += 1
            if (vel_imu_data[idx][0] == 'Novetal'):
                data = vel_imu_data[idx][1]
                last_time = data[0]
                self.v_k0_ = np.array(data[1:4])
                self.x_k0_[:3] = self.v_k0_
                self.R_k0_ = quaternion_matrix(np.array(data[4:]))
                break

        # process data
        while (idx < len(vel_imu_data) - 1):
            idx += 1
            data = vel_imu_data[idx][1]
            delta_t = data[0] - last_time
            last_time = data[0]

            if (vel_imu_data[idx][0] == 'IMU'):
                if 'simulate' in data_file_name:
                    self.gyro_ = np.array(data[1:4]) + np.array(
                        np.random.normal(loc=0, scale=self.gyro_noise_, size=3))
                    self.acc_ = np.array(data[4:]) + np.array(
                        np.random.normal(loc=0, scale=self.acc_noise_, size=3))
                else:
                    self.gyro_ = np.array(data[1:4])
                    self.acc_ = np.array(data[4:])
                self.predict(self.gyro_, self.acc_, delta_t)

            elif (vel_imu_data[idx][0] == 'Novetal'):
                v_z = np.array(data[1:4])  # 当前观测速度
                R_z = quaternion_matrix(np.array(data[4:]))
                self.update(self.gyro_, self.acc_, delta_t, v_z, R_z)

                print('g:', self.x_k0_[3:].transpose(), np.linalg.norm(self.x_k0_[3:]))

    def predict(self, gyro, acc, delta_t):
        u = (self.R_k0_.dot(acc) + self.x_k0_[3:]) * delta_t
        self.x_k0_ = self.A_.dot(self.x_k0_) + self.B_.dot(u)
        self.R_k0_ = self.R_k0_.dot(angle_velocity2rotation(gyro, delta_t))
        self.P_k0_ = self.A_.dot(self.P_k0_).dot(
            self.A_.transpose()) + self.B_.dot(self.Q_).dot(
                self.B_.transpose())

    def update(self, gyro, acc, delta_t, v_z, R_z):
        x_obs = np.zeros(6)
        x_obs[:3] = v_z
        x_obs[3:] = self.x_k0_[3:]
        if (delta_t > 0.):
            x_obs[3:] = (v_z - self.x_k0_[0:3])/delta_t - self.R_k0_.dot(acc)

        if (delta_t > 0.):
            self.predict(gyro, acc, delta_t)
		# direct observation function
        H = np.eye(6)
        # kalman gain
        K = self.P_k0_.dot(H.transpose()).dot(
            np.linalg.inv(H.dot(self.P_k0_).dot(H.transpose()) + self.R_))

        self.x_k0_ = self.x_k0_ + K.dot(x_obs - H.dot(self.x_k0_))
        self.P_k0_ = (np.eye(6) - K.dot(H)).dot(self.P_k0_)
        # self.x_k0_[3:] = self.x_k0_[3:]/np.linalg.norm(self.x_k0_[3:]) * 9.8

        self.R_k0_ = R_z


def main():
    vel_imu_data = []
    print('\n********** Reading data file **********\n')
    with open(data_file_name) as file:
        vel_imu_data = file.readlines()
    vel_imu_data = str_list2float_list(vel_imu_data)  # OK

    print(vel_imu_data[0])

    kf = KalmanFilter(vel_imu_data)
    kf.run(vel_imu_data)


if __name__ == '__main__':
    main()
