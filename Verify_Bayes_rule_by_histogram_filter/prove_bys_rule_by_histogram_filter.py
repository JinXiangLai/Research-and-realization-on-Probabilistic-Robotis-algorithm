# 使用一维直方图滤波算法证明贝叶斯概率必须融合先前的后验概率p(z|x){观测独立性假设}，且必须使用运动模型更新当前状态、后验概率，
# 直方图滤波首先要离散化状态空间，然后代入状态转移方程，更新状态和先验概率，
# 最后再使用量测模型更新后验概率。
# 和粒子滤波的不一样在于，直方图滤波需要已知机器人的整个状态空间取值，如一维机器人位置可以离散化出来，
# 如下例子，机器人在一维空间运动，其状态被均匀离散为0.1m区间的小块，亦即定位精度为0.1m。
# 但是粒子滤波适用于机器人状态空间未知，且无法穷举尽等场景
import numpy as np
from matplotlib import pyplot as plt

door_pos = [1.0, 2.0, 5.0]
corridor_length = 4.8
# an interval is 0.1m which means the localization accurancy is 0.1m
histgram_interval = 0.1
interval_num = corridor_length / histgram_interval
motion_step = 0.1
# robot go from 0.5m
true_state = 0.5

interval_weight = [1.0]
interval_pos = [0.0]

# assume that the robot can only observe the nearest door in front of it.
def find_min_positive_value(obvs:list) -> float:
    min_positive = -1.0
    for obv in obvs:
        if obv > 0:
            min_positive = obv
    
    for i in range(len(obvs)):
        if obvs[i] > 0 and obvs[i] < min_positive:
            min_positive = obvs[i]
    
    # because the robot can only see the door in front of it
    if min_positive < 0:
        return 1e20
    return min_positive

for i in range(1, int(interval_num)):
    interval_weight.append(1.0)
    interval_pos.append(interval_pos[i-1] + histgram_interval)

# change to numpy
interval_weight = np.array(interval_weight)

while true_state < corridor_length:
    # update current true state
    true_state += motion_step
    # the sensor output
    obvs = np.array(door_pos) - true_state
    obv_min_positive_dist = find_min_positive_value(obvs)
    print("true min_positive_dist: ", obv_min_positive_dist)

    # prediction progress:
	# we should first update the robot state based on the motion model
    interval_obv = []
    for i in range(len(interval_pos)):
        interval_pos[i] += motion_step
        obvs = np.array(door_pos) - interval_pos[i]
        ## the distance between the nearest door in front of the robot and the robot
        min_positive_dist = find_min_positive_value(obvs)
        interval_obv.append(min_positive_dist)
    interval_obv = np.array(interval_obv) + 1e-5


    # ok, now we have updated the robotic state and possess the current measurement,
    # it's time to update the weight of histogram intervals
    diff = abs(interval_obv - obv_min_positive_dist)
    obv_weights = 1.0 / diff
    # use bayes rule，p(x|z) ∝ p(x) * p(z|x)
	# for that: p(x|y) = p(x) * p(y|x) / p(y) ==>
	# p(x) ∝ p(x|y)
    # 观测的条件独立性
    interval_weight *= obv_weights
    # no use bayes rule, p(x|y) = p(y|x), ERROR!!!
    # interval_weight = obv_weights
    
    # normalize weight
    interval_weight = interval_weight / np.sum(interval_weight)

    plt.plot(interval_pos, interval_weight)
    plt.scatter(np.array([true_state]), np.array([np.max(interval_weight)]), c='red' )
    plt.xlim((interval_pos[0], interval_pos[-1]))
    plt.ylim((0,1))
    plt.xticks(interval_pos[::3])
    plt.title("pos and weight")
    plt.show()
    plt.close()
