# 二值贝叶斯滤波，一般我们求取后验状态bel(x_t) = p(x_t|z_t)
# p(x|z) = p(x)*p(z|x)/p(z)，但是在二值滤波中，求p(x|z)往往比p(z|x)简单，
# 譬如从图像推断门开、闭状态x比从门的状态x推断相机测量z的像素值简易得多
# 二值贝叶斯滤波中，有
# l(x) = log(p(x)/(1-p(x)))
# bel(x_t) = 1 - 1/(1+exp(l_t)) 
# 作为一种非参数滤波器，其与KF的区别在于，不需要假设噪声服从高斯分布，不需要线性化状态转移、测量函数
# p(x|z_0~z_t) = p(z_t|x, z0~z_t-1) * p(x|z0~z_t-1) / p(z_t|z0~z_t-1) ==>
#              = p(z_t|x) * p(x|z0~z_t-1) / p(z_t|z0~z_t-1)

from math import log, exp
from numpy import random
# the observation probablity
door_open = 0.8
door_close = 0.8
# 处理每个传感器测量时，状态x的先验分布估计，这一般是无法预知的，
# 只有在处理第0个测量时可以给出
l_0 = log(door_open / (1-door_open))
l_t = l_0
obv_time = 10
open_time = 0
for i in range(obv_time):
	obv = random.randint(0, 2)
	l_x = 0.0
	# 每一次量测更新前的先验p(x)无法预知，只有p(x|z_t)可知，根据贝叶斯准则:
	# p(z_t|x) = p(z_t) * p(x|z_t) / p(x) ==>
	# p(x) = p(z_t) * p(x|z_t) / p(z_t|x) : 难以计算
	prior_x = 0.0
	if obv:
		l_x = log(door_open / (1-door_open))
		open_time += 1
	else:
		l_x = log((1-door_close)/door_close)
	# log 运算使得乘法变加法, l_t取值为[-∞, +∞]
	l_t = l_t + l_x - prior_x

p_t = 1.0 - 1.0 / (exp(l_t) + 1.0)
print("total obv time = %d, obv open time = %d, l_t = %f, p_t = %f"%(obv_time, open_time, l_t, p_t))
