import numpy as np

# Calculate gravity at dynamic situation

g_rig = np.array([3, 4, 5])
g_guess = np.array([0, 0, 9.8])
v_last = np.array([3, 0, 0])
delta_t = 10 * 1e-3
acc_measure = np.array([0.5, 0.3, 0.01])
acc_noise = np.array([0.1, 0.1, 0.1])

alpha = 0.5
belta = 1 - alpha

def generate_obv_v(v0, acc, delta_t):
    v = v0 + (acc + g_rig) * delta_t
    return v



def main(data_num):
    global v_last
    global alpha
    global belta
    global g_guess
    acc_list = []
    obv_v_list = [v_last]

    for i in range(0, data_num):  
        acc = np.random.normal(loc=0, scale=acc_measure, size=3)
        obv_v = generate_obv_v(v_last, acc, delta_t)
        v_last = obv_v
        # write
        acc_list.append(acc)
        obv_v_list.append(obv_v)

    use_direct_obv = True
    g_last = 0
    for i in range(0, len(acc_list)):
        if not use_direct_obv:
            v_predict = obv_v_list[i] + (acc_list[i] + g_guess + np.random.normal(loc=0, scale=acc_noise, size=3)) * delta_t
            v_obv = obv_v_list[i + 1]
            v_diff = v_obv - v_predict
            g_diff = v_diff / delta_t
            # x_pred + (1-α)(x_obv - x_pred) = α * x_pred + β * x_obv
            g_last = g_guess
            g_guess = g_guess + belta * g_diff

        else:
            acc = acc_list[i]
            v_diff = obv_v_list[i+1] - obv_v_list[i]
            g_obv = v_diff / delta_t - acc
            # x_pred + (1-α)(x_obv - x_pred) = α * x_pred + β * x_obv
            g_last = g_guess
            g_guess = alpha * g_guess + belta * g_obv 

        if not i%10:
            print("iterate %d times, g: %s"%(i, g_guess))
        
        g_diff = g_last - g_guess
        if np.linalg.norm(g_diff) < np.linalg.norm(acc_noise)*0.2:
            print("Converge after %d iterations!\n true g:%s\n guess g:%s\n diff g:%s"%(i, g_rig, g_guess, np.linalg.norm(g_rig-g_guess)))
            break

if __name__=="__main__":
    main(100)


