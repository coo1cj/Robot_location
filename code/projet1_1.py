import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_lidar = pd.read_csv('scenario_1\scan.csv', sep=',')      #### changer le numero de scenario
df_lidar['%time'] = (df_lidar['%time'] - df_lidar['%time'][0]) / 10 ** 9
df_lidar.set_index('%time', inplace=True)
df_lidar.columns = [name[6:] for name in df_lidar]


df_encoder = pd.read_csv('scenario_1\sensor_state.csv', sep=',')  #### changer le numero de scenario
r = 0.033
L = 0.08
df_encoder['%time'] = (df_encoder['%time'] - df_encoder['%time'][0]) / 10 ** 9
dl = df_encoder['field.left_encoder'].tolist()
dr = df_encoder['field.right_encoder'].tolist()

plt.figure(figsize=(6,15))

def afficher_tra(dl, dr):
    g = [(dl[i+1] - dl[i])/0.05 * (2*np.pi / 4096) for i in range(len(df_encoder['%time'])-1)]
    d = [(dr[i+1] - dr[i])/0.05 * (2*np.pi / 4096) for i in range(len(df_encoder['%time'])-1)]
    v = np.array([r*(g[i] + d[i])/2 for i in range(len(g))])
    w = np.array([r*(d[i] - g[i])/(2*L) for i in range(len(g))])
    vitess = v.tolist()
    theta = w.tolist()
    x = []
    y = []
    x1 = 0
    t1 = 0
    y1 = 0
    for i in range(len(vitess)):
        t1 += theta[i]*0.05
        x1 += 0.05*vitess[i]*np.cos(t1)
        y1 += 0.05*vitess[i]*np.sin(t1)
    x = [i for i in x] 
    y = [-i for i in y]
    x = np.array(x)
    y = np.array(y)
    xy = np.vstack((x,y))
    plt.plot(xy[1,:], xy[0,:], label='encodeur')


def draw_env(df, dl, dr):
    afficher_tra(dl, dr)
    i = 0
    for t, row in df.iterrows():
        ranges = np.array([row[10:10+360]])
        angle = np.arange(0, 360, 1.0) / 180.0 * np.pi
        vec = np.array([np.cos(angle), np.sin(angle)])
        point = vec * ranges
        rot = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)], [np.sin(np.pi / 2), np.cos(np.pi / 2)]])
        
        point = rot @ point
        if i == 0:
            plt.scatter(point[0], point[1], c='r')
            plt.xticks([])
            plt.yticks([])
            plt.legend()
            plt.show()
        elif i >= 200:
            plt.scatter(point[0], point[1])
            plt.arrow(0, 0, 0, -0.2, head_width=0.01, length_includes_head=True, color='red')
            plt.xticks([])
            plt.yticks([])
            plt.draw()
            plt.pause(0.01)
            plt.clf()
        else:
            plt.scatter(point[0], point[1])
            plt.arrow(0, 0, 0, 0.2, head_width=0.01, length_includes_head=True, color='red')
            plt.xticks([])
            plt.yticks([])
            plt.draw()
            plt.pause(0.01)
            plt.clf()
        i += 1        



#### centrale ineritelle
df_imu = pd.read_csv('scenario_1\imu.csv', sep=',') ### changer le numero de scenario
ang_z = df_imu['field.angular_velocity.z'].tolist()

liner_x = df_imu['field.linear_acceleration.x'].tolist() #accélération longitudinal (m/s^2)
liner_z = df_imu['field.linear_acceleration.z'].tolist() #accélération verticale (m/s^2)

v = 0
x = []
y = []
x1 = 0
t1 = 0
t2 = 0
y1 = 0
for i in range(len(liner_x)):
    t1 += ang_z[i] * 0.0077   ### 120.99s / 15811 = 0.0077 intervalle de temps de 
    v = liner_x[i] * 0.0077   ### chaque enregistrement(senario 2)
    x1 += 0.05*v*np.cos(t1) 
    x.append(x1)
    y1 += 0.05*v*np.sin(t1)
    y.append(y1)
y = [-i for i in y]

x = np.array(x)
y = np.array(y)
xy = np.vstack((y,x))
plt.plot(xy[0,:], xy[1,:], label='centrale ineritelle')
draw_env(df_lidar, dl, dr)
plt.show()