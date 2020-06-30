import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clustering import clustering

plt.figure()
fig = plt.gcf()
ax = plt.gca()
plt.ion()

df_lidar = pd.read_csv('scenario_1\scan.csv', sep=',')
df_lidar['%time'] = (df_lidar['%time'] - df_lidar['%time'].tolist()[0])/1000000000
df_lidar.set_index('%time',inplace=True)
df_lidar.columns = [name[6:] for name in df_lidar]

k = 0
scan = [[],[]]

theta = 0.
x = np.array([0., 0., theta]).reshape(-1,1)

def method(x, p_fix, p_r):
    n = p_fix.shape[1]
    semble2 = []
    semble = []
    for i in range(n):
        semble.append(p_fix[0][i])
        semble.append(p_fix[1][i])
        semble2.append(p_r[0][i])
        semble2.append(p_r[1][i])
    semble2 = np.array(semble2).reshape(-1,1)  
    semble = np.array(semble).reshape(-1,1)
    p = np.zeros((n*2, n*2))
    for i in range(1,len(p)):
        if i % 2 == 1:
            p[i-1, i-1] = np.cos(x[2])
            p[i-1, i] = np.sin(x[2])
            p[i, i-1] = -np.sin(x[2])
            p[i, i] = np.cos(x[2])
    d = np.zeros((n*2, 1))
    for i in range(2*n):
        if i % 2 == 0:
            d[i] = semble[i] - x[0]
        else:
            d[i] = semble[i] - x[1]
    b = semble2 - p @ d
    A = np.zeros((n*2, 3))
    for i in range(1, n):
        if i % 2 == 1:
            A[i - 1] = [-np.cos(x[2]), -np.sin(x[2]), -np.sin(x[2]) * (semble[i - 1] - x[0]) + np.cos(x[2]) * (semble[i] - x[1])]
            A[i] = [np.sin(x[2]), -np.cos(x[2]), -np.cos(x[2]) * (semble[i - 1] - x[0]) - np.sin(x[2]) * (semble[i] - x[1])]
    res = np.linalg.inv(A.T @ A) @ A.T @ b
    x += res
    return x, res

d = 0
for time, row in df_lidar.iterrows():
    n = 360  # <-- number of points
    ranges = np.array(row[10:10+n])
    intensities = np.array(row[10+n:10+2*n])

    angle = np.arange(0, 360, 1.0) / 180.0 * np.pi
    vec = np.array([np.cos(angle), np.sin(angle)])

    point = vec * ranges
    rot = np.array([[np.cos(-np.pi/2), np.sin(-np.pi/2)], [-np.sin(-np.pi/2), np.cos(-np.pi/2)]])
    
    point = rot @ point
    scan[0] = point[0,:].tolist()
    scan[1] = point[1,:].tolist()

    (clusters_id, clusters) = clustering(np.array(scan))
    
    M = clusters[3]           ### choisir la meme couleur
    p_r = M.points
    rot1 = np.array([[np.cos(-d), np.sin(-d)], [-np.sin(-d), np.cos(-d)]])
    M.points = rot1 @ M.points
    M.points[0,:] += 1 + x[0]  ### points de coordonnées dans un repère fixe R0
    M.points[1,:] += 1 + x[1]
    x, res = method(x, M.points, p_r)
    
    d = float(res[2,:])        ### le changement de angle 
    print(x)
    ax = plt.gca()
    for clust in clusters :
        rect = plt.Rectangle(xy=(clust.center[0] - clust.width/2, clust.center[1] - clust.length/2) ,width=clust.width, height=clust.length, linewidth=1, color='red', fill=False)
        ax.add_artist(rect)
    plt.scatter(scan[0], scan[1], c=clusters_id)
    plt.draw()
    plt.pause(0.01)
    plt.clf()










