import numpy as np

# Calcul la distance du point pM au segment [p1, p2]
def point2seg(pM, p1, p2):
    vec = p2 - p1
    vec /= np.sqrt(vec[0]**2+vec[1]**2)
    vecT = np.array([-vec[1], vec[0]])
    return np.abs(np.sum((pM-p1)*vecT))


class Cluster:
    def __init__(self, points):
        self.points = np.array(points).T  

        self.center = np.array([
            np.mean(self.points[0,:]),
            np.mean(self.points[1,:])
        ])

        self.width = np.max(self.points[0,:]) - np.min(self.points[0,:]) # along the x coordinate
        self.length = np.max(self.points[1,:]) - np.min(self.points[1,:]) # along the y coordinate

        self.eps = 0.05
        self.poly = np.array(self.rdp(self.points, self.eps))

    def rdp(self, points, eps):
        points = np.array(points)
        Pres = [[],[]]
        dmax = 0
        jmax = 0
        
        l = points.shape[0]

        for i in range(1, l - 1):
            if dmax < point2seg(points[:,i], points[:,0], points[:,l - 1]):
                dmax = point2seg(points[:,i], points[:,0], points[:,l - 1])
                jmax = i
        if dmax > eps:
            Pres1 = self.rdp(points[:jmax], eps)
            Pres2 = self.rdp(points[jmax:], eps)
            Pres[0] = Pres1[:,:Pres1.shape[1]]
            Pres[1] = Pres2
        else:
            Pres = [points[:,0], points[:,-1]]
        return Pres


def clustering(points):
    k = 3
    D = 0.1
    G = np.zeros(len(points[0]), dtype=int)
    Gp = []
    
    g = 0
    for i in range(k, points.shape[1]):
        dmin = 100000
        jmin = 0
        
        for j in range(1, k + 1):
            if dmin > np.linalg.norm(points[:, i] - points[:, i-j]):
                dmin = np.linalg.norm(points[:, i] - points[:, i-j])
                jmin = j
        if dmin < D:
            if G[i - jmin] == 0:
                g += 1
                Gp.append([])
                G[i - jmin] = g
                Gp[g - 1].append(points[:, i - jmin])
            G[i] = G[i - jmin]
            Gp[G[i - jmin] - 1].append(points[:, i])

    clusters = [Cluster(np.array(cluster_p)) for cluster_p in Gp]
    return G, clusters

