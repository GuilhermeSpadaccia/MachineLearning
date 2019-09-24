import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cosine_similarity(vec1, vec2, vec3=None):
    if vec3 is None:
        return np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    else:
        return np.sum(vec1*vec2*vec3)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)*np.linalg.norm(vec3))
    
# First 2D vector (must be on origin)
x1 = [0, 3]
y1 = [0, 3]
# Second 2D vector (must be on origin)
x2 = [0, -4]
y2 = [0, 3]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.quiver([x1[0],x2[0]], [y1[0],y2[0]], [x1[1],x2[1]], [y1[1],y2[1]], color=['r','b','g'], scale=1, scale_units='xy', angles='xy')

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
plt.savefig('2d_vectors.png')

# Calculation of 2d vectors similarity
cos_sim = cosine_similarity(np.array([x1[1], y1[1]]), np.array([x2[1], y2[1]]))
print("2D cosine similarity: ", cos_sim)

# First 3D vector (must be on origin)
x1 = [0, 3]
y1 = [0, 2]
z1 = [0, 3]
# Second 3D vector (must be on origin)
x2 = [0, 1]
y2 = [0, 1]
z2 = [0, 3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver([x1[0],x2[0]], 
          [y1[0],y2[0]], 
          [z1[0],z2[0]], 
          [x1[1],x2[1]], 
          [y1[1],y2[1]], 
          [z1[1],z2[1]])

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
plt.savefig('3d_vectors.png')

# Calculation of 3d vectors similarity
cos_sim = cosine_similarity(np.array([x1[1], y1[1], z1[1]]), np.array([x2[1], y2[1], z2[1]]))
print("3D cosine similarity: ", cos_sim)