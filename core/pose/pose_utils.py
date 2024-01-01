import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def quaternion_slerp(q1: np.array, q2: np.array, t: float):
    dot_product = np.dot(q1, q2)
    
    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product
        
    if dot_product > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot_product)
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q1) + (s1 * q2)

def ms(x, y, z, radius, resolution=20):
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    
    return (X, Y, Z)

def visualize_slerp_3D(q1: np.ndarray, q2: np.ndarray, n: np.ndarray = 100):
    
    t = np.linspace(0, 1, 100)
    
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    q = np.zeros((len(t), 4))
    
    for i in range(len(t)):
        q[i] = quaternion_slerp(q1, q2, t[i])
        
    x_pns_surface, y_pns_surface, z_pns_surface = ms(0, 0, 0, 1)
        
    fig = go.Figure(data=[go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode='markers'),
                          go.Surface(x=x_pns_surface, y=y_pns_surface, z=z_pns_surface, opacity=0.25)])
    
    fig.show()

    
    
if __name__ == "__main__":
    visualize_slerp_3D(np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), 100)