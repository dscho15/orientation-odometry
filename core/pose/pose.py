import numpy as np 
import g2o
from core.pose.pose_utils import quaternion_slerp

class Pose(object):
    def __init__(self, pose=None):

        if pose is None: 
            pose = g2o.Isometry3d()      

        self.set(pose)
        self.covariance = np.identity(6)
        
    def set(self, pose: g2o.Isometry3d):
        
        if isinstance(pose, g2o.SE3Quat) or isinstance(pose, g2o.Isometry3d):
            self._pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self._pose = g2o.Isometry3d(pose)                         
            
        self.Tcw = self._pose.matrix()
        self.Rcw = self.Tcw[:3,:3]
        self.tcw = self.Tcw[:3,3]
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)
        
    def update(self, pose):
        self.set(pose)
        
    @property    
    def isometry3d(self):
        return self._pose 
    
    @property    
    def quaternion(self):
        return self._pose.orientation()  
    
    @property    
    def orientation(self):
        return self._pose.orientation()     
    
    @property    
    def position(self):
        return self._pose.position()        
    
    def get_rotation_angle_axis(self):
        angle_axis = g2o.AngleAxis(self._pose.orientation())
        return angle_axis  
    
    def get_inverse_matrix(self):
        return self._pose.inverse().matrix()     
              
    def set_from_quaternion_and_position(self,quaternion,position):
        self.set(g2o.Isometry3d(quaternion, position))       
        
    def set_from_matrix(self, Tcw):
        self.set(g2o.Isometry3d(Tcw))
        
    def set_from_rotation_and_translation(self, Rcw, tcw): 
        self.set(g2o.Isometry3d(g2o.Quaternion(Rcw), tcw))     
        
    def set_quaternion(self, quaternion):
        self.set(g2o.Isometry3d(quaternion, self._pose.position()))  
                
    def set_rotation_matrix(self, Rcw):
        self.set(g2o.Isometry3d(g2o.Quaternion(Rcw), self._pose.position()))  
        
    def set_translation(self, tcw):
        self.set(g2o.Isometry3d(self._pose.orientation(), tcw))
        
    def __repr__(self) -> str:
        return "Pose: \n" + str(self._pose.matrix())        
        
        
if __name__ == "__main__":
    
    # create a pose 
    g2o_pose = g2o.Isometry3d(g2o.Quaternion(1, 0, 0, 0), np.array([1, 0, 0]))
    
    pose = Pose(g2o_pose)
    print(pose)
    
    print(pose.Ow)
    print(pose.Rcw)
    print(pose.tcw)