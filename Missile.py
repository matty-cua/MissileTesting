import numpy as np 
# import tensorflow as tf 

import random 
import math 

from Tools import * 

class Missile: 

    def __init__(self): 
        # attributes 
        self.position = Vector()
        self.velocity = Vector()

        # # Maybe don't implement this? 
        # self.model = None  # Model to train 

        # behaviors 
        self.speed = 300; 
        self.turn_speed = 1 * math.pi  # rads/second
        self.clip_distance = 200  # clip dd vector to be unit vector outside of this distance 

    def reset(self): 
        # Should reset the model here (if it is stateful)
        ... 

        # # Kinematic reset is handled by envrionment 
        # self.position = Tools.random_unit()
        # self.velocity = Tools.random_unit * self.speed

    def update(self, dt, target, action=None, reward=None): 
        # Turning 
        if action is None: 
            self.turn_towards(target, dt)
        else: 
            self.velocity = self.velocity.rotate(action * dt * self.turn_speed)

        # Kinematics 
        self.position += self.velocity * dt

        if reward is not None: 
            self.backprop(reward)

    def turn_towards(self, target, dt): 
        # Get distance inputs 
        dd = target.position - self.position  # distance vector 
        dm = dd.magnitude()  # distance magnitude 

        # Clip distance for better input to NN 
        if dm > self.clip_distance: 
            dd = dd.unit() * self.clip_distance
        else: 
            dd *= dm / self.clip_distance

        # Get observations 
        vel_angle = self.velocity.angle()  # angle of velo vector (to world)
        dd_angle = dd.angle()  # Angle of distance vector (to world)
        angle = vel_angle - dd_angle  # Difference between angles 
        # Wrap angle betweeen [-180, 180]
        if angle > 180:  # Fix upper bounds 
            angle = 360 - angle
        if angle < -180:  # Fix lower bounds
            angle = 360 + angle
        v_inline = Vector.projection(self.velocity, dd)  # Velocity towards the target 
        v_offline = np.float64(Vector.off_axis(self.velocity, dd) * math.copysign(1, angle))  # Velocity away from the target 
        v_proj = Vector(v_inline, v_offline).unit()  # velocity normalized to the target 

        # Pass to model 
        # [print(type(_)) for _ in (dd.x[0], dd.y[0], v_proj.x, v_proj.y[0])]
        # axis = self.model(np.array([dd.x[0], dd.y[0], v_proj.x, v_proj.y[0]]))
        axis = self.model(np.array([dd.x, dd.y, v_proj.x, v_proj.y]))

        # rotate acording to output 
        self.velocity = self.velocity.rotate(axis * dt * self.turn_speed)

    def get_obs(self, target, dt): 
        # Get distance inputs 
        dd = target.position - self.position  # distance vector 
        dm = dd.magnitude()  # distance magnitude 

        # Clip distance for better input to NN 
        if dm > self.clip_distance: 
            dd = dd.unit()
        else: 
            dd /= self.clip_distance

        # relative velocity 
        dv = self.velocity - target.velocity 

        # Get observations 
        vel_angle = self.velocity.angle()  # angle of velo vector (to world)
        dd_angle = dd.angle()  # Angle of distance vector (to world)
        angle = vel_angle - dd_angle  # Difference between angles 

        # Wrap angle betweeen [-180, 180]
        if angle > 180:  # Fix upper bounds 
            angle = 360 - angle
        if angle < -180:  # Fix lower bounds
            angle = 360 + angle

        v_inline = Vector.projection(dv, dd)  # Velocity towards the target 
        v_offline = np.float64(Vector.off_axis(dv, dd) * math.copysign(1, angle))  # Velocity away from the target 
        v_proj = Vector(v_inline, v_offline).unit()  # velocity normalized to the target 

        # Get distance relative to velocity (the nose) 
        d_inline = Vector.projection(dd, self.velocity)
        d_offline = np.float64(Vector.off_axis(dd, self.velocity) * math.copysign(1, angle))
        d_proj = Vector(d_inline, d_offline).unit()
        d_mag = dd.magnitude()

        return np.array([d_proj.x, d_proj.y, v_proj.x, v_proj.y, d_mag])    

    def model(self, inp): 
        """
        Here we run a model that uses the inputs from the turn_towards call to determine how much to turn 
        """
        if inp[2] > 0: 
            out = -inp[3]
        else: 
            if inp[3] < 0: 
                out = 1
            else: 
                out = -1
        
        # print(f"in : {inp[2]}")
        # print(f"off: {inp[3]}")
        # print(f"out: {out}")
        return(out)

    def backprop(self, reward): 
        pass 
    


# End of class 