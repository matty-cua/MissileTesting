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
        self.speed = 100; 
        self.turn_speed = 1 * math.pi  # rads/second
        self.clip_distance = 100  # clip dd vector to be unit vector outside of this distance 

    def reset(self): 
        # Should reset the model here (if it is stateful)
        ... 

        # # Kinematic reset is handled by envrionment 
        # self.position = Tools.random_unit()
        # self.velocity = Tools.random_unit * self.speed

    def update(self, dt, target, reward=None): 
        # Turning 
        self.turn_towards(target, dt)

        # Kinematics 
        self.position += self.velocity * dt
        # print("Pos: " + str(self.position))
        # print("Vel: " + str(self.velocity / dt))
        # print("Spd: " + str(self.velocity.magnitude() / dt)) 

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
        angle = Vector.angle_between(self.velocity, dd)  # angle between our velocity and the target 
        v_inline = Vector.projection(self.velocity, dd)
        v_offline = Vector.off_axis(self.velocity, dd)
        v_proj = Vector(v_inline, v_offline).unit()

        # Pass to model 
        axis = self.model(np.array([dd.x, dd.y, v_proj.x, v_proj.y]))

        # rotate acording to output 
        self.velocity = self.velocity.rotate(axis * dt * self.turn_speed)

    def model(self, inp): 
        if inp[2] > 0: 
            out = -inp[3]
        else: 
            if inp[3] < 0: 
                out = 1
            else: 
                out = -1
        
        print(f"in : {inp[2]}")
        print(f"off: {inp[3]}")
        print(f"out: {out}")
        return(out)

    def backprop(self, reward): 
        pass 
    


# End of class 