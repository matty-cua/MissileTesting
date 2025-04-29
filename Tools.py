import numpy as np 
import math
import random 

class Tools: 

    @staticmethod
    # === Helper functions 
    def random_unit(seed=None): 
        return Vector(
            (random.random()-.5), 
            (random.random()-.5)
        ).unit()
    
    @staticmethod 
    def get_cam_pos(screen_size, position):
        x = (position.x + screen_size[0])/2
        y = (position.y + screen_size[1])/2
        return Vector(x, y)
    
class Debug: 

    enable_log = True
    enable_warn = True 
    enable_error = True 

    @staticmethod
    def log(s): 
        if Debug.enable_log: 
            print(s)

    @staticmethod 
    def warn(s): 
        if Debug.enable_warn: 
            print(s)

    @staticmethod
    def error(s): 
        if Debug.enable_error: 
            print(s)


class Vector: 

    # NOTE: We work in RADIANS not DEGREES 
    
    def __init__(self, x=None, y=None): 
        if x is None: 
            self.x = 0, 
            self.y = 0
        elif type(x) == Vector: 
            self.x = x.x
            self.y = x.y
        else: 
            self.x = x
            self.y = y 
    
    # operations 

    def __mul__(self, a): 
        return Vector(self.x*a, self.y*a)
    
    def __add__(self, a): 
        if type(a) == Vector: 
            return Vector(self.x + a.x, self.y + a.y)
        else: 
            return Vector(self.x + a, self.y + a)
        
    def __sub__(self, a): 
        return self + (a * -1); 
        
    def __truediv__(self, a): 
        return self * (1/a)
    
    def rotate(self, b): 
        m = self.magnitude()
        a = self.angle() + b
        x = math.cos(a) * m
        y = math.sin(a) * m
        return Vector(x, y)
    
    def as_tuple(self): 
        return (self.x, self.y)

    # properties 

    def __repr__(self): 
        return (f"({self.x}, {self.y})")

    def angle(self): 
        v = self.unit()
        th = math.acos(v.x)
        if v.y < 0: 
            th = 2*math.pi - th
        return th

    def magnitude(self): 
        return math.sqrt(self.x**2 + self.y**2)

    def unit(self): 
        return self / self.magnitude()
    
    # Statics 
    @staticmethod
    def distance(a, b): 
        d = a - b; 
        return math.sqrt(d.x**2 + d.y**2); 
    
    @staticmethod 
    def dot(a, b): 
        return (a.x*b.x) + (a.y*b.y)
    
    @staticmethod
    def angle_between(a, b): 
        return math.acos(Vector.dot(a, b) / (a.magnitude() * b.magnitude()))

    @staticmethod
    def projection(a, b): 
        """
        Project vector a along vector b
            Pretty much the opposite of torque 
        """
        return Vector.dot(a, b.unit()) 
    
    @staticmethod
    def off_axis(a, b): 
        proj = Vector.projection(a, b)
        if proj > a.magnitude():  # Can happen due to machine precision errors 
            return 0
        return math.sqrt(a.magnitude()**2 - proj**2)
    
    @staticmethod
    def to_axis(a, b): 
        bu = b.unit()  # unit vector of the direction 
        inline = projection()






# End of class 