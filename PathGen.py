from Tools import * 
import numpy as np 
import math 

class PathGenerator: 

    def __init__(self): 

        # misc self items 
        self.N = 200; 
        self.inpx = np.expand_dims(np.linspace(0, 2*np.pi, self.N), 1)
        # Theta inputs 
        self.Nth = 5
        self.dth = 0/self.Nth

        # Radius inputs 
        self.Nr = 8
        self.dr = 0/self.Nr

    def get_path(self, size): 
        # Initialize vectors 
        theta_start = np.random.rand(1)
        theta = np.linspace(theta_start, theta_start + 2*np.pi, num=self.N)
        r = np.ones(theta.shape)

        # Randomize theta 
        for i in range(self.Nth): 
            a = (np.random.rand(1)-.5) * self.dth
            # theta += a * np.sin(i * self.inpx)
            theta = theta + (1 * np.sin(i * self.inpx))

        print((theta + (np.sin(i * self.inpx))).size)

        # finalize theta 
        theta = theta * np.sign(np.random.rand(1)-.5)  # Randomly flip the direction
        theta = theta[:-1]  # Cut off the last place

        # Randomize radius 
        for i in range(self.Nr): 
            a = (np.random.rand(1)-.5) * self.dr
            r = r + (a * np.sin(i*self.inpx))

        # Finalize radius
        r = r[:-1]
        # print(f"Radius size: {r.shape}")

        # print(r.size)

        # Swap into coordinate system 
        x = size * r * np.cos(theta)
        y = size * r * np.sin(theta) 

        x = np.squeeze(x)
        y = np.squeeze(y)

        return (x, y)


        


        