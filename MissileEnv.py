import numpy as np 

try: 
    import gymnasium as gym
    gm = True
except: 
    gm = False

try: 
    import pygame
    pg = True  
except: 
    pg = False 

import random 
from dataclasses import dataclass 

from Tools import * 
from Missile import Missile
from PathGen import PathGenerator

# class MissileEnv(_sub): 
class MissileEnv: 

    def __init__(self): 
        # Important behavior vars 
        self.move_target = True
        self.GP = PathGenerator()
        self.training_length_frames = 10*30

        # Gen variables 
        self.dt = 1/30;  # Frame rate 
        self.bound_size = 100; 
        self.training = True; 
        self.frames = 0; 

        # Target variables 
        self.target = Target() 

        # Missile variables 
        self.missile = Missile()

        # Manage pygame 
        self.follow_missile = True  # Center rocket on the frame (instead of (0, 0))
        self.rendered = False  # if we have initiaized the pygame window 
        self.running = False  # Unsure if this is necessary either, most likely pauses things? 
        self.window_size = (400, 400) 
        self.render_mode = None 

        # Rendering variables 
        self.target_color = (255, 100, 100)  # A nice light pink? 
        self.missile_color = (50, 50, 50)  # Dark gray 
        self.missile_width = 4
        self.missile_size = 20
        self.target_size = 10 

    def reset(self, seed=None, options=None): 
        # super stuff 
        try: 
            super().reset(seed=seed)
        except Exception as e: 
            pass 

        # admin stuff 
        self.frames = 0

        # Random target position, zero velocity
        if not self.move_target:  
            self.target.velocity = Vector(0, 0)
            self.target.position = Tools.random_unit() * self.bound_size
        else: 
            self.target_x, self.target_y = self.GP.get_path(self.bound_size)
            self.target.velocity = Vector(0, 0) 
            self.target_i = 0
            self.target.position = Vector(self.target_x[self.target_i], self.target_y[self.target_i])

        # Random missile location, random velocity direction 
        self.missile.position = Tools.random_unit() * self.bound_size
        self.missile.velocity = Tools.random_unit() * self.missile.speed 
        self.missile.reset()

    def step(self, action=None): 
        # Basic inits 
        terminated = False 
        truncated = False 

        # update target kinematics 
        if self.move_target: 
            self.target_i += 1
            if self.target_i >= len(self.target_x): 
                self.target_i = 0
            i = self.target_i
            nv = Vector(self.target_x[self.target_i], self.target_y[self.target_i])
            self.target.velocity = (self.target.position - nv) / self.dt
            self.target.position = nv 

        # Update the missile 
        self.missile.update(self.dt, self.target)

        # Check for finishing clause 
        if Vector.distance(self.missile.position, self.target.position) < self.target_size: 
            reward = 10; 
            terminated = True; 
        else: 
            reward = 0; 

        # Update missile model 
        if self.training: 
            self.missile.backprop(reward)

        # check for truncation 
        self.frames += 1; 
        if self.frames > self.training_length_frames: 
            truncated = True

        # Manage pygame 
        if self.render_mode == "human": 
            self.render()

        # return the goods 
        return self.get_obs(), reward, terminated, truncated, self.get_info()

    def render(self): 
        if not self.rendered and pg: 
            # Initialize pygame 
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            self.running = True  # Unsure if we really need it 


            # Report that we are initialized 
            self.rendered = True 
        
        if self.running and pg: 
            # Set up the canvas (for rgb rendering as well)
            canvas = pygame.Surface(self.window_size)
            canvas.fill((255, 255, 255)) 

            # Draw the target 
            # p = Tools.get_cam_pos(self.window_size, self.target.position)
            # p = self.cam_pos(self.target.position)
            pygame.draw.circle(canvas, self.target_color, self.cam_pos(self.target.position), self.target_size)

            # Draw the missile 
            start = self.missile.position
            end = start - (self.missile.velocity.unit() * self.missile_size)
            pygame.draw.line(canvas, self.missile_color, self.cam_pos(start), self.cam_pos(end), self.missile_width)

            # Draw status bars 
            obs = self.get_obs()
            xcent = self.window_size[0]/2
            xrange = 50
            lw = 3
            pygame.draw.line(canvas, 'green', (xcent, 5), (xcent + xrange*(obs['v_in']), 5), lw)
            pygame.draw.line(canvas, 'red', (xcent, 10), (xcent + xrange*obs['v_out'], 10), lw)
            pygame.draw.line(canvas, 'blue', (xcent, 15), (xcent + xrange*obs['d_in'], 15), lw)
            pygame.draw.line(canvas, 'purple', (xcent, 20), (xcent + xrange*obs['d_out'], 20), lw)
            pygame.draw.line(canvas, 'orange', (xcent, 25), (xcent + xrange*obs['d_mag'], 25), lw)

            if self.render_mode == "human": 
                self.window.blit(canvas, canvas.get_rect())  # draw the canvas to the window 
                pygame.event.pump()  # Manage the event queue(?) might help with not handling inputs 
                pygame.display.update()  # Update the window 
                self.clock.tick(1/self.dt)
    
    def close(self): 
        if self.rendered and pg: 
            pygame.quit(); 

    def get_obs(self): 
        # things that we want to directly track (optional)
        obs = self.missile.get_obs(self.target, self.dt)

        return {
            "d_in": obs[0], 
            'd_out': obs[1], 
            'v_in'  : obs[2], 
            'v_out' : obs[3], 
            'd_mag' : obs[4]
        }

    def get_info(self): 
        # other interesting stats (optional)
        return {}
    
    # Helpers 
    def cam_pos(self, p): 
        if self.follow_missile: 
            return Tools.get_cam_pos(self.window_size, p-self.missile.position).as_tuple()
        else: 
            return Tools.get_cam_pos(self.window_size, p).as_tuple()

@dataclass 
class Target: 
    position: Vector = Vector()
    velocity: Vector = Vector()

# End of class 

