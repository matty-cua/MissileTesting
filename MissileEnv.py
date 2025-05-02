import numpy as np 

import gymnasium as gym
import pygame
from pygame.locals import * 

import random 
from dataclasses import dataclass 

from Tools import * 
from Missile import Missile
from PathGen import PathGenerator

pg = True 
gm = True

class MissileEnv(gym.Env): 
# class MissileEnv: 

    def __init__(self): 
        # Important behavior vars 
        self.move_target = True
        self.GP = PathGenerator()
        self.training_length_frames = 5*30

        # Gym built in vars 
        self.action_size = 5
        self.action_space = gym.spaces.Discrete(self.action_size)

        # Gen variables 
        self.dt = 1/30;  # Frame rate 
        self.bound_size = 200; 
        self.training = True; 
        self.frames = 0; 

        # Target variables 
        self.target = Target() 
        self.target_speed = 150

        # Missile variables 
        self.missile = Missile()
        self.action = None

        # Manage pygame 
        self.follow_missile = True  # Center rocket on the frame (instead of (0, 0))
        self.rendered = False  # if we have initiaized the pygame window 
        self.running_pg = False  # Unsure if this is necessary either, most likely pauses things? 
        self.window_size = (400, 400) 
        self.render_mode = None 

        # Rendering variables 
        self.target_color = (255, 100, 100)  # A nice light pink? 
        self.missile_color = (50, 50, 50)  # Dark gray 
        self.missile_width = 4
        self.missile_size = 20
        self.target_size = 10 

        # input management 
        self.quit = False 
        self.running = False 
        self.key_names = {  
            'up': K_UP, 
            'down': K_DOWN, 
            'right': K_RIGHT, 
            'left': K_LEFT, 
            'reset': K_r, 
        }
        self.key_mapping = {v: k for k, v in self.key_names.items()}
        self.keys = list(self.key_names.values())
        self.player_inputs = {v: False for v in self.key_names.values()}

    def reset(self, seed=None, options=None): 
        # super stuff 
        try: 
            super().reset(seed=seed)
        except Exception as e: 
            pass 

        # admin stuff 
        self.frames = 0

        # Random target position, zero velocity
        if not self.move_target or self.render_mode == 'player':  
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

        # Reset again if missile starts too close to target (recursively) 
        if Vector.distance(self.missile.position, self.target.position) < 4*self.target_size:
            return self.reset(seed, options)
        
        # Render if in player mode (need to initialize pygame before getting events) 
        if self.render_mode == 'player': 
            self.action = 0  # need this to render the model input bar at the bottom 
            self.render()

        # Return observation and info 
        self.running = True 
        return self.get_obs(), self.get_info()

    def step(self, action=None): 
        # Basic inits 
        terminated = False 
        truncated = False 

        # update target kinematics 
        if self.render_mode == 'player': 
            # Grab inputs 
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: 
                    pygame.quit()
                    raise SystemExit("Goodbye!")
                if event.type == pygame.KEYDOWN: 
                    if event.key in self.keys: 
                        self.player_inputs[event.key] = True 
                if event.type == pygame.KEYUP: 
                    if event.key in self.keys: 
                        self.player_inputs[event.key] = False 

            # Update target kinematics 
            if self.running: 
                move_vector = Vector(0, 0)
                if self.player_inputs[self.key_names['right']]: 
                    move_vector += Vector(1, 0) 
                if self.player_inputs[self.key_names['left']]: 
                    move_vector += Vector(-1, 0)
                if self.player_inputs[self.key_names['up']]: 
                    move_vector += Vector(0, -1)
                if self.player_inputs[self.key_names['down']]: 
                    move_vector += Vector(0, 1)
                self.target.position += move_vector.unit() * self.dt * self.target_speed 

            # Manage reset 
            if self.player_inputs[self.key_names['reset']]: 
                # return obersvations, no reward, not terminated, not truncated, info (empty dict)
                self.reset()
                return self.get_obs(), 0, False, False, self.get_info()


        elif self.move_target: 
            self.target_i += 1
            if self.target_i >= len(self.target_x): 
                self.target_i = 0
            i = self.target_i
            nv = Vector(self.target_x[self.target_i], self.target_y[self.target_i])
            self.target.velocity = (self.target.position - nv) / self.dt
            self.target.position = nv 

        # Update the missile 
        if self.running: 
            da = (self.action_size-1)/2
            act_in = (action-da)/da 
            self.missile.update(self.dt, self.target, action=act_in)
            self.action = act_in

        # get observations 
        obs = self.get_obs()

        # Check for finishing clause and calc reward 
        if Vector.distance(self.missile.position, self.target.position) < 2.5*self.target_size: 
            reward = 1; 
            terminated = True; 
            self.running = False 
        else: 
            # reward = .05 * (obs[2] - (.5*np.abs(obs[3])))
            reward = (-.2/self.training_length_frames) *  (obs[-1] + np.abs(obs[1]))
            # reward = -1 

        # check for truncation 
        if not self.render_mode == 'player': 
            self.frames += 1; 
            if self.frames > self.training_length_frames: 
                truncated = True
                self.running = False 

        # Manage pygame 
        if self.render_mode in ["human", 'player']: 
            self.render()
            # print(f"env action: {self.action}")

        # return the goods 
        return obs, reward, terminated, truncated, self.get_info()

    def render(self): 
        if not self.rendered and pg: 
            # Initialize pygame 
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            self.running_pg = True  # Unsure if we really need it 


            # Report that we are initialized 
            self.rendered = True 
        
        if self.running_pg and pg: 
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
            obs = self.get_obs(as_dict=True)
            xcent = self.window_size[0]/2
            yws = self.window_size[1]
            xrange = 50
            lw = 3
            pygame.draw.line(canvas, 'green', (xcent, 5), (xcent + xrange*(obs['v_in']), 5), lw)
            pygame.draw.line(canvas, 'red', (xcent, 10), (xcent + xrange*obs['v_out'], 10), lw)
            pygame.draw.line(canvas, 'blue', (xcent, 15), (xcent + xrange*obs['d_in'], 15), lw)
            pygame.draw.line(canvas, 'purple', (xcent, 20), (xcent + xrange*obs['d_out'], 20), lw)
            pygame.draw.line(canvas, 'orange', (xcent, 25), (xcent + xrange*obs['d_mag'], 25), lw)

            # Draw rocket turning GUI 
            pygame.draw.line(canvas, 'black', (xcent-xrange, yws-5), (xcent-xrange, yws-15), lw)
            pygame.draw.line(canvas, 'black', (xcent+xrange, yws-5), (xcent+xrange, yws-15), lw)
            pygame.draw.line(canvas, 'black', (xcent, yws-5), (xcent, yws-15), lw)
            pygame.draw.line(canvas, 'red', (xcent, self.window_size[1]-10), (xcent + xrange*self.action, self.window_size[1]-10), lw)

            if self.render_mode in ["human", 'player']: 
                self.window.blit(canvas, canvas.get_rect())  # draw the canvas to the window 
                pygame.event.pump()  # Manage the event queue(?) might help with not handling inputs 
                pygame.display.update()  # Update the window 
                self.clock.tick(1/self.dt)
    
    def close(self): 
        if self.rendered and pg: 
            pygame.quit(); 

    def get_obs(self, as_dict=False): 
        # things that we want to directly track (optional)
        obs = self.missile.get_obs(self.target, self.dt)

        if as_dict: 
            return {
                "d_in": obs[0], 
                'd_out': obs[1], 
                'v_in'  : obs[2], 
                'v_out' : obs[3], 
                'd_mag' : obs[4]
            }
        else: 
            return obs

    def get_info(self): 
        # other interesting stats (optional)
        return {}
    
    # Helpers 
    def cam_pos(self, p): 
        if self.follow_missile: 
            return Tools.get_cam_pos(self.window_size, p-self.missile.position).as_tuple()
        else: 
            return Tools.get_cam_pos(self.window_size, p).as_tuple()
        
    def add_frame(self): 
        if self.save_gif: 
            ... 

@dataclass 
class Target: 
    position: Vector = Vector()
    velocity: Vector = Vector()

# End of class 

