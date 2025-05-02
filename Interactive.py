# Basic imports 
from pathlib import Path 

# RL imports 
import torch 
import pygame 
from pygame.locals import * 

# Custom imports
from MissileEnv import MissileEnv 
from Missile import Missile 
from RLManager import DQN, DQN_RNN
from Tools import * 

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class InteractiveEnv(MissileEnv):

    def __init__(self): 
        super().__init__()
        self.render_mode = 'player'
        self.move_target = False 
        self.num_missiles = 1 
        self.missiles = [] 
        self.missile_delays = []
        self.destroy_bounds = 1000  # Distance from center to destroy and reset missile 
        self.bound_size =300
        self.target_speed = 250

        # Gamey stuff 
        self.lives = 3 
        self.add_time = 3
        self.terminate = False 
        self.play_time = 0 
        self.curr_time = 0
        self.font = pygame.font.SysFont('Freesandsbold.tff', 32)
        Debug.enable_log = False  # hide debugging output from console 
        
         # Define the model path 
        save_loc = Path('GameModel')
        use_checkpoint = False
        # save_loc = Path("Archive") / "Missile_08_1600e_dist_loss"

        # input management 
        self.quit = False 
        self.running = False 
        self.key_names = {
            'up': K_UP, 
            'down': K_DOWN, 
            'right': K_RIGHT, 
            'left': K_LEFT, 
            'reset': K_r, 
            'add_missile': K_w, 
            'kill_missile': K_s, 
        }
        self.key_mapping = {v: k for k, v in self.key_names.items()}
        self.keys = list(self.key_names.values())
        self.player_inputs = {v: False for v in self.key_names.values()}

        # Load the model 
        self.model = torch.load(save_loc / 'ModelTorch.pkl', weights_only=False, map_location=torch.device(device))  # Needs access to DQN class \

        # Try reloading the best checkpoint 
        if use_checkpoint: 
            try: 
                weights = torch.load(save_loc / 'CheckpointWeights.wts', weights_only=True)
                self.model.load_state_dict(weights)
                print("Successfully loaded checkpoint.") 
            except Exception as e: 
                print("Loading weights failed: ") 
                print(e)

    def reset(self): 

        # Gamey reset 
        self.play_time = 0 
        self.num_missiles = 1
        self.lives = 3
        self.terminate = False 

        # Target reset 
        self.target.velocity = Vector(0, 0)
        self.target.position = Vector(0, 0) 

        # Missile Reset
        self.missiles = [self.create_missile() for _ in range(self.num_missiles)]
        self.missile_delays = [0 for _ in range(self.num_missiles)]

        self.render()

        self.running = True 

    def step(self): 

        # manage player inputs 
        if self.render_mode == 'player': 
            # Grab inputs 
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: 
                    pygame.quit()
                    raise SystemExit("Goodbye!")
                if event.type == pygame.KEYDOWN: 
                    if event.key == K_ESCAPE: 
                        pygame.quit()
                        raise SystemExit("Adios")
                    if event.key in self.keys: 
                        self.player_inputs[event.key] = True 
                if event.type == pygame.KEYUP: 
                    if event.key in self.keys: 
                        self.player_inputs[event.key] = False

            # Manage reset 
            if self.player_inputs[self.key_names['reset']]: 
                self.player_inputs[self.key_names['reset']] = False 
                self.reset()

            # Update target kinematics 
            if not self.terminate: 
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
                
                # Manage add/subtract missiles
                if self.player_inputs[self.key_names['add_missile']]: 
                    Debug.log("Adding missile")
                    self.num_missiles += 1
                    self.player_inputs[self.key_names['add_missile']] = False 
                if self.player_inputs[self.key_names['kill_missile']]: 
                    if self.num_missiles > 1: 
                        Debug.log("Removing missile")
                        self.num_missiles -= 1
                        self.player_inputs[self.key_names['kill_missile']] = False
            
            
            # Update the missiles
            if self.running: 
                # Check for too few missiles 
                if self.num_missiles > len(self.missiles): 
                    self.missiles.append(self.create_missile())
                if self.num_missiles < len(self.missiles): 
                    d_list = [Vector.distance(self.target.position, m.position) for m in self.missiles]
                    I = [i for i, x in sorted(enumerate(d_list))][0]
                    self.missiles.pop(I)

                for i, m in enumerate(self.missiles): 
                    state = torch.tensor(m.get_obs(self.target, self.dt), dtype=torch.float32, device=device).unsqueeze(0)
                    action = self.model(state).max(1).indices.view(1, 1)
                    da = (self.action_size-1)/2
                    act_in = (action-da)/da 
                    m.update(self.dt, self.target, action=act_in)
                    
                    # Check if out of bounds 
                    if m.position.magnitude() > self.destroy_bounds: 
                        self.missiles[i] = self.create_missile()

                    # Check for collision with target 
                    if Vector.distance(m.position, self.target.position) < 2.5*self.target_size: 
                        self.lives -= 1
                        self.num_missiles -= 1
                        self.missiles.pop(i)
                        # self.missiles[i] = self.create_missile()


            # Manage gamey side 
            if self.running: 
                self.play_time += self.dt 
                self.curr_time += self.dt 
                if self.curr_time >= self.add_time: 
                    self.num_missiles += 1
                    self.curr_time = 0
                if self.lives <= 0: 
                    self.running = False 
                    self.terminate = True 
            
        # rendering 
        self.render()

    def render(self): 
        if not self.rendered: 
            # # Initialize pygame 
            # pygame.init()
            # self.window = pygame.display.set_mode(self.window_size)
            # self.clock = pygame.time.Clock()
            # self.running_pg = True  # Unsure if we really need it 


            # Report that we are initialized 
            self.rendered = True 
        
        if self.running_pg: 
            # Set up the canvas (for rgb rendering as well)
            canvas = pygame.Surface(self.window_size)
            canvas.fill((255, 255, 255)) 

            # Draw the target 
            pygame.draw.circle(canvas, self.target_color, self.cam_pos(self.target.position), self.target_size)

            # Draw the missiles 
            for m in self.missiles: 
                start = m.position
                end = start - (m.velocity.unit() * self.missile_size)
                pygame.draw.line(canvas, self.missile_color, self.cam_pos(start), self.cam_pos(end), self.missile_width)

            # Draw the lives 
            for i in range(self.lives): 
                pygame.draw.circle(canvas, 'Red', (20 * (1+i), 20), 10)

            # Display num missiles 
            txt = self.font.render(str(int(self.num_missiles)), False, 'black')
            canvas.blit(txt, (self.window_size[1]-50, 10))

            # Display time 
            txt = self.font.render(f"{self.play_time:.2f}", False, 'black')
            canvas.blit(txt, (self.window_size[1]/2, 10))

            # Render game over 
            if self.terminate: 
                txt = self.font.render("GAME OVER", False, 'red')
                canvas.blit(txt, (10, 10))

            # Render the screen 
            self.window.blit(canvas, canvas.get_rect())  # draw the canvas to the window 
            pygame.event.pump()  # Manage the event queue(?) might help with not handling inputs 
            pygame.display.update()  # Update the window 
            self.clock.tick(1/self.dt)

    def create_missile(self): 
        m = Missile() 
        m.velocity = Tools.random_unit() * m.speed 
        m.position = self.target.position + (Vector.from_angle(random.random() * 2*math.pi) * self.bound_size)
        return m 
    
    def game_over(self): 
        ... 


# Initialize pygame 
pygame.init()

# Set up the environment 
env = InteractiveEnv()
env.render_mode = 'player'
env.follow_missile = False 
env.window = pygame.display.set_mode(env.window_size)
env.clock = pygame.time.Clock()
env.running_pg = True  # Unsure if we really need it 

# Load the model 
save_loc = Path('Output')
# save_loc = Path("Archive") / "Missile_08_1600e_dist_loss"

model = torch.load(save_loc / 'ModelTorch.pkl', weights_only=False)  # Needs access to DQN class \

# Try reloading the best checkpoint 
use_checkpoint = True
if use_checkpoint: 
    try: 
        weights = torch.load(save_loc / 'CheckpointWeights.wts', weights_only=True)
        model.load_state_dict(weights)
        print("Successfully loaded checkpoint.") 
    except Exception as e: 
        print("Loading weights failed: ") 
        print(e)

# Run the environment 
env.reset() 
while True: 
    env.step()