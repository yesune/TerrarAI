import gym
from gym import spaces
import numpy as np
from websocket_client import TerrariaWebSocketClient

# Handles 10 enemies, 1 boss, 1 player, in standard overworld arena.
class TerrariaOverworldEnv(gym.Env):
    
    num_enemies = 10

    min_x = 52704
    max_x = 88044
    min_y = 656
    max_y = 5254
    max_vx = 20
    max_vy = 20

    def __init__(self):
        super().__init__()

        # initialize websocket client
        self.ws_client = TerrariaWebSocketClient()
        self.ws_client.start()

        # possible actions
        # Up, Down, Left, Right
        # Jump, Grapple, Mount
        # Attack, Heal
        self.single_action_space = spaces.Discrete(9)  # Example: 
        # Define the observation space: player/boss/enemy position + velocity + relative hp
        self.single_observation_space = spaces.Box(low=np.array([0, 0, -1, -1, 0] * 2 + [0, 0, 0, -1, -1, 0] * self.num_enemies),
                                            high=np.array([1, 1, 1, 1, 1]  * 2 + [1, 1, 1, 1, 1, 1] * self.num_enemies),
                                            dtype=np.float32)

    def reset(self):
        # 1. move player back to arena
        # 2. buff
        # 3. spawn
        return self.get_data()
        
    # output
    # next_obs: next observation
    # reward: the reward for action
    # terminations: boolean for whether environment ended normally (death or kill)
    # truncation: boolean for whether environment was forcibly stopped (time limit)
    # infos: idk
    def step(self, action):
        next_state = self.get_data()
        reward = 1
        done = False
        return next_state, reward, done, {}

    # handled by terraria game
    def render(self):
        pass  

    # handled manually lol
    def close(self):
        pass

    # normalizes player_data
    def get_data(self):
        game_data = self.ws_client.player_data
        normalized_data = []
        player = game_data['player']
        normalized_data.extend(
            self.normalize_position(player['x'], player['y']),
            self.normalize_velocity(player['vx'], player['vy']),
            player['hp']
        )
        # Normalize boss data
        for boss in game_data["bosses"]:
            normalized_data.extend([
                self.normalize_position(boss['x'], boss['y']),
                self.normalize_velocity(boss['vx'], boss['vy']),
                boss['hp']
            ])
        # Normalize enemy data
        for enemy in game_data["enemies"][:10]:
            normalized_data.extend([
                self.normalize_position(enemy['x'], enemy['y']),
                self.normalize_velocity(enemy['vx'], enemy['vy']),
                enemy['hp']
            ])
        return np.array(normalized_data)

    def normalize_position(self, x, y):
        return float(x-self.min_x)/(self.max_x-self.min_x), float(y-self.min_y)/(self.max_y-self.min_y)

    def normalize_velocity(self, vx, vy):
        return float(vx)/self.max_vx, float(vy)/self.max_vy
