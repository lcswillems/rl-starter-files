from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

class PhasicDoorKeyEnv(MiniGridEnv):

    """
    ## Description

    This environment has a key that the agent must pick up in order to unlock a
    goal and then get to the green goal square. This environment is difficult,
    because of the sparse reward, to solve using classical RL algorithms. It is
    useful to experiment with curiosity or curriculum learning.

    ## Mission Space

    "use the key to open the door and then get to the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-PhasicDoorKey-5x5-v0`
    - `MiniGrid-PhasicDoorKey-6x6-v0`
    - `MiniGrid-PhasicDoorKey-8x8-v0`
    - `MiniGrid-PhasicDoorKey-16x16-v0`

    """
    def __init__(self, phase, size=7, max_steps: int | None = None, **kwargs):
        self.size = size
        self.phase = phase
        # if phase == 1:
        #     self.has_goal = 

        if max_steps is None:
            max_steps = 10 * size**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        #TODO: needs to be adapted to each phase 
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # select if it has a goal or not (alternatively it has a key)
        has_goal = False if self.phase == 1 else True

        # Place a goal in the bottom-right corner
        #NOTE: maybe randomize this to avoid goal misgeneralization? 
        if has_goal:
            self.put_obj(Goal(), width - 2, height - 2)

        # NOTE: still decide whether we want to have a short cut through a hole in the wall. 
        # wall opening idx
        # wallOpeningIdx = self._rand_int(1, height - 2)

        # Create a vertical splitting wall
        self.splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(self.splitIdx, 0, length=self.height)
        # self.grid.vert_wall(self.splitIdx, 0, length=wallOpeningIdx)
        # self.grid.vert_wall(self.splitIdx, wallOpeningIdx + 1, length=height - wallOpeningIdx - 1)


        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(self.splitIdx, height))

        # Place a door in the wall
        self.doorIdx = self._rand_int(1, width - 2)
        self.put_obj(Door("yellow", is_locked=False), self.splitIdx, self.doorIdx)
        self.door_is_locked = False
        self.door_pos = (self.splitIdx, self.doorIdx)

        # Place a yellow key on the left side
        if self.phase != 2:
            self.place_obj(obj=Key("yellow"), top=(0, 0), size=(self.splitIdx, height))

        self.mission = "Get to the goal, maybe you have to use a key to open a door"

    def step(self, action):
        """
        Changes the reward based on the state. 

        NOTE: there are some bugs/features still questionable. 
            See notes in this method 
        """
        # first execute step as 
        obs, reward, terminated, truncated, info = super().step(action)

        if self.door_is_locked: 
            # if the door is locked there is a 0.05 probability that the door will unlock
            if self._rand_float(0, 1) <= 0.05: 
                self.put_obj(Door("yellow", is_locked=False), self.splitIdx, self.doorIdx)
                self.door_is_locked = False
        
        else:
            # if the door is unlocked, there's a 0.1 probability that the door will lock
            if self._rand_float(0, 1) <= 0.1:
                self.put_obj(Door("yellow", is_locked=True), self.splitIdx, self.doorIdx)
                self.door_is_locked = True
        
        if self.door_is_locked: # maybe also normal reward when the key is picked up? 
            reward = 0 if action == self.actions.done else -0.01

        return obs, reward, terminated, truncated, info

    def gen_obs(self):
        
        return super().gen_obs()