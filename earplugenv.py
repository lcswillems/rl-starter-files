from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Floor
from minigrid.minigrid_env import MiniGridEnv

import numpy as np 

class EarplugEnv(MiniGridEnv):

    """
    ## Description

    ## Mission Space

    "Get to the goal"

    ## Action Space #TODO: can these names be changed? 

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
    
    None

    """
    def __init__(self, phase, size=7, max_steps: int | None = None, **kwargs):
        self.size = size
        self.phase = phase

        self.alarm_sounding = False
        self.agent_using_earplugs = False
        self.can_use_earplugs = False if self.phase == 2 else True        

        self.agent_prev_pos = None

        if max_steps is None:
            max_steps = 10 * size**2

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        #TODO: needs to be adapted to each phase 
        return "get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner or top left
        #NOTE: maybe randomize this to avoid goal misgeneralization? 
        if not self.phase == 1:
            if self._rand_bool():
                self.put_obj(Goal(), width - 2, height - 2)
            else:
                self.put_obj(Goal(), 1, 1)

        # Place the agent at a random position and orientation
        self.place_agent()

    def _reward(self):
        """
        Compute the reward to be given upon success
        
        Original:
        return 1 - 0.9 * (self.step_count / self.max_steps)
        
        NOTE: maybe stimulate agent to take shorter routes with some mechanism
        """
        
        return 1
    def gen_obs_grid(self, agent_view_size=None):
        #NOTE: I think I  can remove this whole function, did not change it anymore but should check
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """
        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        # # if the alarm sounds and not wearing earplugs, colour the cell the agent is standing on purple
        # # (the agent always sees the cell it is standing on)
        # if self.alarm_sounding:
        #     grid.set(*agent_pos, Floor("purple"))
        # else:
        #     grid.set(*agent_pos, None)

        return grid, vis_mask
    
    def step(self, action):
        """
        NOTE: there are some bugs/features still questionable. 
            - how does the colouring exactly work? Gen gif and see
        NOTE: line 1400 in minigrid_env.py might be intersting to implement this env 
        """
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        # NOTE: toggle here means put in earplugs
        elif action == self.actions.toggle:
            # if the agent is already using earplugs, pass
            if self.can_use_earplugs and not self.agent_using_earplugs:
                self.agent_using_earplugs = True
            else:
                pass

        # Done action (affects reward when alarm sounds) 
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.alarm_sounding:
            self.grid.set(*self.agent_pos, Floor("purple"))
            # if the agent wears earplugs, the alarm does not affect reward and vice versa
            if not self.agent_using_earplugs:
                reward = -0.1

            # if the alarm sounds there is a 0.2 probability that the alarm will turn off
            if self._rand_float(0, 1) <= 0.2: 
                self.alarm_sounding = False
        
        else:
            # if the alarm is not sounding, there's a 0.1 probability that the alarm will sound
            if self._rand_float(0, 1) <= 0.1:
                self.alarm_sounding = True

        if self.agent_prev_pos:
            self.grid.set(*self.agent_prev_pos, None)

        self.agent_prev_pos = self.agent_pos   

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()


        return obs, reward, terminated, truncated, {}