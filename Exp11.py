import random
class GridWorld:
  """
  A simple grid-world environment as a Markov Decision Process (MDP).
  """

  def __init__(self, grid_size, start, goal, walls=[]):
    """
    Initialize the grid world environment.

    Args:
      grid_size: Tuple (width, height) of the grid world.
      start: Tuple (x, y) coordinates of the starting state.
      goal: Tuple (x, y) coordinates of the goal state.
      walls: List of tuples (x, y) representing wall locations.
    """
    self.grid_size = grid_size
    self.start = start
    self.goal = goal
    self.walls = walls
    self.state = self.start  # Current state
    self.actions = ['U', 'D', 'L', 'R']  # Up, Down, Left, Right

  def is_valid_action(self, action):
    """
    Checks if the given action is valid within the grid world boundaries and doesn't hit a wall.
    """
    x, y = self.state
    new_x = x + {'U': -1, 'D': 1, 'L': -1, 'R': 1}.get(action, 0)  # Default value 0 for invalid actions
    new_y = x + {'U': 0, 'D': 0, 'L': -1, 'R': 1}.get(action, 0)

    return (0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1] and
            (new_x, new_y) not in self.walls)

  def take_action(self, action):
    """
    Executes the given action and returns the next state and reward.

    Args:
      action: String representing the action (U, D, L, R).

    Returns:
      Tuple (next_state, reward).
    """
    if not self.is_valid_action(action):
      return self.state, 0  # Stay put, no reward for invalid action

    x, y = self.state
    new_x = x + {'U': -1, 'D': 1, 'L': -1, 'R': 1}.get(action, 0)  # Default value 0 for invalid actions
    new_y = x + {'U': 0, 'D': 0, 'L': -1, 'R': 1}.get(action, 0)

    self.state = (new_x, new_y)

    reward = 0
    if self.state == self.goal:
      reward = 1  # Reward for reaching the goal

    return self.state, reward

# Example usage
grid_world = GridWorld(grid_size=(5, 5), start=(0, 0), goal=(4, 4), walls=[(2, 2)])

# Interact with the environment (example)
state = grid_world.state
while state != grid_world.goal:
  # Choose an action based on your policy (e.g., random exploration here)
  action = random.choice(grid_world.actions)
  next_state, reward = grid_world.take_action(action)
  state = next_state
  # Update your agent's knowledge based on the experience (s, a, r, s')

print("Reached goal!")
