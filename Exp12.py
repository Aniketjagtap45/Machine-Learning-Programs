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
    new_x, new_y = x + {'U': -1, 'D': 1, 'L': -1, 'R': 1}[action]
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
    new_x, new_y = x + {'U': -1, 'D': 1, 'L': -1, 'R': 1}[action]
    self.state = (new_x, new_y)

    reward = 0
    if self.state == self.goal:
      reward = 1  # Reward for reaching the goal

    return self.state, reward

def q_learning(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, episodes=1000):
  """
  Implements Q-learning algorithm for the given grid world environment.

  Args:
    env: GridWorld environment object.
    learning_rate: Learning rate for updating Q-values (between 0 and 1).
    discount_factor: Discount factor for future rewards (between 0 and 1).
    epsilon: Exploration rate (probability of random action).
    episodes: Number of training episodes.

  Returns:
    Q-value table after training.
  """
  Q = {}  # Q-value table (state, action) -> Q-value
  for state in range(env.grid_size[0] * env.grid_size[1]):
    for action in env.actions:
      Q[(state, action)] = 0  # Initialize all Q-values to 0

  for _ in range(episodes):
    state = env.reset()  # Reset environment to starting state
    while True:
      # Epsilon-greedy action selection (corrected calculation of max_q)
      if random.uniform(0, 1) < epsilon:
        action = random.choice(env.actions)  # Explore randomly
      else:
        max_q_value = float('-inf')
        for a in env.actions:
          q_value = Q.get((state, a), 0)  # Handle unseen state-action pairs
          max_q_value = max(max_q_value, q_value)  # Corrected calculation
        # Choose the action with the highest Q-value (break ties randomly)
        best_actions = [a for a in env.actions if Q[(state, a)] == max_q_value]
        action = random.choice(best_actions)

      next_state, reward = env.take_action(action)

      # Update Q-value using
