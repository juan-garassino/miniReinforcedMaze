import numpy as np
from miniReinforcedMaze.agent.actions import choose_action
from miniReinforcedMaze.resources.generate_maze import generate_maze
from miniReinforcedMaze.play import play

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

POSSIBLE_ACTIONS = [UP, DOWN, LEFT, RIGHT]

class Player(object):

    def __init__(self):
        self.i = 0

    def move(self, state):
        raise NotImplementedError("Subclass responsibility")

class RandomPlayer(Player):  # Calls the Class Player

    def move(self, state, posible_actions):
        return np.random.choice(posible_actions)

class QPlayer(Player):  # Calls the Class Player

    def __init__(self, Q, explore_factor):
        self.Q = Q
        self.explore_factor = explore_factor
        super(QPlayer, self).__init__()  # Super ?

    def move(self, state, posible_actions):
        if np.random.uniform(0, 1) > self.explore_factor:
            self.i += 1
            taken_action, current_value = choose_action(self.Q,
                                                        state)  # state ?
            return taken_action
        return RandomPlayer().move(state, posible_actions)


def q_learning(Q, observations, learning_rate=0.2, discount_factor=0.95):

    posible_actions=[]

    print(observations)

    for observation in observations:
        old_state, action, reward, new_state = observation

        _, estimated_future_value = choose_action(Q, posible_actions, new_state)
        current_value = Q[(action, old_state)]

        print(estimated_future_value)
        print()
        Q[(action, old_state)] = current_value + learning_rate * (
            reward + discount_factor * estimated_future_value - current_value)
    return Q



# Numeric value assign to the different kind of pictures
IMAGEN_POZO = 0
IMAGEN_TIERRA = 1
IMAGEN_LAVA = 2
IMAGEN_SALIDA = 3
IMAGEN_TIPO = 4

REWARD_WIN = 1000
REWARD_LOSE = -5000
REWARD_STEP = -10
REWARD_REPEAT = -500

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

CONVERT_DIRECTIONS = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}

INITIAL_POSITION = (1, 1)

POSSIBLE_ACTIONS = [UP, DOWN, LEFT, RIGHT]

POSSIBLE_ACTIONS = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
codes = {"TREASURE": 1, "LAND": 0, "HOLE": -1}


def train_q_learning(q_player, maze, cropped_images, amount_games):
    for game_number in range(amount_games):
        maze = generate_maze(maze.shape[0], maze.shape[1])

        observations, result = play(maze,
                            q_player,
                            INITIAL_POSITION,
                            POSSIBLE_ACTIONS,
                            CONVERT_DIRECTIONS,
                            REWARD_STEP,
                            REWARD_REPEAT,
                            REWARD_LOSE,
                            REWARD_WIN,
                            codes,
                            cropped_images,
                            verbose=True,
                            show_gif=True)

        old_Q = q_player.Q.copy()
        new_Q = q_learning(q_player.Q, [], observations)

        # Convergence:
        if game_number % 5000 == 0:
            convergence = sum(
                [abs(old_Q[key] - new_Q[key]) for key in old_Q.keys()])
            print('Game: %s | Convergence: %.2f' % (game_number, convergence))
            # play(maze, q_player)

        q_player = QPlayer(new_Q, explore_factor=0.05)
    return q_player
