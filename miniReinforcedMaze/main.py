from miniReinforcedMaze.preprocessing.cropping import crop
from miniReinforcedMaze.resources.generate_maze import generate_maze
from miniReinforcedMaze.agent.players import RandomPlayer, QPlayer, train_q_learning
from miniReinforcedMaze.play import play
from collections import defaultdict


path = "/Users/juan-garassino/Code/juan-garassino/miniReinforcedMaze/miniReinforcedMaze/data"

image_names = ["cave.png", "objects.png", "agus.png"]

imagen_indexes = [16 * 8 + 2, 16 * 9, 16 * 20, 378, -1]

cropped_images, returned_indexes = crop(path,
                                        image_names,
                                        height=16,
                                        width=16,
                                        sprite_size=32,
                                        show=False,
                                        return_indexes=imagen_indexes)

MAZE_IMAGES = cropped_images

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

# Numeric value assign to the objects to use on the maze
codes = {"TREASURE": 1, "LAND": 0, "HOLE": -1}

maze = generate_maze(16, 16, density=0.2)

random_player = RandomPlayer()  # Creates the player

#print(maze)

observations, result = play(maze,
                            random_player,
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

# play(maze, random_player, show_gif=True)  # with the respective maze and the respective player

print(result)

q_player = QPlayer(defaultdict(lambda: 0), explore_factor=1.0)

q_player = train_q_learning(q_player, maze, cropped_images, amount_games=10000)

q_player = QPlayer(q_player.Q, explore_factor=0.00)

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

print(result)
