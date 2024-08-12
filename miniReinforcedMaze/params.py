image_names = ["cave.png", "objects.png", "agus.png"]

imagen_indexes = [16 * 8 + 2, 16 * 9, 16 * 20, 378, -1]

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

CONVERT_DIRECTIONS = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

INITIAL_POSITION = (5, 5)

POSSIBLE_ACTIONS = [UP, DOWN, LEFT, RIGHT]

POSSIBLE_ACTIONS = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}

# Numeric value assign to the objects to use on the maze
codes = {"TREASURE": 1, "LAND": 0, "HOLE": -1}
