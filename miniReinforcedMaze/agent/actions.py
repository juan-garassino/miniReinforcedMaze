def choose_action(Q_table, posible_actions, state):
    taken_action = None
    current_value = None
    for action in posible_actions:
        possible_value = Q_table[(action, state)]
        if taken_action is None or possible_value >= current_value:
            current_value = possible_value
            taken_action = action
    return taken_action, current_value

def vision_from(maze, position, max_distance=1):
    vision = []
    for distance in range(1, max_distance + 1):
        vision += [
            maze[position[0],
                 position[1] - distance] if position[1] - distance >= 0 else 1,
            maze[position[0], position[1] +
                 distance] if position[1] + distance < maze.shape[1] else 1,
            maze[position[0] - distance,
                 position[1]] if position[0] - distance >= 0 else 1,
            maze[position[0] + distance,
                 position[1]] if position[0] + distance < maze.shape[0] else 1
        ]
    return tuple(vision)

def apply_action(position, maze, actions, action):
    print(action)
    if action == actions['UP']:
        return (position[0], max(position[1] - 1, 0))
    elif action == actions['DOWN']:
        return (position[0], min(position[1] + 1, maze.shape[0] - 1))
    elif action == actions['LEFT']:
        return (max(position[0] - 1, 0), position[1])
    elif action == actions['RIGHT']:
        return (min(position[0] + 1, maze.shape[1] - 1), position[1])
    raise Exception("Action %s not understood." % action)
