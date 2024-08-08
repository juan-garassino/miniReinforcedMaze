from miniReinforcedMaze.agent.actions import vision_from, choose_action, apply_action
from miniReinforcedMaze.resources.generate_animation import GifMaze, show_maze, GifMazeRewards
from collections import defaultdict


def play(maze,
         player,
         initial_position,
         posible_actions,
         converted_directions,
         reward_step,
         reward_repeat,
         reward_lose,
         reward_win,
         codes,
         cropped_images,
         verbose=False,
         show_gif=False):
    position = initial_position
    observations = []
    reward = 0.0
    result = None
    memory = defaultdict(lambda: 0)
    memory[initial_position] = 1

    if verbose:
        show_maze(maze, position)

    if show_gif:
        gif_maze = GifMaze()
        gif_maze.add(maze.copy(), position, cropped_images)

    while True:
        reward = reward_step

        if verbose:
            print('')
            print('Comienzo Turno:')

        old_state = position + vision_from(maze, position) + tuple([
            memory[apply_action(position, maze, posible_actions, action)]
            for _, action in posible_actions.items()
        ])

        if show_gif:
            gif_maze.add(maze.copy(), position, cropped_images)

        if verbose:
            print(' * Estado viejo: %s' % list(old_state))

        action = player.move(old_state, list(posible_actions.values()))

        if verbose:
            print(' * Action: %s' % converted_directions[action])
            if hasattr(player, 'Q'):
                print(' * Value: %s' %
                      choose_action(player.Q, posible_actions, old_state)[1])

        position = apply_action(position, maze, posible_actions, action)
        if memory[position] == 1:
            reward = reward_repeat

        memory[position] = 1

        new_state = position + vision_from(
            maze, position, max_distance=2) + tuple([
                memory[apply_action(position, maze, posible_actions, action)]
                for _, action in posible_actions.items()
            ])

        if verbose:
            print(' * Estado nuevo: %s' % list(new_state))

        if verbose:
            show_maze(maze, position)

        if maze[position] == codes['HOLE']:
            reward = reward_lose
            result = 'lose'
            observations.append((old_state, action, reward, new_state))
            break

        elif maze[position] == codes['TREASURE']:
            reward = reward_win
            result = 'win'
            if verbose:
                print('cantidad explotadas %s' % player.i)
                print('gane %s' % len(observations))
                print('%% explotadas: %.2f' % (player.i / len(observations)))
            observations.append((old_state, action, reward, new_state))
            break

        observations.append((old_state, action, reward, new_state))

    if show_gif:
        gif_maze.add(maze.copy(), position, cropped_images)
        gif_maze.show(maze, cropped_images)
        if hasattr(player, 'Q'):
            GifMazeRewards.from_other(gif_maze).show(
                player.Q,
                cropped_images=cropped_images,
                sprite_size=32,
                posible_actions=posible_actions,
                gif_duration=60)
        elif hasattr(player, 'model'):
            GifMazeRewards.from_other(gif_maze).show_model(player)

    return observations, result
