from PIL import Image, ImageDraw
import IPython.display
import os
from miniReinforcedMaze.resources.generate_maze import generate_maze
from miniReinforcedMaze.agent.actions import vision_from
from IPython import display
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class GifMaze:
    def __init__(self):
        self.mazes = []

    def add(self, maze, position, maze_images):
        """
        Add a maze and its position to the collection.
        """
        self.mazes.append((maze, position, maze_images))

    def show(
        self,
        maze,
        maze_images,
        sprite_size=32,
        image_type=4,
        image_land=1,
        image_hole=0,
        image_treasure=3,
    ):
        """
        Generate and display the animated GIF of the mazes.
        Pass the required image indices for land, hole, treasure, and the maze image type.
        """
        frames = []
        time = 0

        for maze, position, maze_images in self.mazes:
            #print(maze.shape[0])
            img = Image.new(
                "RGB",
                (maze.shape[0] * sprite_size, maze.shape[1] * sprite_size),
                color=(0, 0, 0),
            )

            for height in range(maze.shape[1]):
                for width in range(maze.shape[0]):
                    if (width, height) == position:
                        img.paste(
                            maze_images[image_type],
                            (width * sprite_size, height * sprite_size),
                        )
                    else:
                        cell_value = maze[width, height]
                        if cell_value == os.environ.get("CODE_LAND"):
                            img.paste(
                                maze_images[image_land],
                                (width * sprite_size, height * sprite_size),
                            )
                        elif cell_value == os.environ.get("CODE_HOLE"):
                            img.paste(
                                maze_images[image_hole],
                                (width * sprite_size, height * sprite_size),
                            )
                        elif cell_value == os.environ.get("CODE_TREASURE"):
                            img.paste(
                                maze_images[image_treasure],
                                (width * sprite_size, height * sprite_size),
                            )

            draw = ImageDraw.Draw(img)
            draw.text(
                (sprite_size * (maze.shape[0] - 1) - 10, 10),
                f"t = {time}",
                (255, 255, 255),
            )
            time += 1
            frames.append(img)

        frames[0].save(
            "moving_maze.gif.png",
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=len(frames) * 12,
            loop=0,
        )
        IPython.display.display(display.Image(filename="moving_maze.gif.png"))


def colors_for_values(values, num_colors=4):
    """
    Generate a list of colors based on the values.
    The maximum value is colored as green, the minimum value as red,
    and the remaining values are set as white by default.
    """
    colors = [(255, 255, 255) for _ in range(num_colors)]
    colors[np.argmax(values)] = (0, 255, 0)
    colors[np.argmin(values)] = (255, 0, 0)
    return colors

class GifMazeRewards(object):
    @classmethod
    def from_other(cls, gif_maze):
        return cls(mazes=gif_maze.mazes)

    def __init__(self, mazes):
        self.mazes = mazes
        self.images = []

    def add(self, maze, position):
        self.mazes.append((maze, position))

    def show_model(self, player, cropped_images, sprite_size, posible_actions, gif_duration):
        frames = []
        time = 0
        font = ImageFont.load_default()
        for maze, position in self.mazes:
            img = Image.new(
                'RGB',
                (maze.shape[0] * sprite_size, maze.shape[1] * sprite_size),
                color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            convert = {0: 1, -1: 0, 1: 3}
            for height in range(maze.shape[1]):
                for width in range(maze.shape[0]):
                    if (width, height) == position:
                        img.paste(cropped_images[4],
                                  (width * sprite_size, height * sprite_size))
                    else:
                        img.paste(cropped_images[convert[maze[width, height]]],
                                  (width * sprite_size, height * sprite_size))

            state = position + vision_from(maze, position) + tuple(
                [1 for a in posible_actions])

            draw.text((sprite_size * (maze.shape[0] - 1) - 10, 10),
                      "t = %s" % time, (255, 255, 255))

            values = player.estimates_for(state)
            colors = colors_for_values(values)

            center_y = 25
            center_x = img.size[0] / 2
            bias = 20

            draw.text((center_x, center_y - bias), "%.0f" % values[0], colors[0])
            draw.text((center_x, center_y + bias), "%.0f" % values[1], colors[1])
            draw.text((center_x - bias, center_y), "%.0f" % values[2], colors[2])
            draw.text((center_x + bias, center_y), "%.0f" % values[3], colors[3])
            time += 1
            frames.append(img)

        frames[0].save('moving_maze.gif.png',
                       format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=len(frames) * gif_duration,
                       loop=0)
        IPython.display.display(display.Image(filename="moving_maze.gif.png"))

    def show(self, Q_table, cropped_images=[], sprite_size=32, posible_actions=[],
             gif_duration=60):
        frames = []
        time = 0
        font = ImageFont.load_default()
        for maze, position, _ in self.mazes:
            img = Image.new(
                'RGB',
                (maze.shape[0] * sprite_size, maze.shape[1] * sprite_size),
                color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            convert = {0: 1, -1: 0, 1: 3}
            for height in range(maze.shape[1]):
                for width in range(maze.shape[0]):
                    if (width, height) == position:
                        img.paste(cropped_images[4],
                                  (width * sprite_size, height * sprite_size))
                    else:
                        img.paste(cropped_images[convert[maze[width, height]]],
                                  (width * sprite_size, height * sprite_size))

            state = position + vision_from(maze, position) + tuple(
                [1 for a in posible_actions])
            draw.text((sprite_size * (maze.shape[0] - 1) - 10, 10),
                      "t = %s" % time, (255, 255, 255))

            colors = colors_for_values([Q_table[(i, state)] for i in range(4)])

            center_y = 25
            center_x = img.size[0] / 2
            bias = 20

            draw.text((center_x, center_y - bias), "%.0f" % Q_table[(0, state)], colors[0])
            draw.text((center_x, center_y + bias), "%.0f" % Q_table[(1, state)], colors[1])
            draw.text((center_x - bias, center_y), "%.0f" % Q_table[(2, state)], colors[2])
            draw.text((center_x + bias, center_y), "%.0f" % Q_table[(3, state)], colors[3])
            time += 1
            frames.append(img)

        frames[0].save('moving_maze.gif.png',
                       format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=len(frames) * gif_duration,
                       loop=0)
        IPython.display.display(display.Image(filename="moving_maze.gif.png"))

def show_maze(maze, position):
    # This shows the maze as a string represensation
    string = ""
    convert = {0: '_', -1: 'X', 1: '$'}
    for height in range(maze.shape[1]):
        #print(height)
        for width in range(maze.shape[0]):
            #print(width)
            if (width, height) == position:
                string += "8"
            else:
                #print(maze)
                #print(maze[width, height])
                string += convert[maze[width, height]]
        string += '\n'
    print(string)


# show_maze(generate_maze(WIDTH, HEIGHT, 0.05)[0], (1,1)) # The position is a tuple
