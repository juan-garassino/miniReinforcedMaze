import os
# from miniReinforcedMaze.generate.generate_maze import generate_maze
from miniReinforcedMaze.agent.actions import max_visibility
from IPython import display
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from datetime import datetime

class GifMaze:
    def __init__(self):
        self.mazes = []

    def add(self, maze=None, position=None, cropped_images=None):
        self.mazes.append((maze, position, cropped_images))

    def show(self, Q_table=None, sprite_size=32, possible_actions=None, gif_duration=60, 
             image_type=4, image_land=1, image_hole=0, image_treasure=3):
        frames = []
        time = 0
        font = ImageFont.load_default()
        possible_actions = possible_actions or []

        for maze, position, cropped_images in self.mazes:
            img = Image.new(mode="RGB", size=(maze.shape[0] * sprite_size, maze.shape[1] * sprite_size), color=(0, 0, 0))

            # Draw the maze and position onto the image
            self._draw_maze(img=img, maze=maze, position=position, cropped_images=cropped_images, 
                            sprite_size=sprite_size, image_type=image_type, image_land=image_land, 
                            image_hole=image_hole, image_treasure=image_treasure)
            
            # Now use ImageDraw to add text, etc.
            draw = ImageDraw.Draw(img)
            draw.text(xy=(sprite_size * (maze.shape[0] - 1) - 10, 10), text=f"t = {time}", fill=(255, 255, 255))

            if Q_table is not None:
                self._draw_q_values(draw=draw, Q_table=Q_table, maze=maze, position=position, 
                                    possible_actions=possible_actions, img_size=img.size)

            time += 1
            frames.append(img)

        self._save_gif(frames=frames, gif_duration=gif_duration)

    def _draw_maze(self, img, maze, position, cropped_images, sprite_size, image_type, image_land, image_hole, image_treasure):
        for height in range(maze.shape[1]):
            for width in range(maze.shape[0]):
                if (width, height) == position:
                    image = cropped_images[image_type]
                else:
                    cell_value = maze[width, height]
                    if cell_value == int(os.environ.get("CODE_LAND")):
                        image = cropped_images[image_land]
                    elif cell_value == int(os.environ.get("CODE_HOLE")):
                        image = cropped_images[image_hole]
                    elif cell_value == int(os.environ.get("CODE_TREASURE")):
                        image = cropped_images[image_treasure]

                # Use the img object to paste images, not the draw object
                img.paste(im=image, box=(width * sprite_size, height * sprite_size))

    def _draw_q_values(self, draw, Q_table, maze, position, possible_actions, img_size):
        state = position + self._max_visibility(maze=maze, position=position) + tuple([1 for _ in possible_actions])
        values = [Q_table.get((i, state), 0) for i in range(len(possible_actions))]
        colors = self._colors_for_values(values=values)

        center_y, center_x = img_size[1] / 2, img_size[0] / 2
        bias = 20

        for i, (x_offset, y_offset) in enumerate([(0, -bias), (0, bias), (-bias, 0), (bias, 0)]):
            draw.text(xy=(center_x + x_offset, center_y + y_offset), text=f"{values[i]:.0f}", fill=colors[i])

    def _max_visibility(self, maze, position):
        # Implement max visibility logic here
        # This is a placeholder and should be replaced with actual implementation
        return (0, 0, 0, 0)

    def _colors_for_values(self, values, num_colors=4):
        colors = [(255, 255, 255) for _ in range(num_colors)]
        colors[np.argmax(values)] = (0, 255, 0)
        colors[np.argmin(values)] = (255, 0, 0)
        return colors

    def _save_gif(self, frames, gif_duration):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"moving_maze_{timestamp}.gif"
        frames[0].save(
            fp=filename,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=gif_duration,
            loop=0,
        )
        print(f"Maze GIF saved as {filename}")


def show_maze(maze=None, position=None):
    convert = {0: "_", -1: "X", 1: "$"}
    maze_str = ""
    for height in range(maze.shape[1]):
        for width in range(maze.shape[0]):
            if (width, height) == position:
                maze_str += "8"
            else:
                maze_str += convert[maze[width, height]]
        maze_str += "\n"
    print(maze_str)