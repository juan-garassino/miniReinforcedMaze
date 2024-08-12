import numpy as np
import random
from enum import Enum
from typing import Tuple, Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import random
from typing import Tuple, Dict
from datetime import datetime

# Constants for maze objects
CODE_TREASURE = 3
CODE_START = 2
CODE_LAND = 1
CODE_HOLE = 0

IMAGEN_POZO = 0
IMAGEN_TIERRA = 1
IMAGEN_LAVA = 2
IMAGEN_SALIDA = 3
IMAGEN_TIPO = 4

class MazeEnvironment:
    def __init__(self, width: int, height: int, density: float = 0.2):
        print("🏗️ Constructing a new mysterious maze...")
        self.width = width
        self.height = height
        self.density = density
        self.maze = self.generate_maze()
        self.start_pos = self.find_start()
        self.treasure_pos = self.find_treasure()
        self.current_pos = self.start_pos
        self.possible_actions = {
            "UP": 0,
            "DOWN": 1,
            "LEFT": 2,
            "RIGHT": 3
        }
        print(f"✨ A {self.width}x{self.height} maze has materialized! Density: {self.density}")
        self.validate_maze()

    def generate_maze(self) -> np.ndarray:
        print("🌀 Weaving the fabric of our maze reality...")
        # Generate a random sample of CODE_LAND and CODE_HOLE based on the density
        maze = np.random.choice(
            a=[CODE_LAND, CODE_HOLE],
            size=self.height * self.width,
            p=[1.0 - self.density, self.density],
        ).reshape((self.height, self.width))

        # Set the start position
        maze[1, 1] = CODE_START

        # Position the treasure at the bottom-right corner of the maze
        maze[self.height - 2, self.width - 2] = CODE_TREASURE

        # Create hole boundaries on the edge of the maze
        maze[0, :] = CODE_HOLE
        maze[-1, :] = CODE_HOLE
        maze[:, 0] = CODE_HOLE
        maze[:, -1] = CODE_HOLE

        # Randomly flip the maze horizontally and vertically
        flip_horizontal = np.random.choice([-1, 1])
        flip_vertical = np.random.choice([-1, 1])
        
        return maze[::flip_horizontal, ::flip_vertical]  # Flip the maze and return

    def validate_maze(self):
        print("Validating maze structure...")
        assert np.all(self.maze[0, :] == CODE_HOLE), "Top border is not all holes"
        assert np.all(self.maze[-1, :] == CODE_HOLE), "Bottom border is not all holes"
        assert np.all(self.maze[:, 0] == CODE_HOLE), "Left border is not all holes"
        assert np.all(self.maze[:, -1] == CODE_HOLE), "Right border is not all holes"
        
        assert np.sum(self.maze == CODE_START) == 1, "There should be exactly one start position"
        assert np.sum(self.maze == CODE_TREASURE) == 1, "There should be exactly one treasure"
        
        assert self.maze[self.start_pos] == CODE_START, "Start position mismatch"
        assert self.maze[self.treasure_pos] == CODE_TREASURE, "Treasure position mismatch"
        
        print("Maze validation complete. Structure is correct.")

    def print_maze_with_codes(self):
        print("Maze structure with codes:")
        for row in self.maze:
            print(" ".join(f"{cell:2d}" for cell in row))

    def find_start(self) -> Tuple[int, int]:
        start_positions = np.where(self.maze == CODE_START)
        return (int(start_positions[0][0]), int(start_positions[1][0]))

    def find_treasure(self) -> Tuple[int, int]:
        treasure_positions = np.where(self.maze == CODE_TREASURE)
        return (int(treasure_positions[0][0]), int(treasure_positions[1][0]))

    def reset(self) -> Tuple[int, int]:
        #print("🔄 Resetting the maze. Our hero returns to the start!")
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        y, x = self.current_pos
        if action == 0:  # up
            new_pos = (y - 1, x)
        elif action == 1:  # down
            new_pos = (y + 1, x)
        elif action == 2:  # left
            new_pos = (y, x - 1)
        elif action == 3:  # right
            new_pos = (y, x + 1)
        else:
            raise ValueError("Invalid action")

        if self.is_valid_move(new_pos):
            self.current_pos = new_pos
        
        reward = self.get_reward()
        done = self.is_done()
        info = {"position": self.current_pos}

        return self.current_pos, reward, done, info

    def is_valid_move(self, pos: Tuple[int, int]) -> bool:
        y, x = pos
        return (0 <= y < self.height and 0 <= x < self.width and 
                self.maze[y, x] != CODE_HOLE)

    def get_reward(self) -> float:
        y, x = self.current_pos
        cell_value = self.maze[y, x]
        if cell_value == CODE_TREASURE:
            print("💎 Jackpot! Our hero found the treasure!")
            return 1.0
        elif cell_value == CODE_HOLE:
            print("💥 Oh no! Our hero fell into a hole!")
            return -0.5
        else:
            return -0.01  # Small negative reward for each step

    def is_done(self) -> bool:
        return self.current_pos == self.treasure_pos or self.maze[self.current_pos] == CODE_HOLE

    def get_state(self) -> np.ndarray:
        y, x = self.current_pos
        visible_area = self.maze[max(0, y-2):y+3, max(0, x-2):x+3].flatten()
        return np.concatenate([visible_area, np.array(self.current_pos)])

    def get_possible_actions(self) -> Dict[str, int]:
        return self.possible_actions

    def get_converted_directions(self) -> Dict[int, str]:
        return {v: k for k, v in self.possible_actions.items()}

    def get_codes(self) -> Dict[str, int]:
        return {
            "HOLE": CODE_HOLE,
            "PATH": CODE_LAND,
            "START": CODE_START,
            "TREASURE": CODE_TREASURE,
        }

def verify_codes(maze_env: MazeEnvironment):
    print("Verifying maze codes...")
    codes = maze_env.get_codes()
    reverse_codes = {v: k for k, v in codes.items()}
    
    unique_values = np.unique(maze_env.maze)
    for value in unique_values:
        if value in reverse_codes:
            print(f"Code {value} corresponds to {reverse_codes[value]}")
        else:
            print(f"Warning: Code {value} not found in the codes dictionary")
    
    print("Code verification complete.")

class RenderMaze:
    def __init__(self, cropped_images: List[Image.Image]):
        self.mazes: List[Tuple[np.ndarray, Tuple[int, int]]] = []
        self.cropped_images = cropped_images
        print(f"GifMaze initialized with {len(self.cropped_images)} images")
        
        # Validate cropped_images
        self._validate_images()

    def _validate_images(self):
        required_images = max(IMAGEN_POZO, IMAGEN_TIERRA, IMAGEN_LAVA, IMAGEN_SALIDA, IMAGEN_TIPO) + 1
        if len(self.cropped_images) < required_images:
            raise ValueError(f"Expected at least {required_images} images, got {len(self.cropped_images)}")
        if not all(isinstance(img, Image.Image) for img in self.cropped_images):
            raise TypeError("All elements must be PIL Image objects")

    def add(self, maze: np.ndarray, position: Tuple[int, int]):
        self.mazes.append((maze, position))
        #print(f"Added maze with shape {maze.shape} and position {position}")

    def show(self, Q_table: Optional[Dict] = None, sprite_size: int = 32, possible_actions: Optional[List[str]] = None, 
             gif_duration: int = 60):
        frames = []
        time = 0
        font = ImageFont.load_default()
        possible_actions = possible_actions or []

        for maze, position in self.mazes:
            img = Image.new(mode="RGB", size=(maze.shape[1] * sprite_size, maze.shape[0] * sprite_size), color=(0, 0, 0))

            self._draw_maze(img=img, maze=maze, position=position, sprite_size=sprite_size)
            
            draw = ImageDraw.Draw(img)
            draw.text(xy=(sprite_size * (maze.shape[1] - 1) - 10, 10), text=f"t = {time}", fill=(255, 255, 255))

            if Q_table is not None:
                self._draw_q_values(draw=draw, Q_table=Q_table, maze=maze, position=position, 
                                    possible_actions=possible_actions, img_size=img.size)

            time += 1
            frames.append(img)

        self._save_gif(frames=frames, gif_duration=gif_duration)

    def _draw_maze(self, img: Image.Image, maze: np.ndarray, position: Tuple[int, int], sprite_size: int):
        for height in range(maze.shape[0]):
            for width in range(maze.shape[1]):
                x, y = width * sprite_size, height * sprite_size
                try:
                    if (height, width) == position:
                        image_index = IMAGEN_TIPO
                    else:
                        cell_value = maze[height, width]
                        if cell_value == CODE_LAND:
                            image_index = IMAGEN_TIERRA
                        elif cell_value == CODE_HOLE:
                            image_index = IMAGEN_POZO
                        elif cell_value == CODE_TREASURE:
                            image_index = IMAGEN_SALIDA
                        else:
                            image_index = IMAGEN_TIERRA  # Default to land image
                    
                    image = self.cropped_images[image_index]
                    
                    if image.size != (sprite_size, sprite_size):
                        print(f"Warning: Image size mismatch. Expected {(sprite_size, sprite_size)}, got {image.size}. Resizing...")
                        image = image.resize((sprite_size, sprite_size), Image.LANCZOS)
                    
                    img.paste(im=image, box=(x, y))
                    #print(f"Drew image for cell type {maze[height, width]} at position ({height}, {width})")
                    
                except IndexError as e:
                    print(f"Error accessing image: {e}")
                    print(f"Attempted to access index: {image_index}")
                    print(f"Available indices: 0 to {len(self.cropped_images) - 1}")
                    raise
                except Exception as e:
                    print(f"Unexpected error when drawing maze cell at ({height}, {width}): {e}")
                    raise

    def _draw_q_values(self, draw: ImageDraw.Draw, Q_table: Dict, maze: np.ndarray, position: Tuple[int, int], 
                       possible_actions: List[str], img_size: Tuple[int, int]):
        state = position + self._max_visibility(maze=maze, position=position) + tuple([1 for _ in possible_actions])
        values = [Q_table.get((i, state), 0) for i in range(len(possible_actions))]
        colors = self._colors_for_values(values=values)

        center_y, center_x = img_size[1] / 2, img_size[0] / 2
        bias = 20

        for i, (x_offset, y_offset) in enumerate([(0, -bias), (0, bias), (-bias, 0), (bias, 0)]):
            draw.text(xy=(center_x + x_offset, center_y + y_offset), text=f"{values[i]:.0f}", fill=colors[i])

    def _max_visibility(self, maze: np.ndarray, position: Tuple[int, int], visibility_range: int = 2) -> Tuple[int, ...]:
        height, width = maze.shape
        y, x = position
        
        y_min = max(0, y - visibility_range)
        y_max = min(height, y + visibility_range + 1)
        x_min = max(0, x - visibility_range)
        x_max = min(width, x + visibility_range + 1)
        
        visible_area = maze[y_min:y_max, x_min:x_max]
        
        return tuple(visible_area.flatten())

    def _colors_for_values(self, values: List[float], num_colors: int = 4) -> List[Tuple[int, int, int]]:
        colors = [(255, 255, 255) for _ in range(num_colors)]
        colors[np.argmax(values)] = (0, 255, 0)
        colors[np.argmin(values)] = (255, 0, 0)
        return colors

    def _save_gif(self, frames: List[Image.Image], gif_duration: int):
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

    @staticmethod
    def crop_images(
        input_path: str,
        image_names: List[str],
        height: int = 16,
        width: int = 16,
        sprite_size: int = 32,
        show: bool = False,
        return_indexes: Optional[List[int]] = None,
    ) -> List[Image.Image] | Tuple[List[Image.Image], List[int]]:
        images = []
        indexes = []

        for image_name in image_names:
            image_path = os.path.join(input_path, image_name)
            with Image.open(image_path) as im:
                imgwidth, imgheight = im.size
                for i in range(0, imgheight, height):
                    for j in range(0, imgwidth, width):
                        box = (j, i, j + width, i + height)
                        cropped_image = im.crop(box)
                        resized_image = cropped_image.resize(
                            (sprite_size, sprite_size), Image.LANCZOS
                        )

                        if show:
                            resized_image.show()

                        images.append(resized_image)

                        if return_indexes and len(images) - 1 in return_indexes:
                            indexes.append(len(images) - 1)

        if return_indexes is not None:
            if -1 in return_indexes:
                indexes.append(-1)
            return [images[i] for i in indexes], indexes
        else:
            return images

def create_and_render_maze(
    width: int,
    height: int,
    density: float,
    input_path: str,
    image_names: List[str],
    imagen_indexes: List[int],
    num_steps: int = 10,
    sprite_size: int = 32,
    gif_duration: int = 500,
    crop_height: int = 16,
    crop_width: int = 16
):
    try:
        # Create the maze environment
        print("Creating maze environment...")
        env = MazeEnvironment(width, height, density)

        # Verify codes
        verify_codes(env)

        # Print maze with codes for debugging
        env.print_maze_with_codes()

        # Display the initial maze
        print("\nInitial maze state:")
        print_maze(env.maze, env.current_pos)

        # Crop images
        print("\nCropping images...")
        cropped_images, _ = crop(
            input_path=input_path,
            image_names=image_names,
            height=crop_height,
            width=crop_width,
            sprite_size=sprite_size,
            return_indexes=imagen_indexes
        )
        print(f"Cropped {len(cropped_images)} images")

        # Create RenderMaze instance
        render_maze = RenderMaze(cropped_images)

        # Add initial maze state to RenderMaze
        render_maze.add(env.maze, env.current_pos)

        # Simulate some steps in the maze
        print("\nSimulating steps in the maze:")
        for step in range(num_steps):
            action = random.choice(list(env.possible_actions.values()))
            new_pos, reward, done, _ = env.step(action)
            print(f"Step {step + 1}: Action: {env.get_converted_directions()[action]}, "
                  f"New position: {new_pos}, Reward: {reward}, Done: {done}")
            
            # Add new maze state to RenderMaze
            render_maze.add(env.maze, env.current_pos)
            
            # Display updated maze state
            print_maze(env.maze, env.current_pos)
            
            if done:
                print("Maze exploration completed!")
                break

        # Generate the GIF
        print("\nGenerating GIF animation...")
        render_maze.show(sprite_size=sprite_size, gif_duration=gif_duration)

        print("\nMaze exploration and GIF creation completed successfully!")

    except AssertionError as e:
        print(f"Maze validation failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def print_maze(maze: np.ndarray, position: Tuple[int, int]):
    symbols = {
        0: '⬛',  # HOLE
        1: '⬜',  # PATH
        2: '🟩',  # START
        3: '💎'   # TREASURE
    }
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if (y, x) == position:
                print('🦸', end='')  # hero
            else:
                print(symbols[maze[y, x]], end='')
        print()

# Example usage
def main():
    width, height = 10, 10
    density = 0.2
    input_path = '/Users/juan-garassino/Code/juan-garassino/miniNetworks/miniReinforcedMaze/data'
    image_names = ["cave.png", "objects.png", "agus.png"]
    imagen_indexes = [16 * 8 + 2, 16 * 9, 16 * 20, 378, -1]
    num_steps = 20
    sprite_size = 32
    gif_duration = 500
    crop_height = 16
    crop_width = 16

    create_and_render_maze(
        width=width,
        height=height,
        density=density,
        input_path=input_path,
        image_names=image_names,
        imagen_indexes=imagen_indexes,
        num_steps=num_steps,
        sprite_size=sprite_size,
        gif_duration=gif_duration,
        crop_height=crop_height,
        crop_width=crop_width
    )



def crop(
    input_path: str,
    image_names: List[str],
    height: int = 16,
    width: int = 16,
    sprite_size: int = 32,
    show: bool = False,
    return_indexes: Optional[List[int]] = None,
) -> List[Image.Image] | Tuple[List[Image.Image], List[int]]:
    """
    Crop and resize images from the given path and names.

    Args:
        input_path (str): Path to the directory containing the images.
        image_names (List[str]): List of image file names to process.
        height (int): Height of each crop. Defaults to 16.
        width (int): Width of each crop. Defaults to 16.
        sprite_size (int): Size to which cropped images will be resized. Defaults to 32.
        show (bool): If True, display the resized images. Defaults to False.
        return_indexes (Optional[List[int]]): If provided, return only the specified indexes.

    Returns:
        If return_indexes is None:
            List[Image.Image]: List of cropped and resized images.
        If return_indexes is provided:
            Tuple[List[Image.Image], List[int]]: Tuple of (cropped and resized images, indexes).
    """
    images = []
    indexes = []

    for image_name in image_names:
        image_path = os.path.join(input_path, image_name)
        with Image.open(image_path) as im:
            imgwidth, imgheight = im.size
            for i in range(0, imgheight, height):
                for j in range(0, imgwidth, width):
                    box = (j, i, j + width, i + height)
                    cropped_image = im.crop(box)
                    resized_image = cropped_image.resize(
                        (sprite_size, sprite_size), Image.LANCZOS
                    )

                    if show:
                        resized_image.show()

                    images.append(resized_image)

                    if return_indexes and len(images) - 1 in return_indexes:
                        indexes.append(len(images) - 1)

    if return_indexes is not None:
        if -1 in return_indexes:
            indexes.append(-1)
        return [images[i] for i in indexes], indexes
    else:
        return images

# Example usage
def main():
    width, height = 10, 10
    density = 0.2
    input_path = '/Users/juan-garassino/Code/juan-garassino/miniNetworks/miniReinforcedMaze/data'
    image_names = ["cave.png", "objects.png", "agus.png"]
    imagen_indexes = [16 * 8 + 2, 16 * 9, 16 * 20, 378, -1]
    num_steps = 20
    sprite_size = 32
    gif_duration = 500

    create_and_render_maze(
        width=width,
        height=height,
        density=density,
        input_path=input_path,
        image_names=image_names,
        imagen_indexes=imagen_indexes,
        num_steps=num_steps,
        sprite_size=sprite_size,
        gif_duration=gif_duration
    )

if __name__ == "__main__":
    main()