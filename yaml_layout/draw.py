import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
from typing import Any, Dict, List
from matplotlib.axes import Axes
import random
import os
import hashlib

# Set matplotlib Chinese font display and minus sign issue
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font for Chinese display support
plt.rcParams['axes.unicode_minus'] = False    # Fix minus sign display issue

# Default floor color pool (extracted from 1_english.yaml)
DEFAULT_FLOOR_COLORS = [
    # "#E5E9ED",  # Bathroom - light gray-blue
    # "#F1F2F3",  # Kitchen - light gray
    "#FDFBEB",  # Secondary Bedroom / Master Bedroom - beige
    "#F7F3E3",  # Living Room / Living Room 1 - light beige
    "#F5F1E1",  # Living Room / Living Room 1 - light beige
]

# Default furniture color pool (extracted from 1_english.yaml)
DEFAULT_FURNITURE_COLORS = [
    "#EEF2FA",  # Toilet/Washbasin/Sink - light blue-white
    "#EEEEEE",  # Refrigerator/Stove - light gray
    "#FFEFEF",  # Bed - light pink
    "#D5B575",  # Wardrobe - wood color
    "#F5E9CC",  # Desk - light wood color
    # "#E0C0A0",  # Dining Table - tan
    "#F6F3CA",  # Sofa - light yellow
    "#CCC3A5",  # Coffee Table - gray-brown
    "#DEB78A",  # TV Cabinet - light brown
    "#D8D8D8",  # Treadmill - gray
]

class House:
    def __init__(self, ax: Axes, yaml_path: str = None, yaml_content: str = None) -> None:
        """
        Initialize the House class for drawing house layout.

        Parameters:
            ax (matplotlib.axes.Axes): matplotlib Axes object for drawing.
            yaml_path (str): Path to YAML configuration file containing house layout information.
        """
        self.ax = ax
        if yaml_path:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                self.data = yaml.safe_load(f)
            # Use filename as random seed
            filename = os.path.basename(yaml_path)
        else:
            self.data = yaml.safe_load(yaml_content)
            # Use hash of yaml content as random seed
            filename = str(hash(yaml_content))
        
        self.random_seed = int(hashlib.md5(filename.encode()).hexdigest(), 16) % (2**32)
        self.rng = random.Random(self.random_seed)
        
        # Furniture color counter, used to change color every two pieces of furniture
        self.furniture_counter = 0
        self.current_furniture_color = None
    
    def get_random_floor_color(self):
        """Randomly select a color from the default floor color pool"""
        return self.rng.choice(DEFAULT_FLOOR_COLORS)
    
    def get_random_furniture_color(self):
        """Randomly select a color from the default furniture color pool, changing every two pieces of furniture"""
        if self.furniture_counter % 2 == 0:
            self.current_furniture_color = self.rng.choice(DEFAULT_FURNITURE_COLORS)
        self.furniture_counter += 1
        return self.current_furniture_color
    

    def draw_wall( self, p1: List[float], p2: List[float], width: float = 1.0, color: str = '#999999') -> None:
        """
        Draw a wall: a rectangle from p1 to p2, with configurable line width and color.

        Parameters:
            p1 (list): Wall starting point coordinates [x, y], as a list of floats.
            p2 (list): Wall ending point coordinates [x, y], as a list of floats.
            linewidth (int, optional): Wall line width, default is 1.
            color (str, optional): Wall color, default is 'gray', supports matplotlib color values.
        """
        x1, y1 = p1
        x2, y2 = p2
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        line = patches.Rectangle(p1, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)+width, width if (angle+45)//90%2==0 else -width, angle=angle, linewidth=0, edgecolor=color, facecolor=color, zorder=3)
        self.ax.add_patch(line)

    def draw_floor(self, bottom_left, width, height, color='#F6E9CA', hatch=None):
        """
        Draw floor area (rectangular fill).

        Parameters:
            bottom_left (list): Floor bottom-left corner coordinates [x, y], as a list of floats.
            width (float): Floor width.
            height (float): Floor height.
            color (str, optional): Fill color, default is '#F6E9CA', supports matplotlib color values.
            hatch (str, optional): Fill pattern, options include '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', default is None.
        """
        assert hatch is None or hatch in ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'], "Floor fill pattern not supported"
        x, y = bottom_left
        corners = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
        edgecolor = '#00000011' if hatch else 'none'  # Show semi-transparent border with fill pattern, otherwise no border
        polygon = patches.Polygon(corners, closed=True, facecolor=color, edgecolor=edgecolor, hatch=hatch, zorder=1)
        # Move to bottom layer (zorder=1)

        self.ax.add_patch(polygon)
        

    def draw_door(self, p1, p2, p3, width=1, color='white', alpha=0.5, padding=0.5):
        """
        Draw a door: includes rectangular part (door leaf) and sector part showing opening effect.

        Parameters:
            p1 (list): Door hinge position [x, y], as a list of floats.
            p2 (list): Door leaf one side position [x, y], as a list of floats.
            p3 (list): Door leaf other side position [x, y], as a list of floats.
            width (int, optional): Door leaf line width, default is 6.
            color (str, optional): Door leaf color, default is 'white', supports matplotlib color values.
            alpha (float, optional): Sector transparency, range 0 to 1, default is 0.5.
            padding (float, optional): Door leaf inner padding, default is 0.
        """
        assert padding < (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) / 2, "Padding too large, cannot draw door"
        if p2[0] == p3[0] or p2[1] == p3[1]:
            p1, p2 = p2, p1
        # Adjust padding
        if p1[0] == p2[0]:
            p1[1] = p1[1] + padding if p1[1] < p2[1] else p1[1] - padding
            p2[1] = p2[1] + padding if p2[1] < p1[1] else p2[1] - padding
            p3[1] = p3[1] + padding if p1[1] < p2[1] else p3[1] - padding
            p3[0] = p3[0] + padding if p3[0] < p1[0] else p3[0] - padding
        elif p1[1] == p2[1]:
            p1[0] = p1[0] + padding if p1[0] < p2[0] else p1[0] - padding
            p2[0] = p2[0] + padding if p2[0] < p1[0] else p2[0] - padding
            p3[0] = p3[0] + padding if p1[0] < p2[0] else p3[0] - padding
            p3[1] = p3[1] + padding if p3[1] < p1[1] else p3[1] - padding

        # Draw rectangular part
        x1, y1 = p1 if p1[0] < p2[0] or p1[1] < p2[1] else p2
        x2, y2 = p2 if p1[0] < p2[0] or p1[1] < p2[1] else p1
        # Bottom-left corner is p1, top-right corner is p2

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        line = patches.Rectangle((x1, y1), np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), width if (angle+45)//90%2==0 else -width, angle=angle, linewidth=0, edgecolor=color, facecolor=color, zorder=4)
        self.ax.add_patch(line)
        # edge_line = plt.Line2D([p1[0], p2[0]], [p1[1], p2[1]], linewidth=width, color='gray')
        # self.ax.add_line(edge_line)
        # inner_line = plt.Line2D([p1[0], p2[0]], [p1[1], p2[1]], linewidth=width - 2, color=color)
        # self.ax.add_line(inner_line)
        # mid_line = plt.Line2D([p1[0], p2[0]], [p1[1], p2[1]], linewidth=1, color='grey')
        # self.ax.add_line(mid_line) 
        # Draw sector (opening arc)
        radius = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        theta1 = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        theta2 = np.degrees(np.arctan2(p3[1] - p1[1], p3[0] - p1[0]))
        theta1, theta2 = (theta1 + 360) % 360, (theta2 + 360) % 360

        if abs(theta2 - theta1) <= 180:
            theta_start, theta_end = min(theta1, theta2), max(theta1, theta2)
        else:
            theta_start, theta_end = max(theta1, theta2), min(theta1, theta2)

        wedge = patches.Wedge(p1, radius, theta_start, theta_end, facecolor=color, edgecolor='gray', alpha=alpha, zorder=4)
        self.ax.add_patch(wedge)

    def draw_furniture(self, furn_type, bottom_left, width, height, name=None, color='brown', rotation=0, fontsize=8, padding=0.5, image_path=None):
        """
        Draw furniture (default as rectangle), labeling the name at the center of the furniture.

        Parameters:
            furn_type (str): Furniture type, options are 'rectangle' (rectangle) or 'image' (image).
            bottom_left (list): Furniture bottom-left corner coordinates [x, y], as a list of floats.
            width (float): Furniture width.
            height (float): Furniture height.
            name (str, optional): Furniture name, default is None.
            color (str, optional): Furniture color, default is 'brown', supports matplotlib color values.
            rotation (float, optional): Furniture rotation angle (degrees), default is 0.
            fontsize (int, optional): Furniture name font size, default is 8.
            padding (float, optional): Furniture inner padding, default is 0.
            image_path (str, optional): Image path (used when furn_type is 'image'), default is None.
        """
        # Dynamically adjust font based on furniture size and text length, maintaining visual balance
        # Use square root to make font size change more gradual, avoiding extreme size differences
        furniture_size = min(width, height)  # Use smaller side as reference
        text_length = len(name) if name else 1  # Text length
        
        # Use square root to make font scaling more gradual, base size is 6
        size_factor = np.sqrt(furniture_size / 6.0)
        # Consider text length, longer text needs smaller font
        text_factor = 1.0 / np.sqrt(max(1, text_length / 3.0))
        
        adjusted_fontsize = fontsize * size_factor * text_factor
        # Narrow font size range to make it more balanced
        adjusted_fontsize = max(6, min(adjusted_fontsize, 9))

        # Adjust padding and ensure non-negative dimensions
        fx, fy = bottom_left[0] + padding, bottom_left[1] + padding
        fwidth, fheight = width - padding * 2, height - padding

        if furn_type == 'rectangle':
            rect = patches.Rectangle((fx, fy), fwidth, fheight, linewidth=1, edgecolor='#DDDDDDBB', facecolor=color, zorder=2)
            self.ax.add_patch(rect)
        elif furn_type == 'image':
            img = mpimg.imread(image_path)
            extent = [fx, fx + fwidth, fy, fy + fheight]
            self.ax.imshow(img, extent=extent, origin='upper')
        # 截断过长文字（超过家具框1个字符以上）
        if name:
            text_width = fwidth if fwidth > fheight * 0.75 else fheight
            max_chars = int(text_width / (adjusted_fontsize * 0.5))
            if len(name) > max_chars + 5:
                name = name[:max_chars] + '.'
        
        if fwidth > fheight * 0.75:
            self.ax.text(fx + fwidth / 2, fy + fheight / 2, name, ha='center', va='center', fontsize=adjusted_fontsize, color='#00000066', weight='bold', fontfamily='Comic Sans MS')
        else:
            self.ax.text(fx + fwidth / 2, fy + fheight / 2, name, ha='center', va='center', fontsize=adjusted_fontsize, rotation=90, color='#00000066', weight='bold', fontfamily='Comic Sans MS')
    def draw_room(self, room):
        """
        Draw a single room based on YAML configuration.

        Parameters:
            room (dict): Room configuration dictionary, including position, floor, walls, doors and furniture information.
        """
        # Room position
        pos = room.get('position', {})
        x = pos.get('x', 0)          # Room bottom-left corner x coordinate, default is 0
        y = pos.get('y', 0)          # Room bottom-left corner y coordinate, default is 0
        width = pos.get('width', 10) # Room width, default is 10
        height = pos.get('height', 10) # Room height, default is 10

        # Draw floor
        floor_conf = room.get('floor', {})
        floor_color = floor_conf.get('color', self.get_random_floor_color())
        floor_hatch = floor_conf.get('hatch', None)
        self.draw_floor((x, y), width, height, color=floor_color, hatch=floor_hatch)

        # Draw surrounding walls (default draw all borders, if walls field specified, can customize each wall coordinates)
        wall_defaults = {'width': 1, 'color': '#999999'} # Default wall style
        walls = room.get('walls', {})
        # Top wall
        top_wall = walls.get('top', True)
        if top_wall:
            if isinstance(top_wall, dict) and 'points' in top_wall:
                p1, p2 = top_wall['points']
            else:
                p1, p2 = [x, y + height], [x + width, y + height]
            self.draw_wall(p1, p2, **wall_defaults)
        # Right wall
        right_wall = walls.get('right', True)
        if right_wall:
            if isinstance(right_wall, dict) and 'points' in right_wall:
                p1, p2 = right_wall['points']
            else:
                p1, p2 = [x + width, y], [x + width, y + height]
            self.draw_wall(p1, p2, **wall_defaults)
        # Bottom wall
        bottom_wall = walls.get('bottom', True)
        if bottom_wall:
            if isinstance(bottom_wall, dict) and 'points' in bottom_wall:
                p1, p2 = bottom_wall['points']
            else:
                p1, p2 = [x, y], [x + width, y]
            self.draw_wall(p1, p2, **wall_defaults)
        # Left wall
        left_wall = walls.get('left', True)
        if left_wall:
            if isinstance(left_wall, dict) and 'points' in left_wall:
                p1, p2 = left_wall['points']
            else:
                p1, p2 = [x, y], [x, y + height]
            self.draw_wall(p1, p2, **wall_defaults)

        # Draw extra walls (if configured)
        extra_walls = walls.get('extra', [])
        for ew in extra_walls:
            pts = ew.get('points') # Extra wall start and end point coordinates
            kwargs = {
                'width': ew.get('width', 1), # Line width
                'color': ew.get('color', 'gray')    # Color
            }
            self.draw_wall(pts[0], pts[1], **kwargs)

        # Draw doors
        doors = room.get('doors', [])
        for door in doors:
            pts = door.get('points', []) # Three key points of the door
            if len(pts) == 3:
                d_width = door.get('width', 1)     # Door leaf line width
                d_color = door.get('color', 'white') # Door leaf color
                d_alpha = door.get('alpha', 0.5)   # Sector transparency
                padding = door.get('padding', 0.5)   # Inner padding
                self.draw_door(pts[0], pts[1], pts[2], width=d_width, color=d_color, alpha=d_alpha, padding=padding)

        # Draw furniture
        furnitures = room.get('furniture', [])
        for furn in furnitures:
            furn_type = furn.get('type', 'rectangle')
            pos_f = furn.get('position', {})
            config = furn.get('config', {})
            
            self.draw_furniture(furn_type=furn_type,
                                bottom_left=(pos_f.get('x', 0), pos_f.get('y', 0)),
                                width=pos_f.get('width', 10),
                                height=pos_f.get('height', 10),
                                name=furn.get('name', ''),
                                color=config.get('color', self.get_random_furniture_color()),
                                rotation=furn.get('rotation', 0),
                                fontsize=config.get('fontsize', 8),
                                padding=furn.get('padding', 0.5),
                                image_path=furn.get('image', None))

    def draw(self) -> Axes:
        """
        Draw the entire house layout based on YAML configuration.

        Parameters:
            None (uses YAML data loaded during initialization).

        Returns:
            matplotlib.axes.Axes: The drawn Axes object.

        YAML top-level structure requirements:
            house:
              size: { x: house width (float), y: house height (float) }
              rooms: [ <configuration of each room (dict)> ]
        """
        house = self.data.get('house', {})          # Get overall house configuration
        size = house.get('size', {'x': 100, 'y': 100}) # House size
        rooms = house.get('rooms', [])              # Room list

        for room in rooms:
            self.draw_room(room)

        return self.ax


if __name__ == '__main__':
    yaml_file = "E:/Users/Pearcat/Desktop/first/1-data_simulation/yaml_layout/output/4.yaml"  # YAML file path
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)  # Create canvas, size 8x6

    house = House(ax, yaml_file)     # Initialize House object
    house.draw()                    # Draw house layout
    
    ax.set_xlim(-10, 105)             # Set x-axis range
    ax.set_ylim(-10, 105)             # Set y-axis range
    ax.set_aspect('equal')          # Set equal aspect ratio for axes
    
    plt.show()                      # Display figure
