import os
import yaml
import numpy as np
import math
import heapq
from scipy.ndimage import distance_transform_edt, gaussian_filter
import random
import datetime
import bisect
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
class GridPoint:
    def __init__(self, room=None, obstacle=False, obstacle_class=None, obstacle_name=None, obstacle_id=None, potential=0.0):
        """Represents a single grid cell in the navigation matrix."""
        self.room = room                     # Room name the cell belongs to
        self.obstacle = obstacle             # True if the cell contains an obstacle
        self.obstacle_class = obstacle_class # Obstacle category 'wall', 'furniture'
        self.obstacle_name = obstacle_name   # Obstacle name (e.g., "wall", "chair")
        self.obstacle_id = obstacle_id       # Unique ID of obstacle (None for walls)
        self.potential = potential           # Potential field value for navigation
        
        assert self.obstacle_class in ['wall', 'furniture', None]

class Navigation:
    def __init__(self, yaml_source):
        """
        Initialize Navigation class with YAML file path or YAML content string.

        Parameters:
            yaml_source (str): Path to the YAML file or YAML content string containing house layout.
        """
        if os.path.isfile(yaml_source):
            with open(yaml_source, 'r', encoding='utf-8') as f:
                self.data = yaml.safe_load(f)
        else:
            self.data = yaml.safe_load(yaml_source)
        
        self.house = self.data.get('house', {})
        self.size = self.house.get('size', {'x': 100, 'y': 100})
        self.rooms = self.house.get('rooms', [])
        
        # Grid size
        self.N = int(self.size['x'])+1  # Number of rows
        self.M = int(self.size['y'])+1  # Number of columns
        

        # Initialize grid as 2D list of GridPoint objects
        self.grid = [[GridPoint() for _ in range(self.M)] for _ in range(self.N)]
        
        # List to store furniture obstacles with IDs
        self.obstacles = []
        
        # Build grid matrix
        self._build_grid()
    
    def get_grid(self):
        """Return obstacle grid matrix"""
        return [[1 if self.grid[i][j].obstacle else 0 for j in range(self.M)] for i in range(self.N)]
    def get_potential(self):
        """Return potential field grid matrix"""
        return [[self.grid[i][j].potential for j in range(self.M)] for i in range(self.N)]
    
    def draw_potential(self, ax, potential_matrix, cmap='viridis'):
        """Draw potential field grid matrix"""
        cmap = plt.get_cmap(cmap)
        for i in range(potential_matrix.shape[0]):
            for j in range(potential_matrix.shape[1]):
                if potential_matrix[i, j] != 0 and potential_matrix[i, j] != -1:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor= cmap(1-potential_matrix[i, j]), alpha=0.8, edgecolor=cmap(1-potential_matrix[i, j])))
    

    def find_path(self, start, end, potential=False, potential_k=1.0, potential_decay=1, potential_rand=True, smooth=True, allow_furniture_start=True, allow_furniture_end=True):
        """
        Find a path from start to end using A* algorithm, using random potential field to simulate human trajectory behavior.

        Parameters:
            start (tuple or int): Start point, format (x, y) or obstacle ID.
            end (tuple or int): End point, format (x, y) or obstacle ID.
            potential (bool): If True, use potential field to adjust cost, default is False.
            potential_k (float): Potential field weight, default is 1.0.
            potential_decay (float): Potential field decay rate, default is 1.0.
            potential_rand (bool): If True, randomly adjust potential field values, default is True.
            smooth (bool): If True, use Bezier interpolation to smooth path, default is True.
            allow_furniture_start (bool): If True, allow start point on furniture, default is True.
            allow_furniture_end (bool): If True, allow end point on furniture, default is True.

        Returns:
            list: List of (i, j) grid coordinates representing the path, or None if no path found.
        """
        # Convert start and end to grid indices
        start_i, start_j = self._to_grid_coords(start)
        end_i, end_j = self._to_grid_coords(end)
        
        # Check boundaries
        if not (0 <= start_i < self.N and 0 <= start_j < self.M):
            raise ValueError("Start position out of grid bounds")
        if not (0 <= end_i < self.N and 0 <= end_j < self.M):
            raise ValueError("End position out of grid bounds")
        
        # Check if start and end are outside the house (room is None)
        if self.grid[start_i][start_j].room is None:
            raise ValueError(f"Start position ({start_i}, {start_j}) is outside the house, cannot navigate")
        if self.grid[end_i][end_j].room is None:
            raise ValueError(f"End position ({end_i}, {end_j}) is outside the house, cannot navigate")
        
        # Check if start and end are on furniture, record furniture IDs to ignore during navigation
        ignore_furniture_ids = []
        
        # Check start point
        if self.grid[start_i][start_j].obstacle:
            if self.grid[start_i][start_j].obstacle_class == 'wall':
                raise ValueError("Start position is on a wall, cannot navigate")
            elif self.grid[start_i][start_j].obstacle_class == 'furniture':
                if not allow_furniture_start:
                    raise ValueError("Start position is on furniture and furniture start not allowed")
                # Record furniture ID to ignore during navigation
                furniture_id = self.grid[start_i][start_j].obstacle_id
                if furniture_id is not None:
                    ignore_furniture_ids.append(furniture_id)
        
        # Check end point
        if self.grid[end_i][end_j].obstacle:
            if self.grid[end_i][end_j].obstacle_class == 'wall':
                raise ValueError("End position is on a wall, cannot navigate")
            elif self.grid[end_i][end_j].obstacle_class == 'furniture':
                if not allow_furniture_end:
                    raise ValueError("End position is on furniture and furniture end not allowed")
                # Record furniture ID to ignore during navigation
                furniture_id = self.grid[end_i][end_j].obstacle_id
                if furniture_id is not None and furniture_id not in ignore_furniture_ids:
                    ignore_furniture_ids.append(furniture_id)

        # Generate complete potential field (without excluding any furniture)
        if potential:
            self._generate_potential(sign='positive', target='all', strength=1, decay=potential_decay, rand=potential_rand)

        # A* algorithm implementation, ignoring specified furniture during navigation
        open_set = [(0, start_i, start_j)]  # (f_score, i, j)
        came_from = {}
        g_score = { (start_i, start_j): 0 }
        f_score = { (start_i, start_j): math.sqrt((start_i - end_i)**2 + (start_j - end_j)**2) }
        
        while open_set:
            _, current_i, current_j = heapq.heappop(open_set)
            if (current_i, current_j) == (end_i, end_j):
                path = []
                current = (current_i, current_j)
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append((start_i, start_j))
                path.reverse()
                return self._smooth_path(path) if smooth else path
            
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = current_i + di, current_j + dj
                if not (0 <= ni < self.N and 0 <= nj < self.M):
                    continue
                
                # Check if the neighbor is outside the house (room is None)
                # Do not allow navigation to outside areas
                if self.grid[ni][nj].room is None:
                    continue
                
                # For diagonal movement, check if we would cut through a wall corner
                # This prevents paths from slipping through diagonal gaps in walls
                if di != 0 and dj != 0:  # Diagonal movement
                    # Check the two adjacent cells that the diagonal movement would pass through
                    adj1_i, adj1_j = current_i + di, current_j  # Horizontal adjacent
                    adj2_i, adj2_j = current_i, current_j + dj  # Vertical adjacent
                    
                    # Helper function to check if a cell is a blocking wall
                    def is_wall_blocked(ci, cj):
                        if not (0 <= ci < self.N and 0 <= cj < self.M):
                            return True  # Out of bounds treated as blocked
                        if self.grid[ci][cj].obstacle and self.grid[ci][cj].obstacle_class == 'wall':
                            return True
                        return False
                    
                    # If either adjacent cell is a wall, block diagonal movement to prevent corner cutting
                    if is_wall_blocked(adj1_i, adj1_j) or is_wall_blocked(adj2_i, adj2_j):
                        continue
                
                # Obstacle check during navigation: ignore specified furniture, but keep walls blocking
                is_blocked = False
                if self.grid[ni][nj].obstacle:
                    if self.grid[ni][nj].obstacle_class == 'wall':
                        is_blocked = True
                    elif self.grid[ni][nj].obstacle_class == 'furniture':
                        furniture_id = self.grid[ni][nj].obstacle_id
                        # Only block if furniture ID is not in ignore list
                        if furniture_id is None or furniture_id not in ignore_furniture_ids:
                            is_blocked = True
                
                if is_blocked:
                    continue
                
                # Calculate movement cost (using complete potential field)
                cost = 1.0 + (potential_k * self.grid[ni][nj].potential if potential else 0)
                tentative_g = g_score[(current_i, current_j)] + cost
                
                if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                    came_from[(ni, nj)] = (current_i, current_j)
                    g_score[(ni, nj)] = tentative_g
                    h = math.sqrt((ni - end_i)**2 + (nj - end_j)**2)
                    f_score[(ni, nj)] = tentative_g + h
                    heapq.heappush(open_set, (f_score[(ni, nj)], ni, nj))

        return None  # No path found
    

    def generate_stationary_trajectory(self, center_position, fluctuation_radius, start_time, duration, dt=1, unit='dm'):
        """
        Generate a trajectory sequence where a person fluctuates in a small range at a certain position
        
        Parameters:
            center_position (tuple): Center position coordinates (x, y)
            fluctuation_radius (float): Fluctuation radius (meters)
            start_time (str): Start time, format "YYYY-MM-DD HH:MM:SS"
            duration (float): Duration (seconds)
            dt (float): Sampling interval/time step, unit is seconds, default is 1
            unit (str): Coordinate unit, 'dm' for decimeters, 'm' for meters, default is 'dm'
            
        Returns:
            dict: Contains the following fields:
                - trajectory (list): Trajectory point list, each element is (x, y, t) tuple
                - end_time (str): End time
                - total_duration (float): Actual duration (seconds)
                - center_position (tuple): Center position
                - fluctuation_radius (float): Fluctuation radius
        """
        import datetime
        import random
        import math
        
        # Parse center position
        center_x, center_y = center_position
        
        # Unit conversion: if decimeters, convert to meters for calculation
        if unit == 'dm':
            center_x_m = center_x / 10
            center_y_m = center_y / 10
            fluctuation_radius_m = fluctuation_radius
        else:
            center_x_m = center_x
            center_y_m = center_y
            fluctuation_radius_m = fluctuation_radius
        
        # Parse start time
        current_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        
        trajectory = []
        elapsed_time = 0.0
        
        # Initial position is center position
        if unit == 'dm':
            trajectory.append((center_x, center_y, current_time.strftime("%Y-%m-%d %H:%M:%S")))
        else:
            trajectory.append((center_x_m, center_y_m, current_time.strftime("%Y-%m-%d %H:%M:%S")))
        
        # Generate fluctuation trajectory
        while elapsed_time < duration:
            elapsed_time += dt
            current_time += datetime.timedelta(seconds=dt)
            
            # Generate random angle and distance (within fluctuation radius)
            angle = random.uniform(0, 2 * math.pi)
            # Use square root distribution to make points more evenly distributed in circle
            distance = fluctuation_radius_m * math.sqrt(random.uniform(0, 1))
            
            # Calculate new position (meters)
            new_x_m = center_x_m + distance * math.cos(angle)
            new_y_m = center_y_m + distance * math.sin(angle)
            
            # Convert back to original unit
            if unit == 'dm':
                new_x = new_x_m * 10
                new_y = new_y_m * 10
            else:
                new_x = new_x_m
                new_y = new_y_m
            
            trajectory.append((new_x, new_y, current_time.strftime("%Y-%m-%d %H:%M:%S")))
        
        # Calculate end time
        end_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "trajectory": trajectory,
            "end_time": end_time,
            "total_duration": elapsed_time,
            "center_position": center_position,
            "fluctuation_radius": fluctuation_radius
        }

    def generate_navigation_trajectory(self, start, end, start_time, speed=(0.6, 0.8), dt=1, 
                                     potential=True, potential_k=1.0, potential_decay=1, potential_rand=True, 
                                     smooth=True, speed_start=0.5, speed_end=0.5, unit='dm'):
        """
        Comprehensive navigation function: given start point, end point and parameters, automatically calls navigation and trajectory simulation functions
        
        Parameters:
            start (tuple or int): Start point, format (x, y) or obstacle ID
            end (tuple or int): End point, format (x, y) or obstacle ID
            start_time (str): Start time, format "YYYY-MM-DD HH:MM:SS"
            speed (tuple or float): Walking speed range, default is (0.6, 0.8)
            dt (float): Sampling interval/time step, unit is seconds, default is 1
            potential (bool): Whether to use potential field, default is True
            potential_k (float): Potential field weight, default is 1.0
            potential_decay (float): Potential field decay rate, default is 1
            potential_rand (bool): Whether to randomly adjust potential field, default is True
            smooth (bool): Whether to smooth path, default is True
            speed_start (float): Starting speed, default is 0.5
            speed_end (float): Stopping speed, default is 0.5
            unit (str): Coordinate unit, 'dm' for decimeters, 'm' for meters, default is 'dm'
            
        Returns:
            dict: Contains the following fields:
                - success (bool): Whether trajectory was successfully generated
                - trajectory (list): Trajectory point list, each element is (x, y, t) tuple
                - end_time (str): End time
                - total_distance (float): Total distance (meters)
                - total_duration (float): Total duration (seconds)
                - path_points (int): Number of path points
                - validation_result (dict): Path validation result
                - error_message (str): Error message (if any)
        """
        import datetime
        
        result = {
            "success": False,
            "trajectory": [],
            "end_time": start_time,
            "total_distance": 0.0,
            "total_duration": 0.0,
            "path_points": 0,
            "validation_result": None,
            "error_message": ""
        }
        
        try:
            # First validate path
            validation_result = self.validate_navigation_path(start, end)
            result["validation_result"] = validation_result
            
            if not validation_result["is_valid"]:
                result["error_message"] = f"Path validation failed: {validation_result['error_message']}"
                return result
            
            # Find path
            path = self.find_path(
                start, end,
                potential=potential,
                potential_k=potential_k,
                potential_decay=potential_decay,
                potential_rand=potential_rand,
                smooth=smooth,
                allow_furniture_start=True,
                allow_furniture_end=True
            )
            
            if path is None:
                result["error_message"] = "Cannot find valid path"
                return result
            
            result["path_points"] = len(path)
            
            # Generate human trajectory
            trajectory = self.generate_human_trajectory(
                path=path,
                speed=speed,
                dt=dt,
                start_time=start_time,
                speed_start=speed_start,
                speed_end=speed_end,
                unit=unit
            )
            
            if not trajectory:
                result["error_message"] = "Trajectory generation failed"
                return result
            
            # Calculate total distance (convert to meters)
            total_distance = 0.0
            for i in range(1, len(trajectory)):
                x1, y1 = trajectory[i-1][0], trajectory[i-1][1]
                x2, y2 = trajectory[i][0], trajectory[i][1]
                
                # Convert based on unit
                if unit == 'dm':
                    x1, y1, x2, y2 = x1/10, y1/10, x2/10, y2/10
                
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_distance += distance
            
            # Calculate total duration
            start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.datetime.strptime(trajectory[-1][2], "%Y-%m-%d %H:%M:%S")
            total_duration = (end_dt - start_dt).total_seconds()
            
            # Update result
            result["success"] = True
            result["trajectory"] = trajectory
            result["end_time"] = trajectory[-1][2]
            result["total_distance"] = total_distance
            result["total_duration"] = total_duration
            result["error_message"] = "Trajectory generation successful"
            
        except Exception as e:
            result["error_message"] = f"Error occurred during trajectory generation: {str(e)}"
        
        return result

    def generate_human_trajectory(self, path, speed=(0.6, 0.8), dt=1, start_time="2025-01-01 00:00:00", speed_start=0.5, speed_end=0.5, unit='dm'):
        """
        Generate human-like walking trajectory from path coordinate list.

        Parameters:
            path (list): (x, y) trajectory coordinate list (float coordinates, already smoothed).
            speed (tuple or float): Walking speed range, if tuple like (0.9, 1.1), randomly sample each time step; if constant, use that speed.
            dt (float): Sampling interval/time step, unit is seconds, default is 0.1.
            start_time (str): Start time, format "YYYY-MM-DD HH:MM:SS".
            speed_start (float): Starting speed, default is 0.0.
            speed_end (float): Stopping speed, default is 0.0.

        Returns:
            list: Each element is (x, y, t) tuple representing position and time.
        """

        if not path:
            return []
        if len(path) == 1:
            return [(path[0][0], path[0][1], start_time)]
        
        path = [(x, y) for x, y in path]
        
        
        if unit == 'dm':
            # Adjust scale, decimeters->meters
            for i in range(len(path)):
                path[i] = (path[i][0] / 10, path[i][1] / 10)


        # Calculate cumulative distance between path points
        cum_dists = [0.0]
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            dist = math.hypot(dx, dy)
            cum_dists.append(cum_dists[-1] + dist)
        total_distance = cum_dists[-1]

        # Pre-calculate turning point information (excluding first and last points)
        turning_vertices = []  # Each element is (cumulative distance, turning angle[radians])
        for i in range(1, len(path)-1):
            v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            norm1 = math.hypot(v1[0], v1[1])
            norm2 = math.hypot(v2[0], v2[1])
            if norm1 == 0 or norm2 == 0:
                angle = 0
            else:
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                cos_angle = max(-1, min(1, dot/(norm1*norm2)))
                angle = math.acos(cos_angle)
            turning_vertices.append((cum_dists[i], angle))

        # Acceleration and deceleration parameters
        acceleration_distance = 1.0  # Starting acceleration distance (meters)
        deceleration_distance = 1.0  # Stopping deceleration distance (meters)

        # Turning deceleration parameters
        slowdown_window = 1.0   # Influence range (meters)
        max_slowdown_factor = 0.5  # Minimum speed ratio

        # Convert start time to datetime object
        current_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

        trajectory = []
        current_distance = 0.0
        trajectory.append((path[0][0], path[0][1], current_time.strftime("%Y-%m-%d %H:%M:%S")))

        # Initialize current speed to starting speed
        current_speed = speed_start
        # Calculate target speed (when speed is a range, take middle value, otherwise take speed directly)
        if isinstance(speed, (list, tuple)):
            target_speed_nominal = (speed[0] + speed[1]) / 2.0
        else:
            target_speed_nominal = speed
        # Calculate acceleration (assuming reaching target speed within acceleration_distance)
        acceleration = (target_speed_nominal - speed_start) / acceleration_distance if acceleration_distance > 0 else 0

        while current_distance < total_distance:
            # Randomly get target speed fluctuation
            if isinstance(speed, (list, tuple)):
                target_speed = random.uniform(speed[0], speed[1])
            else:
                target_speed = speed

            # Starting phase: gradually accelerate
            if current_distance < acceleration_distance:
                current_speed = min(target_speed, current_speed + acceleration * dt)
            else:
                current_speed = target_speed

            # Stopping phase: decelerate to speed_end within remaining deceleration_distance
            remaining_distance = total_distance - current_distance
            if remaining_distance < deceleration_distance:
                current_speed = current_speed + (speed_end - current_speed) * ((deceleration_distance - remaining_distance) / deceleration_distance)

            # Turning deceleration processing
            slowdown_multiplier = 1.0
            for (vertex_dist, turn_angle) in turning_vertices:
                dist_diff = abs(current_distance - vertex_dist)
                if dist_diff < slowdown_window:
                    effective_angle = min(turn_angle, math.pi/2)
                    local_multiplier = max_slowdown_factor + (1 - max_slowdown_factor) * (dist_diff / slowdown_window)
                    turn_severity = effective_angle / (math.pi/2)
                    candidate = 1 - turn_severity * (1 - local_multiplier)
                    slowdown_multiplier = min(slowdown_multiplier, candidate)

            effective_speed = current_speed * slowdown_multiplier
            effective_speed = max(effective_speed, 0)

            # Calculate step distance
            step_distance = effective_speed * dt
            current_distance += step_distance
            if current_distance > total_distance:
                current_distance = total_distance

            # Path interpolation
            seg_index = bisect.bisect_right(cum_dists, current_distance) - 1
            if seg_index >= len(path) - 1:
                seg_index = len(path) - 2
            seg_start = path[seg_index]
            seg_end = path[seg_index+1]
            seg_length = cum_dists[seg_index+1] - cum_dists[seg_index]
            ratio = 0 if seg_length == 0 else (current_distance - cum_dists[seg_index]) / seg_length
            x = seg_start[0] + (seg_end[0] - seg_start[0]) * ratio
            y = seg_start[1] + (seg_end[1] - seg_start[1]) * ratio

            current_time += datetime.timedelta(seconds=dt)
            trajectory.append((x, y, current_time.strftime("%Y-%m-%d %H:%M:%S")))
        
        if unit == 'dm':
            # Adjust scale, meters->decimeters
            for i in range(len(trajectory)):
                trajectory[i] = (trajectory[i][0] * 10, trajectory[i][1] * 10, trajectory[i][2])

        return trajectory


    def _generate_potential(self, sign='positive', target='all', strength=1, decay=1.0, rand=False):
        """
        Generate potential field for obstacles.

        Parameters:
            sign (str): 'positive' (repulsive) or 'negative' (attractive), default is 'positive'.
            target (str or list): 'all', obstacle name (str) or ID/name list, default is 'all'.
            strength (float): Obstacle potential field strength, default is 1.0 (maximum potential field value).
            decay (float): Distance decay rate, default is 1.0.
            rand (bool): If True, randomly adjust potential field values, default is False.
        """
        obstacle_map = np.zeros((self.N, self.M), dtype=bool)
        
        if target == 'all':
            for i in range(self.N):
                for j in range(self.M):
                    if self.grid[i][j].obstacle:
                        obstacle_map[i, j] = True
        elif isinstance(target, str):
            for i in range(self.N):
                for j in range(self.M):
                    if self.grid[i][j].obstacle_name == target:
                        obstacle_map[i, j] = True
        elif isinstance(target, list):
            for item in target:
                for i in range(self.N):
                    for j in range(self.M):
                        if ((isinstance(item, int) and self.grid[i][j].obstacle_id == item) or
                            (isinstance(item, str) and self.grid[i][j].obstacle_name == item)):
                            obstacle_map[i, j] = True

        distance = distance_transform_edt(~obstacle_map)
        potential = 1 / (distance + 1e-6) ** decay
        potential = np.minimum(potential, 1)

        # Randomly adjust potential field values in each zone
        if rand:
            n_size = 5
            rate1 = np.random.uniform(1, 5)
            rate2 = np.random.uniform(0, 0.5)
            for i in range(0, self.N, n_size):
                for j in range(0, self.M, n_size):
                    for ii in range(i, i+n_size):
                        for jj in range(j, j+n_size):
                            if ii < self.N and jj < self.M:
                                potential[ii, jj] *= rate1
                                potential[ii, jj] += rate2
                    rate1 = np.random.uniform(1, 6)
                    rate2 = np.random.uniform(0, 0.5)
        # Normalize
        max_potential = np.max(potential)
        min_potential = np.min(potential)
        potential = (potential - min_potential) / (max_potential - min_potential) + 0.001
        # Smooth
        potential = gaussian_filter(potential, sigma=1, radius=1)
        
        if sign == 'negative':
            potential = -potential
        
        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j].obstacle:
                    self.grid[i][j].potential = strength
                elif not self.grid[i][j].room:
                    self.grid[i][j].potential = 0
                else:
                    self.grid[i][j].potential = strength * potential[i, j]
                    


                
        
    def _to_grid_coords(self, point):
        """Convert point (tuple or obstacle ID) to grid indices (i, j)."""
        if isinstance(point, tuple):
            x, y = point
        else:  # Assume it's obstacle ID
            obs = self.get_obstacle_info(point)
            if not obs:
                raise ValueError(f"Obstacle ID {point} not found")
            pos = obs['position']
            x = pos['x'] + pos['width'] / 2
            y = pos['y'] + pos['height'] / 2
        return int(x), int(y)

    def _smooth_path(self, path, steps=2):
        """Quadratic Bezier curve interpolation to smooth path."""
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            x3, y3 = path[i + 1]
            for t in np.linspace(0, 1, steps + 2)[1:-1]:
                x = (1 - t)**2 * x1 + 2 * (1 - t) * t * x2 + t**2 * x3
                y = (1 - t)**2 * y1 + 2 * (1 - t) * t * y2 + t**2 * y3
                smoothed.append((x, y))
        smoothed.append(path[-1])
        return smoothed

    def _build_grid(self):
        """Build grid matrix based on YAML layout."""

        for room in self.rooms:
            pos = room.get('position', {})
            x, y = pos.get('x'), pos.get('y')
            width, height = pos.get('width'), pos.get('height')
            for i in range(x, x+width):
                for j in range(y, y + height):
                    self.grid[i][j].room = room['name']


        for room in self.rooms:
            pos = room.get('position', {})
            x, y = pos.get('x'), pos.get('y')
            width, height = pos.get('width'), pos.get('height')
            # Assign room name to grid cells   
            for i in range(x, x + width):
                for j in range(y, y + height):
                    self.grid[i][j].room = room['name']
            
            # Mark walls        
            walls = room.get('walls', {})
            
            # Define default walls if not specified
            for wall_key, default_p1, default_p2 in [
                ('top', [x, y + height], [x + width, y + height]),
                ('right', [x + width, y], [x + width, y + height]),
                ('bottom', [x, y], [x + width, y]),
                ('left', [x, y], [x, y + height])
            ]:
                wall = walls.get(wall_key, True)
                if wall:
                    p1, p2 = (wall['points'] if isinstance(wall, dict) and 'points' in wall else (default_p1, default_p2))
                    cells = self._get_line_cells(p1, p2)
                    for i, j in cells:
                        self.grid[i][j].obstacle = True
                        self.grid[i][j].obstacle_name = 'wall'
                        self.grid[i][j].obstacle_id = None
                        self.grid[i][j].obstacle_class = 'wall'

            # Extra walls
            extra_walls = walls.get('extra', [])
            for ew in extra_walls:
                p1, p2 = ew.get('points')
                cells = self._get_line_cells(p1, p2)
                for i, j in cells:
                    self.grid[i][j].obstacle = True
                    self.grid[i][j].obstacle_name = 'wall'
                    self.grid[i][j].obstacle_id = None
                    self.grid[i][j].obstacle_class = 'wall'


            # Mark furniture as obstacles
            for furn in room.get('furniture', []):
                pos = furn.get('position', {})
                obstacle_id = furn.get('id')
                fx, fy = pos.get('x'), pos.get('y')
                fwidth, fheight = pos.get('width'), pos.get('height')
                for i in range(int(fx), int(fx+fwidth)):
                    for j in range(int(fy), int(fy+fheight)):
                        self.grid[i][j].obstacle = True
                        self.grid[i][j].obstacle_name = furn['name']
                        self.grid[i][j].obstacle_id = obstacle_id
                        self.grid[i][j].obstacle_class = 'furniture'
                self.obstacles.append({
                    'id': obstacle_id,
                    'name': furn['name'],
                    'position': pos,
                    'room': room['name']
                })

        for room in self.rooms:           
            # Mark doors as non-obstacles, door frames as obstacles
            doors = room.get('doors', [])
            for door in doors:
                p1, p2 = door.get('points')[0:2]
                cells = self._get_line_cells(p1, p2)
                # Set door frame as wall obstacle (so navigation will avoid it)
                for i, j in cells:
                    self.grid[i][j].obstacle = True
                    self.grid[i][j].obstacle_name = 'door_frame'
                    self.grid[i][j].obstacle_id = None
                    self.grid[i][j].obstacle_class = 'wall'  # Treat door frame as wall for navigation

                # Shrink door opening (middle 40% is passable)
                p1, p2 = np.array(p1), np.array(p2)
                p1, p2 = p1+0.3*(p2-p1), p2-0.3*(p2-p1)
                cells = self._get_line_cells(p1, p2)
                for i, j in cells:
                    self.grid[i][j].obstacle = False
                    self.grid[i][j].obstacle_name = None
                    self.grid[i][j].obstacle_id = None
                    self.grid[i][j].obstacle_class = None

    def _get_line_cells(self, p1, p2):
        """Rasterize line segment to grid cells using stepping method."""
        i1, j1 = p1
        i2, j2 = p2
        i1, j1, i2, j2 = int(round(i1)), int(round(j1)), int(round(i2)), int(round(j2))
        dx, dy = abs(j2 - j1), abs(i2 - i1)
        steps = max(int(dx), int(dy)) + 1
        cells = set()
        for t in np.linspace(0, 1, steps):
            # Use round() instead of int() to avoid floating point precision issues
            # e.g., int(0.58 * 50) = 28 instead of 29 due to 0.58*50 = 28.999999999999996
            j = int(round(j1 + t * (j2 - j1)))
            i = int(round(i1 + t * (i2 - i1)))
            if 0 <= i < self.N and 0 <= j < self.M:
                cells.add((i, j))
        return list(cells)
    



    def get_obstacle_info(self, obstacle_id):
        """
        Get obstacle information by ID.

        Parameters:
            obstacle_id (int): Obstacle ID.
        Returns:
            dict: Obstacle information {'id', 'name', 'position', 'room'} or None if not found.
        """
        return next((obs for obs in self.obstacles if obs['id'] == obstacle_id), None)
    

    def draw_trajectory(self, ax, trajectory, cmap='Reds', gap=0.5, linewidth=4, arrow_size=20, node_size=30):
        """
        Draw smooth and beautiful trajectory
        
        Parameters:
            ax: matplotlib axis object
            trajectory: Trajectory point list, each element is (x, y) or (x, y, t) format
            cmap: Color map name, default is 'Reds'
            gap: Gap size between line segments, default 0.5
            linewidth: Thickness of trajectory line, default is 3.5
            arrow_size: Arrow size, default is 10
            node_size: Node (point) size, default is 15
        """
        # Extract coordinates
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        
        # Construct line segments
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create new line segments with gaps
        new_segments = []
        for seg in segments:
            start, end = seg[0], seg[1]
            vec = end - start
            length = np.linalg.norm(vec)
            if length > 2 * gap:
                new_start = start + (vec / length) * gap
                new_end = end - (vec / length) * gap
            else:
                new_start, new_end = start, end
            new_segments.append([new_start, new_end])
        new_segments = np.array(new_segments)
        
        # Create gradient colors
        base_cmap = plt.get_cmap(cmap)
        colors = base_cmap(np.linspace(0.3, 0.8, len(new_segments)))
        custom_cmap = ListedColormap(colors)
        
        # Draw trajectory line segments
        norm = Normalize(0, len(new_segments))
        lc = LineCollection(new_segments, cmap=custom_cmap, norm=norm, linewidth=linewidth)
        lc.set_array(np.arange(len(new_segments)))
        ax.add_collection(lc)
        
        # Draw nodes
        ax.scatter(xs, ys, s=node_size, color='white', edgecolors='black', zorder=5)
        
        # Add arrows
        for i in range(0, len(new_segments)):
            start_seg = new_segments[i][0]
            end_seg = new_segments[i][1]
            # Adjust arrow position
            start_seg = start_seg + 0.5 * (end_seg - start_seg)
            end_seg = end_seg + 0.5 * (end_seg - start_seg)
            # Draw arrow
            arrow = FancyArrowPatch(
                start_seg, end_seg,
                arrowstyle='-|>',
                mutation_scale=arrow_size,
                color=colors[i],
                lw=0.8
            )
            ax.add_patch(arrow)
        
        return ax

    def validate_navigation_path(self, start, end):
        """
        Validate whether the given start and end points can be navigated
        
        Parameters:
            start (tuple or int): Start point, format (x, y) or obstacle ID
            end (tuple or int): End point, format (x, y) or obstacle ID
            
        Returns:
            dict: Validation result, containing the following fields:
                - is_valid (bool): Whether navigation is possible
                - error_type (str): Error type, possible values:
                    - None: No error
                    - "start_out_of_bounds": Start point out of bounds
                    - "end_out_of_bounds": End point out of bounds
                    - "start_outside_house": Start point outside house
                    - "end_outside_house": End point outside house
                    - "start_on_wall": Start point on wall
                    - "end_on_wall": End point on wall
                    - "no_path_found": Path not connected
                - error_message (str): Detailed error message
                - path_found (bool): Whether path was found (even if start/end on furniture)
        """
        result = {
            "is_valid": False,
            "error_type": None,
            "error_message": "",
            "path_found": False
        }
        
        try:
            # Convert coordinates
            start_i, start_j = self._to_grid_coords(start)
            end_i, end_j = self._to_grid_coords(end)
            
            # Check start point bounds
            if not (0 <= start_i < self.N and 0 <= start_j < self.M):
                result["error_type"] = "start_out_of_bounds"
                result["error_message"] = f"Start coordinates ({start_i}, {start_j}) out of grid bounds ({self.N}, {self.M})"
                return result
            
            # Check end point bounds
            if not (0 <= end_i < self.N and 0 <= end_j < self.M):
                result["error_type"] = "end_out_of_bounds"
                result["error_message"] = f"End coordinates ({end_i}, {end_j}) out of grid bounds ({self.N}, {self.M})"
                return result
            
            # Check if start point is outside the house
            if self.grid[start_i][start_j].room is None:
                result["error_type"] = "start_outside_house"
                result["error_message"] = f"Start coordinates ({start_i}, {start_j}) are outside the house, cannot navigate"
                return result
            
            # Check if end point is outside the house
            if self.grid[end_i][end_j].room is None:
                result["error_type"] = "end_outside_house"
                result["error_message"] = f"End coordinates ({end_i}, {end_j}) are outside the house, cannot navigate"
                return result
            
            # Check if start point is on wall
            if (self.grid[start_i][start_j].obstacle and 
                self.grid[start_i][start_j].obstacle_class == 'wall'):
                result["error_type"] = "start_on_wall"
                result["error_message"] = f"Start coordinates ({start_i}, {start_j}) are on a wall, cannot navigate"
                return result
            
            # Check if end point is on wall
            if (self.grid[end_i][end_j].obstacle and 
                self.grid[end_i][end_j].obstacle_class == 'wall'):
                result["error_type"] = "end_on_wall"
                result["error_message"] = f"End coordinates ({end_i}, {end_j}) are on a wall, cannot navigate"
                return result
            
            # Try to find path (allowing furniture start/end points)
            try:
                path = self.find_path(
                    start, end, 
                    potential=False, 
                    smooth=False,
                    allow_furniture_start=True,
                    allow_furniture_end=True
                )
                
                if path is not None:
                    result["is_valid"] = True
                    result["path_found"] = True
                    result["error_message"] = "Navigation path validation successful"
                else:
                    result["error_type"] = "no_path_found"
                    result["error_message"] = "Cannot find valid path from start to end"
                    
            except Exception as path_error:
                result["error_type"] = "no_path_found"
                result["error_message"] = f"Path search failed: {str(path_error)}"
                
        except ValueError as coord_error:
            # Coordinate conversion error (e.g., obstacle ID does not exist)
            result["error_type"] = "invalid_coordinates"
            result["error_message"] = f"Coordinate conversion failed: {str(coord_error)}"
        except Exception as general_error:
            result["error_type"] = "unknown_error"
            result["error_message"] = f"Unknown error: {str(general_error)}"
        
        return result
