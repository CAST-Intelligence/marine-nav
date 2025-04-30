"""
Temporal grid representation for the marine environment with time-varying data.
"""
from typing import Tuple, List, Dict, Optional, Any, Set, Union
import numpy as np
from .spatio_temporal_field import SpatioTemporalField


class TemporalNavigationGrid:
    """
    A grid-based representation of the marine environment with time dimension.
    
    This class manages:
    - Grid coordinates and cell size
    - Obstacles and restricted areas
    - Integration with time-varying current/weather data
    - Time-dependent cost calculations between points
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        time_steps: int,
        cell_size: float = 1.0,
        x_origin: float = 0.0,
        y_origin: float = 0.0,
        time_step_duration: float = 3600.0  # Default 1 hour in seconds
    ):
        """
        Initialize a temporal navigation grid.
        
        Args:
            grid_size: The (width, height) size of the grid in cells
            time_steps: Number of discrete time steps to model
            cell_size: The size of each cell in the grid (in meters)
            x_origin: The x-coordinate of the grid origin
            y_origin: The y-coordinate of the grid origin
            time_step_duration: Duration of each time step in seconds
        """
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.cell_size = cell_size
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.time_step_duration = time_step_duration
        
        # Initialize grid arrays
        # Note: numpy arrays are (rows, columns) = (y, x) order
        self.obstacle_grid = np.zeros((grid_size[1], grid_size[0]), dtype=bool)
        self.cost_grid = np.ones((grid_size[1], grid_size[0]))  # Base cost is 1.0
        
        # Spatio-temporal field for currents and weather
        self.environment_field = None
        
        # Create coordinate meshgrid
        x = np.linspace(x_origin, x_origin + cell_size * grid_size[0], grid_size[0])
        y = np.linspace(y_origin, y_origin + cell_size * grid_size[1], grid_size[1])
        self.X, self.Y = np.meshgrid(x, y)
    
    def set_environment_field(self, field: SpatioTemporalField):
        """
        Set the environment field (currents and weather) for this grid.
        
        Args:
            field: A SpatioTemporalField object representing time-varying data
        """
        self.environment_field = field
    
    def add_obstacle(self, x: int, y: int):
        """
        Mark a cell as an obstacle.
        
        Args:
            x: x-coordinate in grid cells
            y: y-coordinate in grid cells
        """
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            self.obstacle_grid[y, x] = True
        else:
            raise ValueError(f"Cell ({x}, {y}) is outside the grid")
    
    def add_obstacle_region(self, x_min: int, y_min: int, x_max: int, y_max: int):
        """
        Mark a rectangular region as obstacles.
        
        Args:
            x_min: Minimum x-coordinate
            y_min: Minimum y-coordinate
            x_max: Maximum x-coordinate
            y_max: Maximum y-coordinate
        """
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.grid_size[0], x_max)
        y_max = min(self.grid_size[1], y_max)
        
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                self.obstacle_grid[y, x] = True
    
    def is_obstacle(self, x: int, y: int) -> bool:
        """
        Check if a cell contains an obstacle.
        
        Args:
            x: x-coordinate in grid cells
            y: y-coordinate in grid cells
            
        Returns:
            True if the cell is an obstacle, False otherwise
        """
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            return self.obstacle_grid[y, x]
        # Consider cells outside the grid as obstacles
        return True
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get the valid neighboring cells (including diagonals).
        
        Args:
            x: x-coordinate in grid cells
            y: y-coordinate in grid cells
            
        Returns:
            List of (x, y) tuples for valid neighboring cells
        """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if not self.is_obstacle(nx, ny):
                        neighbors.append((nx, ny))
        
        return neighbors
    
    def cell_to_coords(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert grid cell coordinates to world coordinates.
        
        Args:
            x: x-coordinate in grid cells
            y: y-coordinate in grid cells
            
        Returns:
            Tuple of (x, y) world coordinates
        """
        world_x = self.x_origin + (x + 0.5) * self.cell_size
        world_y = self.y_origin + (y + 0.5) * self.cell_size
        return world_x, world_y
    
    def coords_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid cell coordinates.
        
        Args:
            x: x-coordinate in world
            y: y-coordinate in world
            
        Returns:
            Tuple of (x, y) grid cell coordinates
        """
        cell_x = int((x - self.x_origin) / self.cell_size)
        cell_y = int((y - self.y_origin) / self.cell_size)
        
        # Clamp to grid bounds
        cell_x = max(0, min(cell_x, self.grid_size[0] - 1))
        cell_y = max(0, min(cell_y, self.grid_size[1] - 1))
        
        return cell_x, cell_y
    
    def calculate_travel_cost_time(
        self, 
        start_x: int, 
        start_y: int, 
        end_x: int, 
        end_y: int, 
        time_index: Union[int, float],
        usv_cruise_speed: float = 1.0,
        usv_max_speed: float = 1.5,
        optimization_mode: str = 'fastest_time',
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Calculate the travel cost between two adjacent cells, with proper physics.
        
        This method implements correct vector math to determine:
        1. The required USV velocity through water to achieve desired ground velocity
        2. The actual achievable ground velocity and time
        3. The energy consumption based on a cubic power model
        
        Args:
            start_x, start_y: Starting coordinates in grid cells
            end_x, end_y: Ending coordinates in grid cells
            time_index: The time index for the calculation
            usv_cruise_speed: The cruise speed of the USV in m/s
            usv_max_speed: The maximum speed of the USV in m/s
            optimization_mode: 'fastest_time' or 'lowest_energy'
            return_details: If True, return a dict with detailed information
            
        Returns:
            If return_details is False: 
                The primary cost (time or energy) for the edge
            If return_details is True: 
                Dictionary with 'cost', 'energy', 'time', 'usv_vel', 'ground_vel'
        """
        # Convert to world coordinates
        start_world_x, start_world_y = self.cell_to_coords(start_x, start_y)
        end_world_x, end_world_y = self.cell_to_coords(end_x, end_y)
        
        # Calculate Euclidean distance
        dx = end_world_x - start_world_x
        dy = end_world_y - start_world_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate desired ground direction unit vector
        movement_vector = np.array([dx, dy])
        if np.linalg.norm(movement_vector) > 0:
            movement_vector = movement_vector / np.linalg.norm(movement_vector)
        
        # If there's no environment field data, return the distance/time as cost
        if self.environment_field is None:
            travel_time = distance / usv_cruise_speed
            energy = (usv_cruise_speed / usv_max_speed)**3 * travel_time
            
            if return_details:
                return {
                    'cost': travel_time if optimization_mode == 'fastest_time' else energy,
                    'energy': energy,
                    'time': travel_time,
                    'time_idx_end': time_index + travel_time / self.time_step_duration,
                    'usv_vel': (usv_cruise_speed * movement_vector[0], 
                               usv_cruise_speed * movement_vector[1]),
                    'ground_vel': (usv_cruise_speed * movement_vector[0], 
                                  usv_cruise_speed * movement_vector[1]),
                }
            else:
                return travel_time if optimization_mode == 'fastest_time' else energy
        
        # Get the environment data at midpoint and current time
        mid_x = (start_world_x + end_world_x) / 2
        mid_y = (start_world_y + end_world_y) / 2
        
        u, v, w = self.environment_field.get_vector_at_position_time(mid_x, mid_y, time_index)
        current_vector = np.array([u, v])
        current_magnitude = np.linalg.norm(current_vector)
        
        # CORRECT PHYSICS - Different for each optimization mode
        if optimization_mode == 'fastest_time':
            # For fastest time: Given the current and desired ground direction,
            # solve for USV water velocity to get ground velocity along desired direction
            # at desired speed (if possible)
            
            # Calculate required USV water velocity vector
            # (This is the key correction from the original implementation)
            # Desired ground velocity = usv_cruise_speed * movement_vector
            desired_ground_velocity = usv_cruise_speed * movement_vector
            
            # Required water velocity = desired ground velocity - current
            required_usv_water_velocity = desired_ground_velocity - current_vector
            required_usv_water_speed = np.linalg.norm(required_usv_water_velocity)
            
            # If required speed exceeds max_speed, reduce proportionally
            if required_usv_water_speed > usv_max_speed:
                scale_factor = usv_max_speed / required_usv_water_speed
                usv_water_velocity = required_usv_water_velocity * scale_factor
                usv_water_speed = usv_max_speed
            else:
                usv_water_velocity = required_usv_water_velocity
                usv_water_speed = required_usv_water_speed
            
            # Calculate actual ground velocity and speed
            ground_velocity = usv_water_velocity + current_vector
            ground_speed = np.linalg.norm(ground_velocity)
            
            # Convert to travel time
            travel_time = distance / ground_speed if ground_speed > 0 else float('inf')
            
            # Calculate energy consumption (cubic model)
            energy = (usv_water_speed / usv_max_speed)**3 * travel_time
            
            # Primary cost for fastest time mode is travel time
            primary_cost = travel_time
            
        elif optimization_mode == 'lowest_energy':
            # For lowest energy: Try to minimize power while maintaining progress
            
            # First check if pure drift is possible
            if current_magnitude > 0:
                # Dot product to see if current helps in desired direction
                if np.dot(current_vector / current_magnitude, movement_vector) > 0.1:
                    # Possible drift - no motor power
                    drift_time = distance / current_magnitude
                    
                    if return_details:
                        return {
                            'cost': 0,  # Zero energy cost for drifting
                            'energy': 0,
                            'time': drift_time,
                            'time_idx_end': time_index + drift_time / self.time_step_duration,
                            'usv_vel': (0, 0),  # No USV water velocity
                            'ground_vel': current_vector,
                            'drift': True
                        }
                    else:
                        return 0  # Zero energy cost
            
            # If drift isn't beneficial, use minimum power to make progress
            # Target cruise speed but adjust for current
            desired_ground_velocity = usv_cruise_speed * movement_vector
            
            # Required water velocity = desired ground velocity - current
            required_usv_water_velocity = desired_ground_velocity - current_vector
            required_usv_water_speed = np.linalg.norm(required_usv_water_velocity)
            
            # If required speed exceeds max_speed, reduce proportionally
            if required_usv_water_speed > usv_max_speed:
                scale_factor = usv_max_speed / required_usv_water_speed
                usv_water_velocity = required_usv_water_velocity * scale_factor
                usv_water_speed = usv_max_speed
            else:
                usv_water_velocity = required_usv_water_velocity
                usv_water_speed = required_usv_water_speed
            
            # Calculate actual ground velocity and speed
            ground_velocity = usv_water_velocity + current_vector
            ground_speed = np.linalg.norm(ground_velocity)
            
            # Progress is the component of ground velocity in the desired direction
            progress = np.dot(ground_velocity, movement_vector)
            
            # If no progress, this edge is invalid
            if progress <= 0:
                if return_details:
                    return {
                        'cost': float('inf'),
                        'energy': float('inf'),
                        'time': float('inf'),
                        'time_idx_end': time_index,
                        'usv_vel': (0, 0),
                        'ground_vel': (0, 0)
                    }
                else:
                    return float('inf')
            
            # Convert to travel time
            travel_time = distance / ground_speed if ground_speed > 0 else float('inf')
            
            # Apply weather factor to energy calculation
            energy_factor = 1.0 + 0.5 * w  # Simple linear scaling with weather factor
            
            # Calculate energy consumption (cubic model with weather factor)
            energy = (usv_water_speed / usv_max_speed)**3 * travel_time * energy_factor
            
            # Primary cost for energy mode is energy consumption
            primary_cost = energy
        
        else:
            raise ValueError(f"Invalid optimization mode: {optimization_mode}")
        
        # Calculate ending time index
        time_idx_end = time_index + travel_time / self.time_step_duration
        
        if return_details:
            return {
                'cost': primary_cost,
                'energy': energy,
                'time': travel_time,
                'time_idx_end': time_idx_end,
                'usv_vel': tuple(usv_water_velocity),
                'ground_vel': tuple(ground_velocity)
            }
        else:
            return primary_cost
    
    def is_path_collision_free(
        self, 
        start_x: int, 
        start_y: int, 
        end_x: int, 
        end_y: int
    ) -> bool:
        """
        Check if a straight path between cells is collision-free.
        
        Args:
            start_x, start_y: Starting coordinates in grid cells
            end_x, end_y: Ending coordinates in grid cells
            
        Returns:
            True if the path is collision-free, False otherwise
        """
        # Get intermediate points using Bresenham's line algorithm
        points = self._line_points(start_x, start_y, end_x, end_y)
        
        # Check if any point is an obstacle
        for x, y in points:
            if self.is_obstacle(x, y):
                return False
        
        return True
    
    def _line_points(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Generate a list of cell coordinates along a line using Bresenham's algorithm.
        
        Args:
            x0, y0: Starting coordinates
            x1, y1: Ending coordinates
            
        Returns:
            List of (x, y) cell coordinates along the line
        """
        points = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                if x0 == x1:
                    break
                err -= dy
                x0 += sx
            if e2 < dx:
                if y0 == y1:
                    break
                err += dx
                y0 += sy
        
        return points