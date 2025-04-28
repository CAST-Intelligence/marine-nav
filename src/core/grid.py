"""
Grid representation for the marine environment.
"""
from typing import Tuple, List, Dict, Optional, Any, Set
import numpy as np
from .vector_field import VectorField


class NavigationGrid:
    """
    A grid-based representation of the marine environment.
    
    This class manages:
    - Grid coordinates and cell size
    - Obstacles and restricted areas
    - Integration with current vector field
    - Distance/cost calculations between points
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        cell_size: float = 1.0,
        x_origin: float = 0.0,
        y_origin: float = 0.0
    ):
        """
        Initialize a navigation grid.
        
        Args:
            grid_size: The (width, height) size of the grid in cells
            cell_size: The size of each cell in the grid (in meters)
            x_origin: The x-coordinate of the grid origin
            y_origin: The y-coordinate of the grid origin
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.x_origin = x_origin
        self.y_origin = y_origin
        
        # Initialize grid arrays
        self.obstacle_grid = np.zeros(grid_size, dtype=bool)
        self.cost_grid = np.ones(grid_size)  # Base cost is 1.0
        
        # Vector field for currents (initially None)
        self.current_field = None
        
        # Create coordinate meshgrid
        x = np.linspace(x_origin, x_origin + cell_size * grid_size[0], grid_size[0])
        y = np.linspace(y_origin, y_origin + cell_size * grid_size[1], grid_size[1])
        self.X, self.Y = np.meshgrid(x, y)
    
    def set_current_field(self, current_field: VectorField):
        """
        Set the current vector field for this grid.
        
        Args:
            current_field: A VectorField object representing currents
        """
        self.current_field = current_field
    
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
    
    def calculate_travel_cost(
        self, 
        start_x: int, 
        start_y: int, 
        end_x: int, 
        end_y: int, 
        usv_speed: float = 1.0
    ) -> float:
        """
        Calculate the travel cost between two adjacent cells, accounting for currents.
        
        Args:
            start_x: Starting x-coordinate in grid cells
            start_y: Starting y-coordinate in grid cells
            end_x: Ending x-coordinate in grid cells
            end_y: Ending y-coordinate in grid cells
            usv_speed: The speed of the USV in m/s
            
        Returns:
            The cost to travel between the cells
        """
        # Convert to world coordinates
        start_world_x, start_world_y = self.cell_to_coords(start_x, start_y)
        end_world_x, end_world_y = self.cell_to_coords(end_x, end_y)
        
        # Calculate Euclidean distance
        dx = end_world_x - start_world_x
        dy = end_world_y - start_world_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # If there's no current field, return the distance as cost
        if self.current_field is None:
            return distance
        
        # Get the current vector at the midpoint
        mid_x = (start_world_x + end_world_x) / 2
        mid_y = (start_world_y + end_world_y) / 2
        
        current_u, current_v = self.current_field.get_vector_at_position(mid_x, mid_y)
        current_speed = np.sqrt(current_u**2 + current_v**2)
        
        # Calculate the dot product to see if the current is helping or hindering
        movement_vector = np.array([dx, dy])
        movement_vector = movement_vector / np.linalg.norm(movement_vector)
        current_vector = np.array([current_u, current_v])
        
        # If current is zero, return the distance
        if current_speed < 1e-6:
            return distance
            
        current_vector = current_vector / current_speed
        
        # Dot product gives the cosine of the angle between movement and current
        dot_product = np.dot(movement_vector, current_vector)
        
        # Calculate effective speed (USV speed +/- current effect)
        # Full help when dot product is 1, full hindrance when -1
        effective_speed = usv_speed + current_speed * dot_product
        
        # Ensure effective speed is positive (minimum slow crawl)
        effective_speed = max(effective_speed, 0.1 * usv_speed)
        
        # Cost is proportional to time = distance / speed
        cost = distance / effective_speed
        
        return cost
    
    def build_distance_matrix(self, usv_speed: float = 1.0) -> np.ndarray:
        """
        Build a complete distance/cost matrix between all navigable cells.
        
        Args:
            usv_speed: The speed of the USV in m/s
            
        Returns:
            A matrix of travel costs between all cells
        """
        num_cells = self.grid_size[0] * self.grid_size[1]
        distances = np.ones((num_cells, num_cells)) * np.inf
        
        # Compute costs for adjacent cells
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                if self.is_obstacle(x, y):
                    continue
                    
                cell_idx = y * self.grid_size[0] + x
                
                # Set diagonal to 0
                distances[cell_idx, cell_idx] = 0
                
                for nx, ny in self.get_neighbors(x, y):
                    neighbor_idx = ny * self.grid_size[0] + nx
                    cost = self.calculate_travel_cost(x, y, nx, ny, usv_speed)
                    distances[cell_idx, neighbor_idx] = cost
        
        # Floyd-Warshall algorithm to find all-pairs shortest paths
        for k in range(num_cells):
            for i in range(num_cells):
                for j in range(num_cells):
                    if distances[i, k] + distances[k, j] < distances[i, j]:
                        distances[i, j] = distances[i, k] + distances[k, j]
        
        return distances