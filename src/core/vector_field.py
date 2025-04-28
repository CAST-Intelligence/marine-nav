"""
Vector field implementation for current modeling in marine environments.
"""
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from scipy import interpolate


class VectorField:
    """
    A 2D vector field representation for marine currents.
    
    This class provides methods to:
    - Create a 2D grid with vector values (magnitude and direction)
    - Interpolate vector values at arbitrary positions
    - Identify critical points in the field
    - Calculate field properties (divergence, curl)
    """
    
    def __init__(
        self, 
        grid_size: Tuple[int, int], 
        x_range: Tuple[float, float] = (0, 1), 
        y_range: Tuple[float, float] = (0, 1)
    ):
        """
        Initialize a vector field with a specified grid size.
        
        Args:
            grid_size: The (width, height) of the grid in cells
            x_range: The (min, max) range of x coordinates
            y_range: The (min, max) range of y coordinates
        """
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        
        # Create coordinate meshgrid
        x = np.linspace(x_range[0], x_range[1], grid_size[0])
        y = np.linspace(y_range[0], y_range[1], grid_size[1])
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initialize vector components to zero
        self.U = np.zeros(grid_size)  # x-component of vectors
        self.V = np.zeros(grid_size)  # y-component of vectors
        
        # Interpolation functions (initialized as None)
        self.u_interp = None
        self.v_interp = None

    def set_field(self, U: np.ndarray, V: np.ndarray):
        """
        Set the vector field components.
        
        Args:
            U: x-component of the vector field
            V: y-component of the vector field
        """
        if U.shape != self.grid_size or V.shape != self.grid_size:
            raise ValueError(f"Vector field components must have shape {self.grid_size}")
        
        self.U = U
        self.V = V
        
        # Create interpolation functions
        self.u_interp = interpolate.RectBivariateSpline(
            np.linspace(self.x_range[0], self.x_range[1], self.grid_size[0]),
            np.linspace(self.y_range[0], self.y_range[1], self.grid_size[1]), 
            self.U
        )
        self.v_interp = interpolate.RectBivariateSpline(
            np.linspace(self.x_range[0], self.x_range[1], self.grid_size[0]),
            np.linspace(self.y_range[0], self.y_range[1], self.grid_size[1]), 
            self.V
        )

    def generate_from_function(self, u_func, v_func):
        """
        Generate vector field from functions.
        
        Args:
            u_func: Function that takes (x, y) and returns the x-component
            v_func: Function that takes (x, y) and returns the y-component
        """
        U = np.zeros(self.grid_size)
        V = np.zeros(self.grid_size)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                U[j, i] = u_func(self.X[j, i], self.Y[j, i])
                V[j, i] = v_func(self.X[j, i], self.Y[j, i])
        
        self.set_field(U, V)

    def get_vector_at_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get the vector at a specific position using interpolation.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            
        Returns:
            Tuple of (u, v) vector components
        """
        if self.u_interp is None or self.v_interp is None:
            raise ValueError("Vector field not initialized")
            
        u = float(self.u_interp(x, y, grid=False))
        v = float(self.v_interp(x, y, grid=False))
        
        return u, v
    
    def get_magnitude_at_position(self, x: float, y: float) -> float:
        """
        Get the vector magnitude at a specific position.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            
        Returns:
            Magnitude of the vector
        """
        u, v = self.get_vector_at_position(x, y)
        return np.sqrt(u**2 + v**2)
    
    def calculate_divergence(self) -> np.ndarray:
        """
        Calculate the divergence of the vector field.
        
        Returns:
            Array of divergence values across the grid
        """
        dx = (self.x_range[1] - self.x_range[0]) / (self.grid_size[0] - 1)
        dy = (self.y_range[1] - self.y_range[0]) / (self.grid_size[1] - 1)
        
        du_dx = np.gradient(self.U, dx, axis=1)
        dv_dy = np.gradient(self.V, dy, axis=0)
        
        return du_dx + dv_dy
    
    def calculate_curl(self) -> np.ndarray:
        """
        Calculate the curl of the vector field (z-component in 2D).
        
        Returns:
            Array of curl values across the grid
        """
        dx = (self.x_range[1] - self.x_range[0]) / (self.grid_size[0] - 1)
        dy = (self.y_range[1] - self.y_range[0]) / (self.grid_size[1] - 1)
        
        dv_dx = np.gradient(self.V, dx, axis=1)
        du_dy = np.gradient(self.U, dy, axis=0)
        
        return dv_dx - du_dy
    
    def find_critical_points(self, threshold: float = 1e-10) -> List[Tuple[float, float, str]]:
        """
        Find critical points in the vector field.
        
        Args:
            threshold: Magnitude threshold for critical points
            
        Returns:
            List of (x, y, type) tuples for critical points
        """
        critical_points = []
        
        for i in range(1, self.grid_size[0]-1):
            for j in range(1, self.grid_size[1]-1):
                x = self.X[j, i]
                y = self.Y[j, i]
                magnitude = np.sqrt(self.U[j, i]**2 + self.V[j, i]**2)
                
                if magnitude < threshold:
                    # Calculate Jacobian matrix
                    dx = (self.x_range[1] - self.x_range[0]) / (self.grid_size[0] - 1)
                    dy = (self.y_range[1] - self.y_range[0]) / (self.grid_size[1] - 1)
                    
                    du_dx = (self.U[j, i+1] - self.U[j, i-1]) / (2 * dx)
                    du_dy = (self.U[j+1, i] - self.U[j-1, i]) / (2 * dy)
                    dv_dx = (self.V[j, i+1] - self.V[j, i-1]) / (2 * dx)
                    dv_dy = (self.V[j+1, i] - self.V[j-1, i]) / (2 * dy)
                    
                    jacobian = np.array([[du_dx, du_dy], [dv_dx, dv_dy]])
                    
                    # Classify critical point based on eigenvalues
                    try:
                        eigenvalues = np.linalg.eigvals(jacobian)
                        
                        real_parts = eigenvalues.real
                        imag_parts = eigenvalues.imag
                        
                        if np.all(real_parts < 0) and np.any(imag_parts != 0):
                            cp_type = "spiral-sink"
                        elif np.all(real_parts > 0) and np.any(imag_parts != 0):
                            cp_type = "spiral-source"
                        elif np.all(real_parts < 0):
                            cp_type = "sink"
                        elif np.all(real_parts > 0):
                            cp_type = "source"
                        elif np.prod(real_parts) < 0:
                            cp_type = "saddle"
                        else:
                            cp_type = "center"
                            
                        critical_points.append((x, y, cp_type))
                    except np.linalg.LinAlgError:
                        pass
                    
        return critical_points