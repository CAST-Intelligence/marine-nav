"""
Spatio-temporal vector field for time-varying current modeling in marine environments.
"""
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
from scipy import interpolate


class SpatioTemporalField:
    """
    A time-varying 2D vector field representation for marine currents.
    
    This class extends the VectorField concept to include time variation,
    allowing for forecasts of currents/weather at discrete time steps.
    It provides methods to:
    - Represent currents at multiple time steps
    - Interpolate values in both space and time
    - Calculate time-specific vector values at arbitrary positions
    """
    
    def __init__(
        self, 
        grid_size: Tuple[int, int], 
        time_steps: int,
        x_range: Tuple[float, float] = (0, 1), 
        y_range: Tuple[float, float] = (0, 1),
        time_step_duration: float = 3600.0  # Default 1 hour in seconds
    ):
        """
        Initialize a spatio-temporal vector field with multiple time steps.
        
        Args:
            grid_size: The (width, height) of the grid in cells
            time_steps: Number of discrete time steps in the forecast
            x_range: The (min, max) range of x coordinates
            y_range: The (min, max) range of y coordinates
            time_step_duration: Duration of each time step in seconds
        """
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.x_range = x_range
        self.y_range = y_range
        self.time_step_duration = time_step_duration
        
        # Create coordinate meshgrid
        x = np.linspace(x_range[0], x_range[1], grid_size[0])
        y = np.linspace(y_range[0], y_range[1], grid_size[1])
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initialize vector components for all time steps
        # Shape: [time_step, y, x]
        self.U = np.zeros((time_steps, grid_size[1], grid_size[0]))  # x-component
        self.V = np.zeros((time_steps, grid_size[1], grid_size[0]))  # y-component
        
        # Weather factor (e.g., wind, waves) that affects effort
        self.W = np.zeros((time_steps, grid_size[1], grid_size[0]))
        
        # Interpolation functions (initialized as None)
        self.u_interp = None
        self.v_interp = None
        self.w_interp = None
        self._interp_initialized = False

    def set_field_at_time(self, time_index: int, U: np.ndarray, V: np.ndarray, W: Optional[np.ndarray] = None):
        """
        Set the vector field components for a specific time step.
        
        Args:
            time_index: The index of the time step to set
            U: x-component of the vector field
            V: y-component of the vector field
            W: Optional weather factor values
        """
        if time_index < 0 or time_index >= self.time_steps:
            raise ValueError(f"Time index {time_index} out of range [0, {self.time_steps-1}]")
            
        if U.shape != self.grid_size[::-1] or V.shape != self.grid_size[::-1]:
            raise ValueError(f"Vector field components must have shape {self.grid_size[::-1]}")
        
        self.U[time_index] = U
        self.V[time_index] = V
        
        if W is not None:
            if W.shape != self.grid_size[::-1]:
                raise ValueError(f"Weather factor must have shape {self.grid_size[::-1]}")
            self.W[time_index] = W
        
        # Reset interpolation when field is modified
        self._interp_initialized = False

    def _initialize_interpolation(self):
        """
        Initialize the interpolation functions for all field components.
        Uses RegularGridInterpolator for efficient 3D interpolation.
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create coordinate grids for interpolation
        x_grid = np.linspace(self.x_range[0], self.x_range[1], self.grid_size[0])
        y_grid = np.linspace(self.y_range[0], self.y_range[1], self.grid_size[1])
        t_grid = np.arange(self.time_steps)
        
        # Create interpolation functions for each component
        self.u_interp = RegularGridInterpolator(
            (t_grid, y_grid, x_grid), self.U, 
            bounds_error=False, fill_value=0.0
        )
        
        self.v_interp = RegularGridInterpolator(
            (t_grid, y_grid, x_grid), self.V,
            bounds_error=False, fill_value=0.0
        )
        
        self.w_interp = RegularGridInterpolator(
            (t_grid, y_grid, x_grid), self.W,
            bounds_error=False, fill_value=0.0
        )
        
        self._interp_initialized = True

    def get_vector_at_position_time(
        self, 
        x: float, 
        y: float, 
        time_index: Union[int, float]
    ) -> Tuple[float, float, float]:
        """
        Get the vector and weather factor at a specific position and time.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            time_index: Time index (can be fractional for interpolation between steps)
            
        Returns:
            Tuple of (u, v, w) - current vector components and weather factor
        """
        if not self._interp_initialized:
            self._initialize_interpolation()
            
        # Ensure time_index is within bounds (clamp if needed)
        time_index = max(0, min(time_index, self.time_steps - 1.001))
        
        # Use 3D interpolation to get values at the specified position and time
        point = np.array([time_index, y, x])
        u = float(self.u_interp(point))
        v = float(self.v_interp(point))
        w = float(self.w_interp(point))
        
        return u, v, w
    
    def get_magnitude_at_position_time(self, x: float, y: float, time_index: Union[int, float]) -> float:
        """
        Get the vector magnitude at a specific position and time.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            time_index: Time index (can be fractional for interpolation between steps)
            
        Returns:
            Magnitude of the vector
        """
        u, v, _ = self.get_vector_at_position_time(x, y, time_index)
        return np.sqrt(u**2 + v**2)

    def integrate_drift_path(
        self, 
        start_x: float, 
        start_y: float, 
        start_time_index: float, 
        duration_seconds: float,
        dt: float = 10.0  # Time step for integration in seconds
    ) -> List[Tuple[float, float, float]]:
        """
        Simulate a drift path by integrating the current vectors over time.
        
        This method performs numerical integration (Euler method) to estimate
        the path a passive object would follow under the time-varying current field.
        
        Args:
            start_x: Starting x-coordinate
            start_y: Starting y-coordinate
            start_time_index: Starting time index
            duration_seconds: Duration of drift in seconds
            dt: Time step for numerical integration in seconds
            
        Returns:
            List of (x, y, time_index) points along the drift path
        """
        # Initialize the path with the starting point
        path = [(start_x, start_y, start_time_index)]
        
        # Calculate number of integration steps
        steps = int(duration_seconds / dt)
        
        # Current position and time
        x, y = start_x, start_y
        time_index = start_time_index
        
        # Integrate the path
        for _ in range(steps):
            # Convert time duration to time index increment
            time_index_increment = dt / self.time_step_duration
            
            # Get current at this position and time
            u, v, _ = self.get_vector_at_position_time(x, y, time_index)
            
            # Update position using Euler integration
            x += u * dt
            y += v * dt
            
            # Update time index
            time_index += time_index_increment
            
            # Check if we're past the last time step
            if time_index >= self.time_steps:
                break
                
            # Add point to path
            path.append((x, y, time_index))
        
        return path

    def load_forecast_data(self, data_source: str, format_type: str = "txt"):
        """
        Load forecast data from files or data source.
        
        Args:
            data_source: Path to directory or file containing forecast data
            format_type: Format of the data ("txt", "netcdf", etc.)
            
        This is a placeholder method - implementation would depend on
        specific data formats used in the application.
        """
        # Implementation would depend on specific data formats
        # This placeholder just fills with random values for testing
        import numpy as np
        
        for t in range(self.time_steps):
            # Create random currents with some temporal coherence
            angle = np.pi * 2 * (t / self.time_steps)
            base_u = 0.5 * np.cos(angle) 
            base_v = 0.5 * np.sin(angle)
            
            # Create random field with spatial correlation
            x = np.linspace(0, 5, self.grid_size[0])
            y = np.linspace(0, 5, self.grid_size[1])
            X, Y = np.meshgrid(x, y)
            
            U = base_u + 0.5 * np.sin(X/2) * np.cos(Y/2)
            V = base_v + 0.5 * np.cos(X/2) * np.sin(Y/2)
            W = 0.3 + 0.2 * np.sin(X/3 + Y/3 + t/self.time_steps * np.pi)
            
            self.set_field_at_time(t, U, V, W)