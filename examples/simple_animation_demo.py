#!/usr/bin/env python3
"""
Simple animation demo for time-varying currents.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.spatio_temporal_field import SpatioTemporalField


def create_rotating_gyre(grid_size=(40, 40), time_steps=24):
    """Create a rotating gyre pattern that changes with time."""
    # Create spatio-temporal field
    field = SpatioTemporalField(
        grid_size, time_steps, 
        x_range=(0, 100), 
        y_range=(0, 100)
    )
    
    # Generate field data for each time step
    center_x, center_y = 50, 50  # Center of the domain
    
    for t in range(time_steps):
        # Create meshgrid for this time step
        x = np.linspace(0, 100, grid_size[0])
        y = np.linspace(0, 100, grid_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Time-varying factors
        phase = 2 * np.pi * t / time_steps
        strength = 0.5 + 0.3 * np.sin(phase)
        
        # Gyre pattern with time variation
        dx = X - center_x
        dy = Y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        mask = (distance != 0)
        
        # Initialize U and V
        U = np.zeros(grid_size[::-1])
        V = np.zeros(grid_size[::-1])
        
        # Current speed peaks at distance = 20 from center
        radial_factor = np.exp(-((distance - 20) / 15)**2)
        
        # Tangential velocity (creates circular flow)
        U[mask] = -strength * radial_factor[mask] * dy[mask] / distance[mask]
        V[mask] = strength * radial_factor[mask] * dx[mask] / distance[mask]
        
        # Add a general flow in time-varying direction
        base_angle = phase
        base_u = 0.2 * np.cos(base_angle) 
        base_v = 0.2 * np.sin(base_angle)
        
        U += base_u
        V += base_v
        
        # Create weather factor (higher in center, varies with time)
        W = 0.2 + 0.8 * np.exp(-distance / 30) * (0.5 + 0.5 * np.sin(phase))
        
        # Set field data for this time step
        field.set_field_at_time(t, U, V, W)
    
    return field


def animate_field(field):
    """Create an animation of the time-varying vector field."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up plot parameters
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_title('Time-Varying Current Field')
    
    # Create a grid for the quiver plot
    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 100, 20)
    X, Y = np.meshgrid(x, y)
    
    # Initial quiver plot (will be updated in animation)
    u, v, _ = field.get_vector_at_position_time(X[0, 0], Y[0, 0], 0)
    quiver = ax.quiver(X, Y, np.ones_like(X) * u, np.ones_like(Y) * v)
    
    # Create colorbar for magnitude
    magnitude = np.zeros_like(X)
    cbar = plt.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(0, 1), 
        cmap='viridis'
    ), ax=ax)
    cbar.set_label('Current Speed (m/s)')
    
    # Animation update function
    def update(frame):
        # Get vectors at this time step
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        M = np.zeros_like(X)  # Magnitude for colors
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                u, v, _ = field.get_vector_at_position_time(X[i, j], Y[i, j], frame)
                U[i, j] = u
                V[i, j] = v
                M[i, j] = np.sqrt(u**2 + v**2)
        
        # Update quiver
        quiver.set_UVC(U, V, M)
        
        # Update title
        ax.set_title(f'Time-Varying Current Field (Time: {frame:.1f})')
        
        return quiver,
    
    # Create animation
    ani = FuncAnimation(
        fig, update, frames=np.linspace(0, field.time_steps-1, 50),
        interval=100, blit=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return ani


def main():
    print("Creating time-varying field...")
    field = create_rotating_gyre(grid_size=(40, 40), time_steps=24)
    
    print("Generating animation...")
    ani = animate_field(field)
    
    # Save animation (optional)
    # ani.save('gyre_animation.mp4', writer='ffmpeg')


if __name__ == "__main__":
    main()