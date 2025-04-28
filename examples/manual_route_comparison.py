"""
Interactive tool for creating and comparing manual routes against algorithmic paths.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as mpatches

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.vector_field import VectorField
from src.core.grid import NavigationGrid
from src.visualization.grid_viz import plot_navigation_grid
from src.algorithms.path_planning import a_star_current_aware
from src.algorithms.network_path_finding import find_shortest_time_path
from src.algorithms.energy_optimal_path import find_energy_optimal_path

# Import the current field creation function from the demo file
from path_planning_demo import create_test_current_field


class ManualRouteCreator:
    """Interactive tool for creating and comparing manual routes."""
    
    def __init__(self):
        # Create navigation grid with current field
        self.grid_size = (100, 100)
        self.nav_grid = NavigationGrid(self.grid_size, cell_size=0.1, x_origin=0, y_origin=0)
        
        # Set the current field
        self.current_field = create_test_current_field()
        self.nav_grid.set_current_field(self.current_field)
        
        # Add obstacles (same as in the demo)
        self.setup_obstacles()
        
        # Set USV parameters
        self.usv_speed = 0.7  # m/s
        self.max_speed = 1.0
        
        # Define the same start and goal points as in the demo
        self.demo_start = (15, 80)  # Top left
        self.demo_goal = (85, 30)   # Bottom right
        
        # Initialize routes storage
        self.manual_routes = {}
        self.current_route = []
        self.route_index = 1
        
        # Setup the figure and plot
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Track created colorbars to prevent duplicates
        self.current_cbar = None
        
        self.setup_plot()
        
        # Add buttons
        self.setup_buttons()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Algorithmic routes for comparison
        self.algo_routes = {}

    def setup_obstacles(self):
        """Set up the same obstacles as in the demo for consistency."""
        # Island 1
        self.nav_grid.add_obstacle_region(20, 20, 35, 35)
        
        # Island 2 - Elongated
        self.nav_grid.add_obstacle_region(60, 50, 80, 60)
        
        # Coastal area
        for x in range(self.grid_size[0]):
            for y in range(10):
                self.nav_grid.add_obstacle(x, y)
    
    def setup_plot(self):
        """Initial plot setup."""
        # Clear any existing axes
        self.ax.clear()
        
        # Remove existing colorbar if it exists
        if hasattr(self, 'cbar') and self.cbar is not None:
            self.cbar.remove()
            self.cbar = None
        
        # Draw the grid and currents
        plot_navigation_grid(
            self.nav_grid,
            ax=self.ax,
            show_obstacles=True,
            show_currents=True,
            title=f"Click to create Manual Route {self.route_index}"
        )
        
        # Plot the demo start and end points
        start_x, start_y = self.nav_grid.cell_to_coords(*self.demo_start)
        goal_x, goal_y = self.nav_grid.cell_to_coords(*self.demo_goal)
        
        # Draw start point - green star
        self.ax.scatter(start_x, start_y, marker='*', color='green', s=200, 
                     edgecolor='black', linewidth=1, zorder=10,
                     label='Demo start')
        
        # Draw end point - red diamond
        self.ax.scatter(goal_x, goal_y, marker='D', color='red', s=150, 
                     edgecolor='black', linewidth=1, zorder=10,
                     label='Demo end')
        
        # Add instructions
        self.instructions = self.ax.text(
            0.5, 0.02, 
            "Left-click to add waypoints. Use buttons to manage routes.",
            transform=self.ax.transAxes,
            ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Add legend for start/end
        self.ax.legend(loc='upper right')
        
        self.fig.tight_layout()
    
    def setup_buttons(self):
        """Create interactive buttons."""
        # Add button positions
        button_width = 0.12
        button_height = 0.05
        button_y = 0.01
        
        # Complete current route button
        self.ax_complete = plt.axes([0.1, button_y, button_width, button_height])
        self.btn_complete = Button(self.ax_complete, 'Complete Route')
        self.btn_complete.on_clicked(self.on_complete_route)
        
        # Clear current route button
        self.ax_clear = plt.axes([0.25, button_y, button_width, button_height])
        self.btn_clear = Button(self.ax_clear, 'Clear Route')
        self.btn_clear.on_clicked(self.on_clear_route)
        
        # Next route button
        self.ax_next = plt.axes([0.4, button_y, button_width, button_height])
        self.btn_next = Button(self.ax_next, 'Next Route')
        self.btn_next.on_clicked(self.on_next_route)
        
        # Calculate metrics button
        self.ax_metrics = plt.axes([0.55, button_y, button_width, button_height])
        self.btn_metrics = Button(self.ax_metrics, 'Calculate Metrics')
        self.btn_metrics.on_clicked(self.on_calculate_metrics)
        
        # Compare with algorithms button
        self.ax_compare = plt.axes([0.7, button_y, button_width, button_height])
        self.btn_compare = Button(self.ax_compare, 'Compare All')
        self.btn_compare.on_clicked(self.on_compare_all)
    
    def on_click(self, event):
        """Handle mouse clicks on the plot."""
        if event.inaxes != self.ax:
            return
        
        # Convert click coordinates to grid cell
        world_x, world_y = event.xdata, event.ydata
        cell_x, cell_y = self.nav_grid.coords_to_cell(world_x, world_y)
        
        # Check if cell is valid
        if (0 <= cell_x < self.grid_size[0] and 
            0 <= cell_y < self.grid_size[1] and 
            not self.nav_grid.is_obstacle(cell_x, cell_y)):
            
            # Add to current route
            self.current_route.append((cell_x, cell_y))
            
            # Update plot
            self.update_plot()
    
    def update_plot(self):
        """Update the plot with current waypoints."""
        # Clear previous plot
        self.ax.clear()
        
        # Remove existing colorbar if it exists
        if hasattr(self, 'cbar') and self.cbar is not None:
            self.cbar.remove()
            self.cbar = None
        
        # Draw navigation grid
        plot_navigation_grid(
            self.nav_grid,
            ax=self.ax,
            show_obstacles=True,
            show_currents=True,
            title=f"Manual Route {self.route_index} ({len(self.current_route)} waypoints)"
        )
        
        # Plot the demo start and end points
        start_x, start_y = self.nav_grid.cell_to_coords(*self.demo_start)
        goal_x, goal_y = self.nav_grid.cell_to_coords(*self.demo_goal)
        
        # Draw start point - green star
        self.ax.scatter(start_x, start_y, marker='*', color='green', s=200, 
                     edgecolor='black', linewidth=1, zorder=10,
                     label='Demo start')
        
        # Draw end point - red diamond
        self.ax.scatter(goal_x, goal_y, marker='D', color='red', s=150, 
                     edgecolor='black', linewidth=1, zorder=10,
                     label='Demo end')
        
        # Draw current route waypoints
        if self.current_route:
            # Convert to world coordinates
            world_points = []
            for x, y in self.current_route:
                world_x, world_y = self.nav_grid.cell_to_coords(x, y)
                world_points.append((world_x, world_y))
            
            # Extract x and y
            wx = [p[0] for p in world_points]
            wy = [p[1] for p in world_points]
            
            # Plot route line
            self.ax.plot(wx, wy, 'r-', linewidth=1.5)
            
            # Plot waypoints
            self.ax.scatter(wx, wy, marker='o', color='orange', s=80, 
                         edgecolor='black', linewidth=0.5,
                         label='Manual waypoints')
            
            # Add waypoint numbers
            for i, (x, y) in enumerate(world_points):
                self.ax.text(x, y, str(i+1), fontsize=10, 
                          ha='center', va='center',
                          color='white', fontweight='bold')
        
        # Add instructions
        self.instructions = self.ax.text(
            0.5, 0.02, 
            "Left-click to add waypoints. Use buttons to manage routes.",
            transform=self.ax.transAxes,
            ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        self.fig.canvas.draw_idle()
    
    def on_complete_route(self, event):
        """Save the current route and prepare for next one."""
        if len(self.current_route) < 2:
            self.show_message("Need at least 2 waypoints to complete a route.")
            return
        
        # Save current route
        self.manual_routes[self.route_index] = self.current_route.copy()
        self.show_message(f"Route {self.route_index} saved with {len(self.current_route)} waypoints.")
        
        # Clear for next route
        self.current_route = []
        self.route_index += 1
        
        # Update plot
        self.update_plot()
    
    def on_clear_route(self, event):
        """Clear the current route."""
        self.current_route = []
        self.update_plot()
        self.show_message("Current route cleared.")
    
    def on_next_route(self, event):
        """Move to the next route without saving current one."""
        if self.current_route:
            self.show_message("Current route was discarded.")
        
        self.current_route = []
        self.route_index += 1
        self.update_plot()
    
    def calculate_route_metrics(self, route):
        """Calculate metrics for a given route using energy-optimal movement between waypoints."""
        if len(route) < 2:
            return None
        
        full_path = []
        power_settings = []
        total_distance = 0
        total_time = 0
        total_energy = 0
        
        # Process each segment between waypoints
        for i in range(len(route) - 1):
            start = route[i]
            goal = route[i + 1]
            
            # Find energy-optimal path between these waypoints
            segment_path, segment_powers = find_energy_optimal_path(
                self.nav_grid, start, goal, 
                max_speed=self.usv_speed, 
                power_levels=6
            )
            
            if not segment_path:
                print(f"Warning: No path found between waypoints {i+1} and {i+2}")
                continue
            
            # Add to full path (avoiding duplicates)
            if full_path and full_path[-1] == segment_path[0]:
                segment_path = segment_path[1:]
                
            full_path.extend(segment_path)
            power_settings.extend(segment_powers)
            
            # Calculate metrics for this segment
            segment_distance = 0
            segment_time = 0
            segment_energy = 0
            
            for j in range(len(segment_path) - 1):
                x1, y1 = segment_path[j]
                x2, y2 = segment_path[j + 1]
                
                # Calculate distance
                wx1, wy1 = self.nav_grid.cell_to_coords(x1, y1)
                wx2, wy2 = self.nav_grid.cell_to_coords(x2, y2)
                dist = np.sqrt((wx2 - wx1)**2 + (wy2 - wy1)**2)
                segment_distance += dist
                
                # Calculate time
                time = self.nav_grid.calculate_travel_cost(x1, y1, x2, y2, self.usv_speed)
                segment_time += time
                
                # Calculate energy
                if j < len(segment_powers):
                    power = segment_powers[j]
                    energy = (power / 100.0) ** 3  # Energy ~ power^3
                    segment_energy += energy
            
            total_distance += segment_distance
            total_time += segment_time
            total_energy += segment_energy
            
            print(f"Segment {i+1}-{i+2}: {len(segment_path)} cells, "
                 f"Distance: {segment_distance:.2f}m, Time: {segment_time:.2f}s, "
                 f"Energy: {segment_energy:.2f}")
        
        return {
            'path': full_path,
            'powers': power_settings,
            'distance': total_distance,
            'time': total_time,
            'energy': total_energy
        }
    
    def on_calculate_metrics(self, event):
        """Calculate and show metrics for all saved routes."""
        if not self.manual_routes:
            self.show_message("No saved routes to calculate metrics for.")
            return
        
        # Calculate metrics for all routes
        results = {}
        for route_idx, route in self.manual_routes.items():
            print(f"\nCalculating metrics for Route {route_idx}...")
            metrics = self.calculate_route_metrics(route)
            if metrics:
                results[route_idx] = metrics
                print(f"Route {route_idx}: {len(metrics['path'])} total cells, "
                     f"Distance: {metrics['distance']:.2f}m, "
                     f"Time: {metrics['time']:.2f}s, "
                     f"Energy: {metrics['energy']:.2f}")
        
        if results:
            self.show_results_plot(results)
    
    def on_compare_all(self, event):
        """Compare manual routes with algorithmic approaches."""
        if not self.manual_routes:
            self.show_message("No saved routes to compare.")
            return
        
        # Calculate metrics for manual routes if not already done
        manual_results = {}
        for route_idx, route in self.manual_routes.items():
            metrics = self.calculate_route_metrics(route)
            if metrics:
                manual_results[f"Manual {route_idx}"] = metrics
        
        # Calculate metrics for algorithmic routes
        algo_results = {}
        
        # Find common start/end points from all manual routes
        all_starts = [route[0] for route in self.manual_routes.values()]
        all_ends = [route[-1] for route in self.manual_routes.values()]
        
        # For simplicity, use the first route's endpoints
        start = self.manual_routes[1][0]
        goal = self.manual_routes[1][-1]
        
        print(f"\nComparing all routes with algorithmic paths from {start} to {goal}...")
        
        # A* current-aware path
        a_star_path = a_star_current_aware(
            self.nav_grid, start, goal, usv_speed=self.usv_speed
        )
        
        # Shortest time (Dijkstra) path
        dijkstra_path = find_shortest_time_path(
            self.nav_grid, start, goal, usv_speed=self.usv_speed
        )
        
        # Energy-optimal path
        energy_path, power_settings = find_energy_optimal_path(
            self.nav_grid, start, goal, max_speed=self.usv_speed, power_levels=6
        )
        
        # Define placeholder for metrics
        algo_paths = {
            'A* Current-Aware': a_star_path,
            'Shortest Time (Dijkstra)': dijkstra_path,
            'Energy-Optimal': energy_path
        }
        
        # Calculate metrics for algorithmic paths
        for name, path in algo_paths.items():
            if not path:
                continue
                
            if name == 'Energy-Optimal':
                # We already have power settings for this one
                powers = power_settings
            else:
                # Assume constant 100% power for other algorithms
                powers = [100] * (len(path) - 1)
            
            metrics = self.calculate_path_metrics(path, powers)
            algo_results[name] = metrics
        
        # Combine all results
        all_results = {**manual_results, **algo_results}
        
        # Display comparison
        self.show_comparison_plot(all_results)
    
    def calculate_path_metrics(self, path, powers=None):
        """Calculate metrics for a path."""
        if len(path) < 2:
            return None
            
        total_distance = 0
        total_time = 0
        total_energy = 0
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Calculate distance
            wx1, wy1 = self.nav_grid.cell_to_coords(x1, y1)
            wx2, wy2 = self.nav_grid.cell_to_coords(x2, y2)
            dist = np.sqrt((wx2 - wx1)**2 + (wy2 - wy1)**2)
            
            # Calculate time
            time = self.nav_grid.calculate_travel_cost(x1, y1, x2, y2, self.usv_speed)
            
            # Calculate energy
            if powers and i < len(powers):
                power = powers[i]
                energy = (power / 100.0) ** 3  # Energy ~ power^3
            else:
                # Default to 100% power if not specified
                energy = 1.0
            
            total_distance += dist
            total_time += time
            total_energy += energy
        
        return {
            'path': path,
            'powers': powers if powers else [100] * (len(path) - 1),
            'distance': total_distance,
            'time': total_time,
            'energy': total_energy
        }
    
    def show_results_plot(self, results):
        """Show metrics for all routes in a visual format."""
        # Create a figure to display results
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Store reference to colorbar
        colorbar = None
        
        # Plot the navigation grid
        plot_navigation_grid(
            self.nav_grid,
            ax=ax,
            show_obstacles=True,
            show_currents=True,
            title="Manual Routes with Energy-Optimal Movements"
        )
        
        # Plot the demo start and end points
        start_x, start_y = self.nav_grid.cell_to_coords(*self.demo_start)
        goal_x, goal_y = self.nav_grid.cell_to_coords(*self.demo_goal)
        
        # Draw demo start point - green star
        ax.scatter(start_x, start_y, marker='*', color='green', s=200, 
                 edgecolor='black', linewidth=1, zorder=10,
                 label='Demo start')
        
        # Draw demo end point - red diamond
        ax.scatter(goal_x, goal_y, marker='D', color='red', s=150, 
                 edgecolor='black', linewidth=1, zorder=10,
                 label='Demo end')
        
        # Plot each route with a different color
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for (route_idx, metrics), color in zip(results.items(), colors):
            path = metrics['path']
            powers = metrics['powers']
            
            # Convert to world coordinates
            world_points = []
            for x, y in path:
                world_x, world_y = self.nav_grid.cell_to_coords(x, y)
                world_points.append((world_x, world_y))
            
            # Extract x and y
            wx = [p[0] for p in world_points]
            wy = [p[1] for p in world_points]
            
            # Plot route
            ax.plot(wx, wy, '-', color=color, linewidth=2, label=f'Route {route_idx}')
            
            # Plot original waypoints
            waypoints = self.manual_routes[route_idx]
            waypoint_x = []
            waypoint_y = []
            for x, y in waypoints:
                wx, wy = self.nav_grid.cell_to_coords(x, y)
                waypoint_x.append(wx)
                waypoint_y.append(wy)
            
            # Plot waypoints with the route color
            ax.scatter(waypoint_x, waypoint_y, marker='o', color=color, s=100,
                     edgecolor='black', linewidth=1)
                
            # Add waypoint numbers
            for i, (x, y) in enumerate(zip(waypoint_x, waypoint_y)):
                ax.text(x, y, str(i+1), fontsize=10, ha='center', va='center',
                      color='white', fontweight='bold')
            
            # Add metrics text
            ax.text(0.02, 0.98 - 0.05 * (route_idx-1), 
                  f"Route {route_idx}: {len(path)} cells, Dist: {metrics['distance']:.2f}m, "
                  f"Time: {metrics['time']:.2f}s, Energy: {metrics['energy']:.2f}",
                  transform=ax.transAxes, fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.7),
                  color=color)
        
        # Add legend with custom elements
        legend_elements = [
            plt.scatter([0], [0], marker='*', color='green', s=100, edgecolor='black', label='Demo start'),
            plt.scatter([0], [0], marker='D', color='red', s=60, edgecolor='black', label='Demo end'),
        ]
        
        # Add route-specific legend entries
        for (route_idx, _), color in zip(results.items(), colors):
            legend_elements.append(
                plt.Line2D([0], [0], color=color, lw=2, marker='o', markersize=8,
                          markeredgecolor='black', markerfacecolor=color,
                          label=f'Route {route_idx}')
            )
        
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.show()
    
    def show_comparison_plot(self, results):
        """Show comparison between manual and algorithmic routes."""
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot the navigation grid in the first axis
        plot_navigation_grid(
            self.nav_grid,
            ax=ax1,
            show_obstacles=True,
            show_currents=True,
            title="Route Comparison"
        )
        
        # Plot the demo start and end points first
        start_x, start_y = self.nav_grid.cell_to_coords(*self.demo_start)
        goal_x, goal_y = self.nav_grid.cell_to_coords(*self.demo_goal)
        
        # Draw demo start point - green star
        ax1.scatter(start_x, start_y, marker='*', color='green', s=200, 
                  edgecolor='black', linewidth=1, zorder=10,
                  label='Demo start')
        
        # Draw demo end point - red diamond
        ax1.scatter(goal_x, goal_y, marker='D', color='red', s=150, 
                  edgecolor='black', linewidth=1, zorder=10,
                  label='Demo end')
        
        # Plot each path
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for (route_name, metrics), color in zip(results.items(), colors):
            path = metrics['path']
            
            # Convert to world coordinates
            world_points = []
            for x, y in path:
                world_x, world_y = self.nav_grid.cell_to_coords(x, y)
                world_points.append((world_x, world_y))
            
            # Extract x and y
            wx = [p[0] for p in world_points]
            wy = [p[1] for p in world_points]
            
            # Plot route
            ax1.plot(wx, wy, '-', color=color, linewidth=2, label=route_name)
            
            # If it's a manual route, mark the waypoints
            if route_name.startswith('Manual'):
                route_idx = int(route_name.split()[1])
                waypoints = self.manual_routes[route_idx]
                waypoint_x = []
                waypoint_y = []
                for x, y in waypoints:
                    wx, wy = self.nav_grid.cell_to_coords(x, y)
                    waypoint_x.append(wx)
                    waypoint_y.append(wy)
                
                # Plot waypoints with the route color
                ax1.scatter(waypoint_x, waypoint_y, marker='o', color=color, s=100,
                          edgecolor='black', linewidth=1)
        
        # Add legend with custom elements
        legend_elements = [
            plt.scatter([0], [0], marker='*', color='green', s=100, edgecolor='black', label='Demo start'),
            plt.scatter([0], [0], marker='D', color='red', s=60, edgecolor='black', label='Demo end'),
        ]
        
        # Add route-specific legend entries
        for (route_name, _), color in zip(results.items(), colors):
            # Manual routes
            if route_name.startswith('Manual'):
                legend_elements.append(
                    plt.Line2D([0], [0], color=color, lw=2, marker='o', markersize=8,
                              markeredgecolor='black', markerfacecolor=color,
                              label=route_name)
                )
            # Algorithm routes
            else:
                legend_elements.append(
                    plt.Line2D([0], [0], color=color, lw=2, label=route_name)
                )
        
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # Create bar chart in the second axis
        route_names = list(results.keys())
        distances = [results[name]['distance'] for name in route_names]
        times = [results[name]['time'] for name in route_names]
        energies = [results[name]['energy'] for name in route_names]
        
        # Set up bar positions
        x = np.arange(len(route_names))
        width = 0.25
        
        # Create bars
        ax2.bar(x - width, distances, width, label='Distance (m)', color='skyblue')
        ax2.bar(x, times, width, label='Time (s)', color='orange')
        ax2.bar(x + width, energies, width, label='Energy (units)', color='green')
        
        # Add labels and legend
        ax2.set_title('Route Metrics Comparison')
        ax2.set_xlabel('Route')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x)
        ax2.set_xticklabels(route_names, rotation=45, ha='right')
        ax2.legend()
        
        # Add data table
        cell_text = []
        for name in route_names:
            metrics = results[name]
            cell_text.append([
                f"{metrics['distance']:.2f}",
                f"{metrics['time']:.2f}",
                f"{metrics['energy']:.2f}"
            ])
        
        table = ax2.table(
            cellText=cell_text,
            rowLabels=route_names,
            colLabels=['Distance (m)', 'Time (s)', 'Energy'],
            loc='bottom',
            bbox=[0, -0.40, 1, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Adjust layout
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        plt.show()
    
    def show_message(self, message):
        """Display a message to the user."""
        print(message)
        # You could also add a text message on the plot
        
    def run(self):
        """Start the interactive session."""
        plt.show()


if __name__ == "__main__":
    route_creator = ManualRouteCreator()
    route_creator.run()