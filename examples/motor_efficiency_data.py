"""
Motor efficiency data for different propulsion systems.

Data extracted from: https://www.electricpaddle.com/efficiency.html

This module contains power-speed data for three different electric propulsion systems:
- Endura 30
- EP Carry
- Torqeedo 1003

It also includes functions to plot the data and interpolate power requirements
for specific speeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Data extracted from chart at https://www.electricpaddle.com/efficiency.html
MOTOR_DATA = {
    "Endura 30": {
        "speed_kts": [1.4, 1.7, 2.1, 2.3, 2.9],
        "power_w": [100, 150, 175, 200, 250]
    },
    "EP Carry": {
        "speed_kts": [1.8, 2.2, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1],
        "power_w": [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    },
    "Torqeedo 1003": {
        "speed_kts": [1.7, 2.0, 2.7, 3.0, 3.3, 3.5, 3.7, 4.0, 4.2, 4.3, 4.4, 4.7],
        "power_w": [100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 1000]
    }
}

def get_interpolation_function(motor_type):
    """
    Get a power interpolation function for the specified motor type.
    
    Args:
        motor_type: One of "Endura 30", "EP Carry", or "Torqeedo 1003"
        
    Returns:
        A function that takes speed (kts) and returns power (W)
    """
    if motor_type not in MOTOR_DATA:
        raise ValueError(f"Unknown motor type: {motor_type}")
        
    speed_data = MOTOR_DATA[motor_type]["speed_kts"]
    power_data = MOTOR_DATA[motor_type]["power_w"]
    
    # Create interpolation function (use cubic for smoother curves)
    return interp1d(speed_data, power_data, kind='cubic', 
                   bounds_error=False, fill_value='extrapolate')

def calculate_power_for_speed(speed_kts, motor_type):
    """
    Calculate power required (W) for a given speed (kts) using interpolation.
    
    Args:
        speed_kts: Speed in knots
        motor_type: One of "Endura 30", "EP Carry", or "Torqeedo 1003"
        
    Returns:
        Required power in Watts
    """
    interp_func = get_interpolation_function(motor_type)
    return float(interp_func(speed_kts))

def plot_motor_efficiency(speed_on_x=True, figsize=(10, 6)):
    """
    Create a plot of motor efficiency data.
    
    Args:
        speed_on_x: If True, plot speed on x-axis. If False, plot power on x-axis.
        figsize: Figure size tuple (width, height)
        
    Returns:
        The matplotlib figure object
    """
    colors = {
        "Endura 30": "black",
        "EP Carry": "royalblue",
        "Torqeedo 1003": "saddlebrown"
    }
    
    markers = {
        "Endura 30": "D",
        "EP Carry": "s",
        "Torqeedo 1003": "^"
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each motor dataset
    for motor, data in MOTOR_DATA.items():
        speed = data["speed_kts"]
        power = data["power_w"]
        
        # Create smooth interpolated curves
        interp_func = get_interpolation_function(motor)
        
        if speed_on_x:
            # Speed on x-axis, Power on y-axis
            speed_smooth = np.linspace(min(speed), max(speed), 100)
            power_smooth = interp_func(speed_smooth)
            
            # Plot data points
            ax.plot(speed, power, markers[motor], color=colors[motor], 
                   markersize=8, label=f"{motor} (data)")
            
            # Plot smooth curve
            ax.plot(speed_smooth, power_smooth, '-', color=colors[motor], 
                   linewidth=2, label=f"{motor} (curve)")
            
            ax.set_xlabel('Speed (knots)')
            ax.set_ylabel('Input Power (Watts)')
        else:
            # Power on x-axis, Speed on y-axis
            power_smooth = np.linspace(min(power), max(power), 100)
            # Use inverse interpolation
            speed_interp = interp1d(power, speed, kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')
            speed_smooth = speed_interp(power_smooth)
            
            # Plot data points
            ax.plot(power, speed, markers[motor], color=colors[motor], 
                   markersize=8, label=f"{motor} (data)")
            
            # Plot smooth curve
            ax.plot(power_smooth, speed_smooth, '-', color=colors[motor], 
                   linewidth=2, label=f"{motor} (curve)")
            
            ax.set_xlabel('Input Power (Watts)')
            ax.set_ylabel('Speed (knots)')
    
    # Set plot appearance
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_title('Electric Motor Efficiency Comparison')
    
    # Add legend with only the main entries (remove duplicate curve entries)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = [label.split(' ')[0] for label in labels]
    unique_indices = []
    for i, label in enumerate(unique_labels):
        if label not in unique_labels[:i]:
            unique_indices.append(i)
    
    ax.legend([handles[i] for i in unique_indices], 
             [labels[i].split(' ')[0] for i in unique_indices], 
             loc='best')
    
    plt.tight_layout()
    return fig

def calculate_energy_consumption(distance_nm, speed_kts, motor_type):
    """
    Calculate total energy consumption for a given distance and speed.
    
    Args:
        distance_nm: Distance in nautical miles
        speed_kts: Speed in knots
        motor_type: One of "Endura 30", "EP Carry", or "Torqeedo 1003"
        
    Returns:
        Dictionary with:
        - power_w: Required power in Watts
        - time_hours: Travel time in hours
        - energy_wh: Total energy consumption in Watt-hours
    """
    power_w = calculate_power_for_speed(speed_kts, motor_type)
    time_hours = distance_nm / speed_kts
    energy_wh = power_w * time_hours
    
    return {
        "power_w": power_w,
        "time_hours": time_hours,
        "energy_wh": energy_wh
    }

# Example usage
if __name__ == "__main__":
    # Plot with speed on x-axis
    fig = plot_motor_efficiency(speed_on_x=True)
    plt.savefig("motor_efficiency_speed_x.png", dpi=300)
    
    # Example energy calculation
    distance = 10  # nautical miles
    speed = 3.0  # knots
    motor = "Torqeedo 1003"
    
    result = calculate_energy_consumption(distance, speed, motor)
    print(f"To travel {distance} nautical miles at {speed} knots with {motor}:")
    print(f"  Required power: {result['power_w']:.1f} Watts")
    print(f"  Travel time: {result['time_hours']:.2f} hours")
    print(f"  Energy consumption: {result['energy_wh']:.1f} Watt-hours")