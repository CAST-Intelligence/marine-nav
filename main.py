"""
Main entry point for USV Mission Route Planning application.
"""
import argparse
import sys
import os

from src.core.vector_field import VectorField
from src.core.grid import NavigationGrid


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="USV Mission Route Planning with Current Awareness"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Vector field demo
    vf_parser = subparsers.add_parser("vector-field-demo", help="Run vector field demo")
    
    # Path planning demo
    path_parser = subparsers.add_parser("path-planning-demo", help="Run path planning demo")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "vector-field-demo":
        # Import and run the vector field demo
        sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))
        from examples.vector_field_demo import main as vf_demo
        vf_demo()
    elif args.command == "path-planning-demo":
        # Import and run the path planning demo
        sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))
        from examples.path_planning_demo import main as path_demo
        path_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()