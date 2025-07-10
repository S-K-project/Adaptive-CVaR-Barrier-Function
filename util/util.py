
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import os

def log_error(message):
    print(f"\033[31m[ERROR] {message}\033[0m")  # 红色

def log_debug(message):
    print(f"\033[34m[DEBUG] {message}\033[0m")  # 蓝色

def log_info(message):
    print(f"\033[32m[INFO] {message}\033[0m")   # 绿色
    

    

def generate_crowd_positions(range_low, range_high, number, radii, mode="y_upper_lower", existing_points=None, existing_radii=None):
    """
    Generate initial and target positions based on specified spatial division mode, avoiding collisions
    with existing points and considering their radii.

    Args:
        range_low (list): Lower bounds for x and y dimensions (e.g., [x_min, y_min]).
        range_high (list): Upper bounds for x and y dimensions (e.g., [x_max, y_max]).
        number (int): Number of people to generate.
        radii (float): Radii of each point.
        mode (str): Spatial division mode for initial and target positions.
        existing_points (np.ndarray): Existing points to avoid collisions with (optional).
        existing_radii (np.ndarray): Radii of existing points (optional).

    Returns:
        tuple: (initial_positions, target_positions)
            - initial_positions: np.ndarray, initial positions ([x, y]).
            - target_positions: np.ndarray, target positions ([x, y]).
    """
    if mode == "y_upper_lower":
        # y 上下半区域
        range_low_initial = [range_low[0], (range_low[1] + range_high[1]) / 2]
        range_high_initial = [range_high[0], range_high[1]]
        range_low_target = [range_low[0], range_low[1]]
        range_high_target = [range_high[0], (range_low[1] + range_high[1]) / 2]

    elif mode == "x_left_right":
        # x 左右半区域
        range_low_initial = [range_low[0], range_low[1]]
        range_high_initial = [(range_low[0] + range_high[0]) / 2, range_high[1]]
        range_low_target = [(range_low[0] + range_high[0]) / 2, range_low[1]]
        range_high_target = [range_high[0], range_high[1]]

    elif mode == "xy_left_up_right_down":
        # xy 左上和右下区域
        range_low_initial = [range_low[0], (range_low[1] + range_high[1]) / 2]
        range_high_initial = [(range_low[0] + range_high[0]) / 2, range_high[1]]
        range_low_target = [(range_low[0] + range_high[0]) / 2, range_low[1]]
        range_high_target = [range_high[0], (range_low[1] + range_high[1]) / 2]

    elif mode == "xy_right_up_left_down":
        # xy 右上和左下区域
        range_low_initial = [(range_low[0] + range_high[0]) / 2, (range_low[1] + range_high[1]) / 2]
        range_high_initial = [range_high[0], range_high[1]]
        range_low_target = [range_low[0], range_low[1]]
        range_high_target = [(range_low[0] + range_high[0]) / 2, (range_low[1] + range_high[1]) / 2]

    elif mode == "whole_space":
        # 整个区域
        range_low_initial = [range_low[0], range_low[1]]
        range_high_initial = [range_high[0], range_high[1]]
        range_low_target = [range_low[0], range_low[1]]
        range_high_target = [range_high[0], range_high[1]]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Generate initial positions
    initial_positions = generate_points_with_radius(
        range_low_initial, range_high_initial, number, radii, existing_points, existing_radii
    )

    # Combine existing points and radii with newly generated initial positions
    combined_points = (
        np.vstack((existing_points, initial_positions)) if existing_points is not None else initial_positions
    )
    combined_radii = (
        np.vstack((existing_radii, radii * np.ones((len(initial_positions), 1)))) if existing_radii is not None else radii * np.ones((len(initial_positions), 1))
    )

    # Generate target positions
    target_positions = generate_points_with_radius(
        range_low_target, range_high_target, number, radii, combined_points, combined_radii
    )

    return initial_positions, target_positions


def generate_points_with_radius(range_low, range_high, number, radii, existing_points=None, existing_radii=None):
    """
    Generate points such that the distance between any two points (including existing ones) 
    is not less than the sum of their radii.

    Args:
        range_low (list): Lower bounds for x and y dimensions (e.g., [x_min, y_min]).
        range_high (list): Upper bounds for x and y dimensions (e.g., [x_max, y_max]).
        number (int): Number of points to generate.
        radii (float): Radius of each new point.
        existing_points (np.ndarray): Array of existing points to avoid collisions with (optional).
        existing_radii (np.ndarray): Array of existing radii corresponding to existing points (optional).

    Returns:
        np.ndarray: Generated points with shape (number, 2).
    """
    points = []  # Initialize list to store new points
    for _ in range(number):
        while True:
            # Generate a random point within the specified range
            point = np.random.uniform(range_low[:2], range_high[:2])

            # Check distance from this point to all existing points
            valid = True
            if existing_points is not None and existing_radii is not None:
                for i, existing_point in enumerate(existing_points):
                    existing_radius = existing_radii[i, 0]  # Extract radius as scalar
                    distance = np.linalg.norm(point - existing_point)
                    if distance < (radii + existing_radius):  # Sum of radii
                        valid = False
                        break

            # If the point is valid, add it to the list and break the loop
            if valid:
                points.append(point)
                break

    return np.array(points)




def generate_obstacles_and_goals(line_x, line_goal_x, num_obstacles, y_range, min_distance, match_y=True):
    """
    Generate obstacles and goals.

    Parameters:
        line_x (float): The x-coordinate of the line where obstacles are placed.
        line_goal_x (float): The x-coordinate of the line where goals are placed.
        num_obstacles (int): Number of obstacles to generate.
        y_range (tuple): The range of y-coordinates (min_y, max_y).
        min_distance (float): Minimum distance between obstacles on the y-axis.
        match_y (bool): Whether the y-coordinates of goals should match the obstacles.

    Returns:
        x0s (np.ndarray): N x 2 array of initial obstacle positions [(x, y)].
        targets (np.ndarray): N x 2 array of goal positions [(x, y)].
    """
    obstacles = []
    goals = []

    # Generate non-uniform y-coordinates for obstacles
    while len(obstacles) < num_obstacles:
        y = np.random.uniform(y_range[0], y_range[1])

        # Check for minimum distance between obstacles
        if all(abs(y - obs[1]) >= min_distance for obs in obstacles):
            obstacles.append((line_x, y))

    # Generate corresponding goals
    for _, y in obstacles:
        if match_y:
            goals.append((line_goal_x, y))
        else:
            goal_y = np.random.uniform(y_range[0], y_range[1])
            goals.append((line_goal_x, goal_y))

    # Ensure both obstacles and goals are consistent N x 2 arrays
    x0s = np.array([[float(x), float(y)] for x, y in obstacles])
    targets = np.array([[float(x), float(y)] for x, y in goals])

    return x0s, targets


