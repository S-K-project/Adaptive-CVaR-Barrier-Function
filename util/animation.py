#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
from matplotlib.animation import FuncAnimation


def performance_metrics(env, controllers, info, computation_time, filename):
        
    (state_histories, control_inputs_list, h_values_list, cons_values_list,
     _, obs_state_list, _, velocities_list, cost_list) = load_data_from_controller(env, controllers)
    
    metrics = []
    time_step = env.robots[0].dt  
    robot_info = info

    for robot_idx, state_history in enumerate(state_histories):
            
        feasible = robot_info.get("feasible", controllers[robot_idx].feasible)
        
        metric = {
            "robot_id": robot_idx,
            "feasible": feasible,
            "collision": robot_info.get("collision", False),
            "frozen": robot_info.get("frozen", False),
            "reached_goal": robot_info.get("reached_goal", False),
        }

            
        success = feasible and robot_info.get("reached_goal", False) \
                  and (not robot_info.get("collision", False)) and (not robot_info.get("frozen", False))
        metric["success"] = success

        if success:
            x_positions = state_history[0, :]
            y_positions = state_history[1, :]
            robot_radius = env.robots[robot_idx].radius

            diff_x = np.diff(x_positions)
            diff_y = np.diff(y_positions)
            distances = np.sqrt(diff_x**2 + diff_y**2)
            trajectory_length = float(np.sum(distances))

            trajectory_time = float((state_history.shape[1] - 1) * time_step)

            min_safety_margin = float('inf')
            for t in range(state_history.shape[1]):
                for obs_idx, obs in enumerate(env.obstacles):
                    if t < obs_state_list[obs_idx].shape[1]:
                        ox = obs_state_list[obs_idx][0, t]
                        oy = obs_state_list[obs_idx][1, t]
                        r_obs = obs.radius
                        dist = np.sqrt((x_positions[t] - ox)**2 + (y_positions[t] - oy)**2)
                        margin = dist - (robot_radius + r_obs)
                        if margin < min_safety_margin:
                            min_safety_margin = float(margin)

            if len(h_values_list[robot_idx]) > 0:
                min_h_value = float(np.min(h_values_list[robot_idx]))
            else:
                min_h_value = None

            if len(cons_values_list[robot_idx]) > 0:
                min_cons_value = float(np.min(cons_values_list[robot_idx]))
            else:
                min_cons_value = None

            mean_cost = float(np.mean(cost_list[robot_idx]))

        else:
            trajectory_length = None
            trajectory_time = None
            min_safety_margin = None
            min_h_value = None
            min_cons_value = None
            mean_cost = None

            
        if computation_time is not None:
            metric["computation_time"] = computation_time
        else:
            metric["computation_time"] = None
            
        metric.update({
            "trajectory_length": trajectory_length,
            "trajectory_time": trajectory_time,
            "min_safety_margin": min_safety_margin,
            "min_h_value": min_h_value,
            "min_cons_value": min_cons_value,
            "cost": mean_cost
        })

        metrics.append(metric)

    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else o)

    print(f"Metrics saved to {filename}")


def compute_safety_margins(env, state_histories, obs_state_list):
    """
    Compute the safety margin for each robot over time, ensuring the time steps
    match the robot trajectories.
    """
    safety_margin_list = []

    for i, robot_states in enumerate(state_histories):
        T = robot_states.shape[1]  # Get the number of time steps for this robot
        r_robot = env.robots[i].radius

        margins = np.zeros(T)
        for t in range(T):
            # Compute the minimum safety margin for this robot at time t
            min_dist_minus_radii = np.inf
            for j, obs in enumerate(env.obstacles):
                if t < obs_state_list[j].shape[1]:  # Ensure obstacle data exists
                    ox = obs_state_list[j][0, t]
                    oy = obs_state_list[j][1, t]
                    r_obs = obs.radius

                    dist = np.sqrt((robot_states[0, t] - ox)**2 + (robot_states[1, t] - oy)**2)
                    dist_minus_radii = dist - (r_robot + r_obs)

                    if dist_minus_radii < min_dist_minus_radii:
                        min_dist_minus_radii = dist_minus_radii

            margins[t] = min_dist_minus_radii

        safety_margin_list.append(margins)

    return safety_margin_list



def load_data_from_controller(env, controllers = []):
    """
    Returns:
        state_histories, control_inputs_list, h_values_list, beta_values_list,
        obs_state_list, nominal_inputs_list, velocities_list
    """
    state_histories = [np.hstack(robot.xlog) for robot in env.robots]

    control_inputs_list = [np.hstack(robot.ulog) for robot in env.robots]
            
    h_values_list = []
    for controller in controllers:
        # Compute the minimum for each entry in hlog independently
        h_values = [np.min(entry) for entry in controller.hlog]
        h_values_list.append(h_values)
    cons_values_list = []
    for controller in controllers:
        cons_values = [np.min(entry) for entry in controller.cons_log]
        cons_values_list.append(cons_values)
            
    if hasattr(controllers[0], "beta_log") and len(controllers[0].beta_log) > 0:
        beta_values_list = []
        for controller in controllers:
            beta_values = [np.min(entry) for entry in controller.beta_log]
            beta_values_list.append(beta_values)
    else:
        beta_values_list = [np.zeros((len(controller.hlog))) for controller in controllers]


    obs_state_list = [np.hstack(obs.trajectory) for obs in env.obstacles]

    if len(env.robots[0].un_log) > 0:
        nominal_inputs_list = [np.hstack(robot.un_log) for robot in env.robots]
    else:
        nominal_inputs_list = [np.zeros((2, 1)) for _ in env.robots]

    # 机器人速度
    velocities_list = []
    if env.robots[0].type == "doubleint":
        for state_history in state_histories:
            velocities = state_history[2:4, :] # vx, vy
            velocities_list.append(velocities)
    elif env.robots[0].type == "unicycle_v2":
        for state_history in state_histories:
            velocities = state_history[3, :] # v
            thetas = state_history[2, :] # theta
            vx = velocities * np.cos(thetas)
            vy = velocities * np.sin(thetas)
            velocities_list.append(np.vstack((vx,vy)))
        
    if hasattr(controllers[0], "cost_log") and len(controllers[0].cost_log) > 0:
        cost_list = [np.hstack(controller.cost_log) for controller in controllers]
    else:
        cost_list = [np.zeros((1, 1)) for _ in env.robots]
    
    return (state_histories, control_inputs_list, h_values_list, beta_values_list, cons_values_list,
            obs_state_list, nominal_inputs_list, velocities_list, cost_list)

def load_data_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 机器人轨迹（假设只有一个机器人），转换为 shape (2, num_steps)
    robot_traj = np.array(data["robot"]).T  # 例如：[[x1, x2, ...], [y1, y2, ...]]
    state_histories = [robot_traj]
    
    # 障碍物轨迹：所有 key 以 "human" 开头的视为障碍物轨迹
    obs_state_list = []
    for key in data:
        if key.startswith("human"):
            traj = np.array(data[key]).T  # shape (2, num_steps)
            velocities = np.zeros_like(traj)
            # obs_state = np.vstack((traj, velocities))
            # obs_state_list.append(obs_state) 
            obs_state_list.append(traj)
    
    num_steps = robot_traj.shape[1]
    control_inputs_list = [np.zeros((2, num_steps))]
    h_values_list = [np.zeros(num_steps)]
    beta_values_list = [np.zeros(num_steps)]
    cons_values_list = [np.zeros(num_steps)]
    nominal_inputs_list = [np.zeros((2, num_steps))]
    velocities_list = [np.zeros((2, num_steps))]
    cost_list = [np.zeros(num_steps)]
    
    return (state_histories, control_inputs_list, h_values_list, beta_values_list, 
            cons_values_list, obs_state_list, nominal_inputs_list, velocities_list, cost_list)

def create_figure_and_axes():
    """
    Create a figure and subplots for the animation.
    """
    fig = plt.figure(figsize=(12, 8))
    # Adjust spacing and margins
    fig.subplots_adjust(
        left=0.09,    # Left margin (fraction of figure width)
        right=0.95,   # Right margin
        top=0.95,     # Top margin
        bottom=0.05,  # Bottom margin
        wspace=0.2,   # Space between columns
        hspace=0.2    # Space between rows
    )

    gs = fig.add_gridspec(4, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1, 1])

    ax_left = fig.add_subplot(gs[0:2, 0])  
    ax_cost = fig.add_subplot(gs[3, 0])

    ax_beta = fig.add_subplot(gs[2, 0])    
    ax_h = fig.add_subplot(gs[0:2, 1])      
    ax_margin = fig.add_subplot(gs[2, 1])    
    ax_u = fig.add_subplot(gs[3, 1])     

    # ax_factor = fig.add_subplot(gs[3, :])  # bottom row, spanning both columns
    # ax_cost = fig.add_subplot(gs[3, :])  # bottom row, spanning both columns
    gs.update(wspace=0.2, hspace=0.5)


    return fig, ax_left, ax_beta, ax_h, ax_margin, ax_u, ax_cost


def init_obstacles(ax_left, obs_state_list, env):
    obstacle_patches = []
    obs_patches_original = []
    obstacle_texts = []

    for i, obs in enumerate(env.obstacles):
        initial_x = obs_state_list[i][0, 0]
        initial_y = obs_state_list[i][1, 0]
        radius = obs.radius

        # 动态膨胀圆（初始与原始半径相同，颜色更浅）
        obstacle_circle = plt.Circle((initial_x, initial_y), radius, 
                                     color='grey', alpha=0.3)
        ax_left.add_patch(obstacle_circle)
        obstacle_patches.append(obstacle_circle)

        # 原始圆形
        obs_orig = plt.Circle(
            (initial_x, initial_y),
            obs.radius,
            color=obs.color,
            alpha=0.3,
            fill=True
        )
        ax_left.add_patch(obs_orig)
        obs_patches_original.append(obs_orig)

        obstacle_texts= []
        
    return obstacle_patches, obs_patches_original, obstacle_texts

def init_uncertainty(ax_left, env, controllers):
    obstacle_uncert_scatter = []
    obstacle_uncert_circles = []
    # Uncertainty scatter and circle
    for i, obs in enumerate(env.obstacles):
        samples = controllers[0].obs_wx_samples[i]
        if samples.size > 0:
            uncert_scatter = ax_left.scatter(samples[:, 0], samples[:, 1],
                                                color='red', s=10, label='Uncertainty', alpha=0.5)
            center = np.mean(samples, axis=0)
            distances = np.linalg.norm(samples - center, axis=1)
            uncert_radius = distances.max()
        else:
            uncert_scatter = ax_left.scatter([], [], color='red', s=10, label='Uncertainty', alpha=0.5)
            center = (0, 0)
            uncert_radius = 0
        uncert_circle = Circle(center, uncert_radius, edgecolor='red', facecolor='red',
                                alpha=0.3, fill=True, lw=2)
        ax_left.add_patch(uncert_circle)

        obstacle_uncert_scatter.append(uncert_scatter)
        obstacle_uncert_circles.append(uncert_circle)
    
    return obstacle_uncert_scatter, obstacle_uncert_circles

def init_robots(ax_left, env):
    scatters = []
    circles = []
    circles_original = []

    for i, robot in enumerate(env.robots):
        # initial and target positions
        ax_left.plot(robot.x0[0], robot.x0[1], 'yo', label=f"Start",markersize=8, alpha=0.6)
        ax_left.plot(robot.target[0], robot.target[1], 'y^', 
                     label=f"Goal", markersize=8, alpha=0.6)

        scat = ax_left.scatter([], [], s=20, color='blue', alpha =0.4)  # Use 'color' instead of 'c'

        scatters.append(scat)

        # dynamic radius circle (initially same as original radius, but lighter color)
        radius_circle = plt.Circle((robot.x0[0], robot.x0[1]), robot.radius, 
                                   color='lightblue', alpha=0.5)
        ax_left.add_patch(radius_circle)
        circles.append(radius_circle)

        # original circle
        circle_orig = plt.Circle((robot.x0[0], robot.x0[1]), robot.radius, 
                                 color='gray', alpha=0.3, fill=True)
        ax_left.add_patch(circle_orig)
        circles_original.append(circle_orig)

    return scatters, circles, circles_original

def init_lines(ax_h, ax_margin, ax_u, ax_beta, h_values_list, safety_margin_list, beta_values_list, env, ax_cost=None):
    """
    Initialize the lines for h, velocity, control inputs, and beta values.
    """
    lines_h = []
    lines_margin = []
    lines_u1 = []
    lines_u2 = []
    lines_beta = []
    lines_beta_u = []
    lines_beta_u_bar = []

    lines_nom_u1 = []
    lines_nom_u2 = []
    lines_cost = []

    for i in range(len(env.robots)):
        colors = ['#ef8a62', '#67a9cf']

        # min h
        line_h, = ax_h.plot([], [],linewidth=2,color = 'grey',alpha= 0.9)
        lines_h.append(line_h)
        # min beta
        line_beta, = ax_beta.plot([], [],linewidth= 2,color = 'orange',alpha= 0.9, label = r"$\beta_k$")
        lines_beta.append(line_beta)
        # min beta_u
        line_beta_u, = ax_beta.plot([], [],linewidth= 1,color = 'green',alpha= 0.9, linestyle='--', label = r"$\beta_u$")
        lines_beta_u.append(line_beta_u)
        # min beta_u_bar
        # line_beta_u_bar, = ax_beta.plot([], [],linewidth= 1,color = 'red',alpha= 0.9, linestyle='--', label = r"$\bar{\beta}_u$")
        line_beta_u_bar, = ax_beta.plot([], [],linewidth= 1,color = 'red',alpha= 0.9, linestyle='--')
        lines_beta_u_bar.append(line_beta_u_bar)

        # velocity
        line_margin, = ax_margin.plot([], [],linewidth= 2,color = 'grey',alpha= 0.9)
        lines_margin.append(line_margin)

        # control inputs
        line_u1, = ax_u.plot([], [], linewidth= 2, label=f'$u_x$',color=colors[0],alpha= 0.9)
        line_u2, = ax_u.plot([], [], linewidth= 2, label=f'$u_y$',color=colors[1],alpha= 0.9)
        
        lines_u1.append(line_u1)
        lines_u2.append(line_u2)
    
        line_nom_u1, = ax_u.plot([], [], '--')
        line_nom_u2, = ax_u.plot([], [], '--')
        lines_nom_u1.append(line_nom_u1)
        lines_nom_u2.append(line_nom_u2)
        
        if ax_cost is not None:
            line_cost, = ax_cost.plot([], [])
            lines_cost.append(line_cost)
        
        ax_h.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
        ax_margin.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
        ax_u.axhline(y=env.robots[0].u_min[0], color='r', linestyle='--', linewidth=0.8)
        ax_u.axhline(y=env.robots[0].u_max[0], color='r',linestyle='--',linewidth=0.8)
 
    
    return (lines_h, lines_margin, lines_u1, lines_u2,
            lines_beta, lines_beta_u, lines_beta_u_bar, lines_nom_u1, lines_nom_u2, lines_cost)
    
def set_axes_and_legends(ax_left, ax_h, ax_margin, ax_u, ax_beta, env, ax_cost=None):
    ax_left.set_aspect('equal', adjustable='box')

    ax_left.set_xlim(env.mapsize[0]-0.2, env.mapsize[1]+0.2)
    ax_left.set_ylim(env.mapsize[2], env.mapsize[3])
    ax_left.set_xlabel("x (m)", labelpad=0, fontsize=12)
    ax_left.set_ylabel("y (m)", labelpad=0, fontsize=12)
    # ax_left.set_title("Robot Position", fontsize=12)
    ax_left.grid(False)

    # ax_h.set_aspect('auto')
    pos_left = ax_left.get_position()
    pos_h = ax_h.get_position()
    ax_h.set_position([pos_h.x0, pos_left.y0, pos_h.width, pos_left.height])

    # ax_h.set_title('CBF Value',fontsize=12)
    ax_h.set_xlabel('Time Step', labelpad=0,fontsize=12)
    ax_h.set_ylabel('h(x_t)', labelpad=0,fontsize=12)
    # ax_h.legend()
    ax_h.grid(False)

    ax_margin.set_title('Closest Dist to Obs', fontsize=12)
    ax_margin.set_xlabel('Time Step', labelpad=0, fontsize=12)
    ax_margin.set_ylabel('Closest Dist to Obs', labelpad=0, fontsize=12)
    # ax_margin.legend()
    ax_margin.grid(False)

    ax_u.set_title('Control Inputs', fontsize=12)
    ax_u.set_xlabel('Time Step', labelpad=0, fontsize=12)
    ax_u.set_ylabel('Control Input', labelpad=0, fontsize=12)
    # ax_u.legend()
    ax_u.grid(False)

    ax_beta.set_title('Risk Level', fontsize=12)
    ax_beta.set_xlabel('Time Step', labelpad=0, fontsize=12)
    ax_beta.set_ylabel(r'Risk Level $\beta_k$', labelpad=0, fontsize=12)
    ax_beta.legend()
    ax_beta.grid(False)
    
    if ax_cost is not None:
        ax_cost.set_title('Cost Over Time', fontsize=12)    
        ax_cost.set_xlabel('Time Step', labelpad=0, fontsize=12)
        ax_cost.set_ylabel('Cost Value', labelpad=0, fontsize=12)
        # ax_cost.legend()
        ax_cost.grid(False)

def set_y_axis_limits(ax_h, ax_margin, ax_u, ax_beta,
                      h_values_list, safety_margin_list,
                      control_inputs_list, beta_values_list, cost_list, ax_cost=None):
    ax_h.set_ylim(-0.5, 15.0)
    ax_margin.set_ylim(-0.5, 15.0)
    ax_u.set_ylim(-4, 4)
    # ax_beta.set_ylim(-0.1, 0.5)
    ax_beta.set_ylim(-0.1, 1.1)
    if ax_cost is not None:
        cost_min = min([np.min(cost) for cost in cost_list])
        cost_max = max([np.max(cost) for cost in cost_list])
        ax_cost.set_ylim(cost_min - 0.1 * abs(cost_min), cost_max + 0.1 * abs(cost_max))
       
def compute_influence_factors(env, num, compute_factor_and_params):
    """
    Compute the influence factors between robots and obstacles.

    For each robot, the function finds the obstacle that produces the highest influence
    factor, and for each obstacle, the function finds the robot that produces the highest
    influence factor.

    Parameters
    ----------
    env : object
        The environment object that contains lists of robots and obstacles.
    num : int
        The current frame number.
    compute_factor_and_params : function
        A function that computes the influence factor and additional parameters given a
        robot index, an obstacle index, and a time index. It should return a tuple whose
        first element is the factor.

    Returns
    -------
    robot_factors : list of float
        The maximum influence factor for each robot.
    robot_best_obs : list of int
        The index of the obstacle that has the most influence on each robot.
    obstacle_factors : list of float
        The maximum influence factor for each obstacle.
    obstacle_best_robot : list of int
        The index of the robot that has the most influence on each obstacle.
    """
    # Use time_index = num - 1 (or 0 if num is 0) for indexing into your state histories.
    time_index = num - 1 if num > 0 else 0

    # 1) For each robot, find the obstacle that has the most influence.
    robot_factors = [1.0] * len(env.robots)
    robot_best_obs = [-1] * len(env.robots)
    for i in range(len(env.robots)):
        best_factor = 0.0
        best_j = -1
        for j in range(len(env.obstacles)):
            # compute_factor_and_params is expected to return a tuple,
            # where the first element is the factor.
            f, _, _, _ = compute_factor_and_params(i, j, time_index)
            if f > best_factor:
                best_factor = f
                best_j = j
        robot_factors[i] = best_factor
        robot_best_obs[i] = best_j

    # 2) For each obstacle, find the robot that has the most influence.
    obstacle_factors = [1.0] * len(env.obstacles)
    obstacle_best_robot = [-1] * len(env.obstacles)
    for j in range(len(env.obstacles)):
        best_factor = 0.0
        best_i = -1
        for i in range(len(env.robots)):
            f, _, _, _ = compute_factor_and_params(i, j, time_index)
            if f > best_factor:
                best_factor = f
                best_i = i
        obstacle_factors[j] = best_factor
        obstacle_best_robot[j] = best_i

    return robot_factors, robot_best_obs, obstacle_factors, obstacle_best_robot

def compute_closest_obstacle_factors(env, num, compute_factor_and_params, state_histories, obs_state_list):
    """
    For each robot, find the obstacle that is closest (in Euclidean distance)
    and compute the corresponding factor using the provided compute_factor_and_params
    function. Also, for each obstacle, find the closest robot and compute its factor.
    
    Parameters
    ----------
    env : object
        The environment object that contains lists of robots and obstacles.
    num : int
        The current frame number.
    compute_factor_and_params : function
        A function that computes the factor (and additional parameters) given a
        robot index, an obstacle index, and a time index. It should return a tuple
        whose first element is the factor.
    state_histories : list of np.ndarray
        The state history arrays for each robot. It is assumed that, for each robot,
        state_histories[i][0, t] and state_histories[i][1, t] give the x and y positions.
    obs_state_list : list of np.ndarray
        The state history arrays for each obstacle. It is assumed that, for each obstacle,
        obs_state_list[j][0, t] and obs_state_list[j][1, t] give the x and y positions.
    
    Returns
    -------
    robot_closest_factors : list of float
        The computed factor for the closest obstacle for each robot.
    robot_closest_obs : list of int
        The index of the closest obstacle for each robot.
    obstacle_closest_factors : list of float
        The computed factor for the closest robot for each obstacle.
    obstacle_closest_robot : list of int
        The index of the closest robot for each obstacle.
    """
    # Use time_index = num - 1 (or 0 if num is 0) for indexing into the state histories.
    time_index = num - 1 if num > 0 else 0

    # For each robot, find the obstacle that is closest (minimum Euclidean distance)
    robot_closest_factors = [1.0] * len(env.robots)
    robot_closest_obs = [-1] * len(env.robots)
    robot_closest_distances = [float('inf')] * len(env.robots)
    robot_delta = [0.0] * len(env.robots)

    for i in range(len(env.robots)):
        # Get the current position of robot i.
        robot_x = state_histories[i][0, time_index]
        robot_y = state_histories[i][1, time_index]
        for j in range(len(env.obstacles)):
            # Get the current position of obstacle j.
            obs_x = obs_state_list[j][0, time_index]
            obs_y = obs_state_list[j][1, time_index]
            # Compute Euclidean distance between robot and obstacle.
            distance = np.sqrt((robot_x - obs_x) ** 2 + (robot_y - obs_y) ** 2)
            if distance < robot_closest_distances[i]:
                robot_closest_distances[i] = distance
                robot_closest_obs[i] = j
                # Compute the factor for this robot-obstacle pair.
                factor, _, _, Delta = compute_factor_and_params(i, j, time_index)
                robot_closest_factors[i] = factor
                robot_delta[i] = Delta

    # Similarly, for each obstacle, find the closest robot.
    obstacle_closest_factors = [1.0] * len(env.obstacles)
    obstacle_closest_robot = [-1] * len(env.obstacles)
    obstacle_closest_distances = [float('inf')] * len(env.obstacles)

    for j in range(len(env.obstacles)):
        obs_x = obs_state_list[j][0, time_index]
        obs_y = obs_state_list[j][1, time_index]
        for i in range(len(env.robots)):
            robot_x = state_histories[i][0, time_index]
            robot_y = state_histories[i][1, time_index]
            distance = np.sqrt((robot_x - obs_x) ** 2 + (robot_y - obs_y) ** 2)
            if distance < obstacle_closest_distances[j]:
                obstacle_closest_distances[j] = distance
                obstacle_closest_robot[j] = i
                factor, _, _, _ = compute_factor_and_params(i, j, time_index)
                obstacle_closest_factors[j] = factor

    return (robot_closest_factors, robot_closest_obs,
            obstacle_closest_factors, obstacle_closest_robot, robot_delta)
    


def animate(env, controllers=[], filename=None, save_ani=True):
    """
    Animate the trajectories of robots and obstacles in the environment.

    Parameters
    ----------
    env : object
        Environment object that contains robots, obstacles, and map size.
    controllers : list
        A list of controller objects for each robot.
    filename : str
        The name of the output .mp4 file to save the animation.
    save_ani : bool
        Whether to save the animation (mp4 and gif). If False, just visualize without saving.
    """
    (state_histories, control_inputs_list, h_values_list, beta_values_list, cons_values_list,
        obs_state_list, nominal_inputs_list, velocities_list, cost_list) = load_data_from_controller(env, controllers)
 
        
    # Compute safety margins
    safety_margin_list = compute_safety_margins(env, state_histories, obs_state_list)
    
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1])  
    ax_left   = fig.add_subplot(gs[:, 0])
    ax_beta   = fig.add_subplot(gs[0, 1])   # row: 0, col: 1
    ax_u      = fig.add_subplot(gs[1, 1])   # row: 1, col: 1
    ax_h      = fig.add_subplot(gs[0, 2])   # row: 0, col: 2
    ax_margin = fig.add_subplot(gs[1, 2])   # row: 1, col: 2

    plt.subplots_adjust(
        left=0.08,   # 整个图像左边缘
        right=0.95,  # 整个图像右边缘
        top=0.90,    # 整个图像上边缘
        bottom=0.15, # 整个图像下边缘
        wspace=0.25, # 子图之间水平间隔
        hspace=0.35  # 子图之间垂直间隔
    )

    obstacle_patches, obs_patches_original, obstacle_texts = init_obstacles(ax_left, obs_state_list, env)
    scatters, circles, circles_original = init_robots(ax_left, env)
    
    (lines_h, lines_margin, lines_u1, lines_u2, 
     lines_beta, lines_beta_u, lines_beta_u_bar, lines_nom_u1, lines_nom_u2,
     lines_cost) = init_lines(
         ax_h, ax_margin, ax_u, ax_beta,
         h_values_list, safety_margin_list, beta_values_list,
         env,
         ax_cost=None   
    )
     
    ax_left.set_aspect('equal', adjustable='box')
    ax_left.set_xlim(env.mapsize[0]-0.2, env.mapsize[1]+0.2)
    ax_left.set_ylim(env.mapsize[2], env.mapsize[3])
    ax_left.set_xlabel("x (m)", labelpad=0, fontsize=12)
    ax_left.set_ylabel("y (m)", labelpad=0, fontsize=12)
    ax_left.legend(loc='upper right',fontsize=10)
    ax_left.grid(False)

    # ax_h.set_xlabel('Time Step', labelpad=0,fontsize=12)
    ax_h.set_ylabel(r'CBF Value $h_k$', fontsize=12)
    # ax_h.legend()
    ax_h.grid(False)

    # ax_margin.set_title('Closest Distance to obstacle', fontsize=12)
    ax_margin.set_xlabel('Time Step', fontsize=12)
    ax_margin.set_ylabel('Closest Dist. to Obs.', fontsize=12)
    # ax_margin.legend()
    ax_margin.grid(False)
    

    # ax_u.set_title('Control Inputs Over Time', fontsize=12)
    ax_u.set_xlabel('Time Step', fontsize=12)
    ax_u.set_ylabel('Control Input', fontsize=12)
    ax_u.legend()
    ax_u.grid(False)

    # ax_beta.set_title('Risk Level', fontsize=12)
    # ax_beta.set_xlabel('Time Step', labelpad=0, fontsize=12)
    ax_beta.set_ylabel(r'Risk Level $\beta_k$',fontsize=12)
    ax_beta.legend()
    ax_beta.grid(False)
    
    set_y_axis_limits(ax_h, ax_margin, ax_u, ax_beta,
                      h_values_list, safety_margin_list, 
                      control_inputs_list, beta_values_list, 
                      cost_list)

    def init_func():
        for scat, circle in zip(scatters, circles):
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array([])
            circle.set_center((0, 0))
        
        for line_h in lines_h:
            line_h.set_data([], [])
        for line_beta, line_beta_u, line_beta_u_bar in zip(lines_beta, lines_beta_u, lines_beta_u_bar):
            line_beta.set_data([], [])
            line_beta_u.set_data([], [])
            line_beta_u_bar.set_data([], [])
            
        for line_margin_obj in lines_margin:
            line_margin_obj.set_data([], [])
            
        for line_u1_obj, line_u2_obj in zip(lines_u1, lines_u2):
            line_u1_obj.set_data([], [])
            line_u2_obj.set_data([], [])

        for ln_u1, ln_u2 in zip(lines_nom_u1, lines_nom_u2):
            ln_u1.set_data([], [])
            ln_u2.set_data([], [])

        for idx, obs_patch in enumerate(obstacle_patches):
            obs_x = obs_state_list[idx][0, 0]
            obs_y = obs_state_list[idx][1, 0]
            obs_patch.center = (obs_x, obs_y)
            obs_patches_original[idx].center = (obs_x, obs_y)
            # obstacle_texts[idx].set_position((obs_x, obs_y))
            
        for line_cost in lines_cost:
            line_cost.set_data([], [])
        
        return ( lines_h + lines_margin + lines_u1 + lines_u2   
                + lines_beta + lines_beta_u + lines_beta_u_bar + scatters + circles
                + obstacle_patches + circles_original
                + obs_patches_original + lines_nom_u1 + lines_nom_u2
                + obstacle_texts + lines_cost )
        
    def update_frame(num):
        max_num = max([s.shape[1] for s in state_histories])

        def compute_factor_and_params(robot_idx, obs_idx, t):
            if (t < state_histories[robot_idx].shape[1] and 
                t < obs_state_list[obs_idx].shape[1]):
                if controllers[robot_idx].htype == "dist_cone":
                    dot_product = env.robots[robot_idx].collision_cone_value(state_histories[robot_idx][:, t], obs_state_list[obs_idx][:, t])
                    w = np.linalg.norm(obs_state_list[obs_idx][2:4, t]) / env.robots[robot_idx].v_obs_est
                    theta_value = (1.0 / env.robots[robot_idx].rate) * np.log(1 + np.exp(-env.robots[robot_idx].rate * dot_product)) 
                    factor = np.sqrt(1.0 + w * theta_value)
                    return factor, dot_product, 0.0, w * theta_value
                else:
                    return (1.0, 0.0, 0.0, 0.0)
            else:
                return (1.0, 0.0, 0.0, 0.0)

        (robot_factors, robot_best_obs,
         obstacle_factors, obstacle_best_robot, robot_delta) = compute_closest_obstacle_factors(
            env, num, compute_factor_and_params, state_histories, obs_state_list)
        
        for i in range(len(env.robots)):
            x = state_histories[i][0, :]
            y = state_histories[i][1, :]
            offsets = np.column_stack((x[:num], y[:num]))
            scatters[i].set_offsets(offsets)
            scatters[i].set_array(np.linspace(0, 1, num))

            if num > 0:
                circles[i].set_center((x[num - 1], y[num - 1]))
                circles_original[i].set_center((x[num - 1], y[num - 1]))
                if robot_best_obs[i] != -1:
                    r_eff_robot = env.robots[i].radius + (env.robots[i].radius + env.obstacles[robot_best_obs[i]].radius) * (robot_factors[i] -1)
                else:
                    r_eff_robot = env.robots[i].radius
                circles[i].set_radius(r_eff_robot)
                
        for idx, obs_patch in enumerate(obstacle_patches):
            obs_patch.set_facecolor(env.obstacles[idx].color)
            # obstacle_texts[idx].set_color('black') 
            
            if num < obs_state_list[idx].shape[1]:
                obs_x = obs_state_list[idx][0, num]
                obs_y = obs_state_list[idx][1, num]
                obs_patch.center = (obs_x, obs_y)
                obs_patches_original[idx].center = (obs_x, obs_y)
                # obstacle_texts[idx].set_position((obs_x, obs_y))
            r_eff_obs = env.obstacles[idx].radius
            obs_patch.set_radius(r_eff_obs)

        previous_robot_delta = [0.0 for _ in range(1)]  # 根据你的机器人数量初始化

        for i in range(len(env.robots)):
            best_j = robot_best_obs[i]
            if best_j != -1:
                obstacle_patches[best_j].set_facecolor('red')
                obstacle_patches[best_j].set_alpha(0.3)
                # obstacle_texts[best_j].set_color('red')
                
            xdata = np.arange(num)
            h_vals = h_values_list[i][:num]
            lines_h[i].set_data(xdata, h_vals)
            beta_vals = beta_values_list[i][:num]
            lines_beta[i].set_data(xdata, beta_vals)
            margin = safety_margin_list[i][:num]
            lines_margin[i].set_data(xdata, margin)

            if controllers[i].type == "cvar" or controllers[i].type == "adap_cvarbf":
                if controllers[i].htype in ["dist_cone"]:
                    beta_u = np.full(num, 0.5) 
                    lines_beta_u[i].set_data(xdata, beta_u)
                
                elif controllers[i].htype in ["dist"]:
                    beta_u = np.full(num, 0.5) 
                    lines_beta_u[i].set_data(xdata, beta_u)


            u1 = control_inputs_list[i][0, :num]
            u2 = control_inputs_list[i][1, :num]
            lines_u1[i].set_data(xdata, u1)
            lines_u2[i].set_data(xdata, u2)


        ax_h.set_xlim(0, max_num)
        ax_margin.set_xlim(0, max_num)
        ax_u.set_xlim(0, max_num)
        ax_beta.set_xlim(0, max_num)
        ax_beta.legend()   

        
        return (lines_h + lines_margin + lines_u1 + lines_u2
                + lines_beta + lines_beta_u + lines_beta_u_bar + scatters + circles + circles_original
                + obstacle_patches + obs_patches_original
                + lines_nom_u1 + lines_nom_u2 + obstacle_texts
                + lines_cost)
        
    num_frames = min([s.shape[1] for s in state_histories])


    ani = FuncAnimation(
        fig,
        update_frame,
        frames=num_frames ,
        init_func=init_func,
        blit=False,
        interval=50,
        repeat=False
    )

    if save_ani and filename:
        filename = filename + ".gif"
        ani.save(filename, writer='pillow', fps=25)
        # png_filename = filename.replace('.gif', '_lastframe.jpg')
        # plt.savefig(png_filename, dpi=100)  

    plt.close(fig)


