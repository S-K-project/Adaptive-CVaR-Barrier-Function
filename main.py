import os
import click
from environment.env_base import EnvironmentBase
from controller.cbf_controller_nlp import DCLFDCBF
from controller.cvar_cbf_controller_nlp import DCLFCVARDCBF
from controller.cvar_cbf_controller_nlp_beta_dt import DCLFCVARDCBF as DCLFCVARDCBFMPCBETADT
from lib.sfm import SFM
import time
from util.util import *
from util.animation import *


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--htype", "htype", type=str, default='dist_cone', show_default=True, help="Type of h function") # cone, vel, dist, dist_cone 
@click.option("--S", "S", type=int, default=15, show_default=True, help="Number of samples in the uncertainty distribution")
@click.option("--beta", "beta", type=float, default=0.99, show_default=True, help="Risk aversion parameter") # 0.1 for cvar risk aversion, 0.9 for cvar risk neutral
@click.option("--show-ani", "show_ani", type=bool, default=False, show_default=True, help="Show animation")
@click.option("--save-ani", "save_ani", type=bool, default=True, show_default=True, help="Save animation")
@click.option("--time-total", "time_total", type=float, default=50.0, show_default=True, help="Total simulation time")
@click.option("--ctrl-type", "ctrl_type", type=str, default="adap_cvarbf", show_default=True, help="Controller type") #cbf, cvar, rcbf, adap_cvarbf

def main(htype, S, beta, show_ani, save_ani, time_total, ctrl_type):
    
    params = {
        "htype": htype,
        "beta": beta,
        "S": S,
    }

    # config_folder = SCRIPT_DIR +'/config/' + 'demo_case/' 
    # config_folder = SCRIPT_DIR +'/config/' + 'obstacles_20_noise0.05/seed14_noise0.05_obs8_umax0.9_besfm_umax0.9_besfm/' 
    config_folder = SCRIPT_DIR +'/config/' + 'video20obs/seed13_noise0.025_obs8_umax0.9_besfm_umax0.3_besfm/' 
    # config_folder = SCRIPT_DIR +'/config/' + 'video15obs/seed4_noise0.05_obs5_umax0.9_besfm_umax0.9_besfm/' 
    # config_folder = SCRIPT_DIR +'/config/' + 'one_obs/' 
 
    
    figure = 'figures/' # ctrl beta htype
    figures_folder = os.path.join(config_folder, figure)
    os.makedirs(figures_folder, exist_ok=True)

    file = 'config' # config config_1obs_dyn
    env = EnvironmentBase(config_file = config_folder + file + '.yaml') 

    controllers = create_controllers(env, ctrl_type, params)
    obs_controllers = create_obs_controller(env)
    
    done = False
    while not done:
        start_time = time.time()
        for robot, controller in zip(env.robots, controllers):
            
            loc_obstacles = robot.update_local_objects(env.obstacles, env.robots, R_sensing=5)
            xk, uk, status = controller.solve_opt(env.t_curr, loc_obstacles, env.robots)
            
            log_info(f"{status} at time {env.t_curr}")
            if uk is not None:
                robot.step(xk, uk)
                print(f"Robot {robot.id} - Time Step {env.t_curr}:")
                print(f"  x_t = {robot.x_curr.flatten()}")
                print(f"  u_t = {uk.flatten()}")
                print(f"  h_t = {controller.hlog[-1].flatten()}")
            else:
                log_error(f"Optimization failed for Robot {robot.id}. Exiting simulation.")
                controller.feasible = False

                    
            end_time = time.time()
            computation_time = end_time - start_time
            log_info(f"Solver Computation Time: {computation_time:.6f} seconds")
            
        for obs, obs_controller in zip(env.obstacles, obs_controllers):
            ob_obs = obs.observation(env.obstacles, env.robots, obs_controller.type)
            xok, uok = obs_controller.solve_opt(ob_obs)
            obs.step(uk = uok)
 
        env.t_curr += env.dt  # Use the time step from the first robot

        done, info = env.done(env.t_curr, time_total, controller.feasible)
        if done:
            print(info)
            
    end_time = time.time()
    computation_time = (end_time - start_time) / (env.t_curr / env.dt)

    if save_ani:
        filename = os.path.join(
            figures_folder, f"{ctrl_type}_beta{beta}_h{htype}"
            )  
 
        animate(env, controllers, None, filename, save_ani=save_ani, show_ani=show_ani)
        performance_metrics(env, controllers, info, computation_time, filename)


def create_controllers(env, ctrl_type, params):
    """
    Create controllers based on the type.
    """
    controller_mapping = {
        "cvar": DCLFCVARDCBF, # robot/obs uncertain
        "cbf": DCLFDCBF, # no uncertain
        "adap_cvarbf": DCLFCVARDCBFMPCBETADT, # robot/obs uncertain
    }

    if ctrl_type not in controller_mapping:
        raise ValueError(f"Unknown controller type: {ctrl_type}")

    controllers = [
        controller_mapping[ctrl_type](robot, env.obstacles, env.robots, params)
        for robot in env.robots
    ]
    return controllers


def create_obs_controller(env):
    """
    Create controllers based on the type.
    """
    controller_mapping = {
        "sfm" : SFM, 
    }
    obs_controllers = []
    for obs in env.obstacles:   
        if obs.behavior not in controller_mapping:
            raise ValueError(f"Unknown controller type: {obs.behavior}")
        else:
            obs_controllers.append( 
                controller_mapping[obs.behavior](obs, env.obstacles, env.robots)
            )
        
    return obs_controllers

if __name__ == "__main__":
    main()