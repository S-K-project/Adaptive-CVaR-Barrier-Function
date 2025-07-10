import math
from environment.state_plus import ObservableState, FullState
import numpy as np
from scipy.stats import norm
import casadi as ca
def _pdf_or_ones(values, std):
    """
    Returns norm.pdf(values, 0, std) if std > 0,
    otherwise an array of ones.
    """
    if std == 0:
        return np.ones((len(values), 1))
    else:
        return norm.pdf(values, 0, std).reshape(-1,1)
        

class ObjectBase:
    def __init__(self, x0, radius, dt, noise, id, target, seed =1):
        """
        Initialize the base robot and obstacle class with general attributes.
        """
        seed_obj = seed+id
        self.rng = np.random.default_rng(seed_obj)  
        
        self.dt = dt           # Time step
        self.id = id  # Assign a unique ID
        
        self.target = np.array(target, dtype=float).reshape(-1, 1)
        self.x0 = np.array(x0).reshape(-1, 1)  # Initial state vector
        self.noise = noise
        self.radius = radius

        self.m = None             # Control input dimension
        self.n = None             # State dimension
        
        self.u = None          # Control input vector
    
        self.x_min = None     
        self.x_max = None
        self.u_min = None         
        self.u_max = None
        
        self.xlog = []
        self.ulog = []
        self.un_log = []
        

        # self.u_cost = 0
        self.x_curr = self.x0.astype(float)  # Current state vector
        self.velocity_xy = np.zeros((2, 1))

    
    def dynamics(self):
        raise NotImplementedError("Dynamics method must be implemented in subclass.")
                                  
    def reset(self):
        raise NotImplementedError("Reset method must be implemented in subclass.")

    def nominal_input(self):
        raise NotImplementedError("Nominal input method must be implemented in subclass.")

    def step(self, x_k1=None, uk=None):
        raise NotImplementedError("Step method must be implemented in subclass.")

    def u_disturbance(self, control_input):
        if max(self.noise[1]) <= 1e-5 or isinstance(control_input, ca.MX): # dont add noise for x_next in optimization process
            return np.zeros((self.m, 1))
        else:
            std_ux = np.sqrt(self.noise[1][0]*(control_input[0, 0]**2) + self.noise[1][1]*(control_input[1,0]**2))
            std_uy = np.sqrt( self.noise[1][2]*(control_input[0, 0]**2) +  self.noise[1][3]*(control_input[1,0]**2))
            sample = self.rng.normal( [[0], [0]], scale=[[std_ux], [std_uy]])
            return  sample.reshape(-1,1)

    def x_disturbance(self, state):
        if max(self.noise[0]) <= 1e-5 or isinstance(state, ca.MX): # dont add noise for x_next in optimization process
            return np.zeros((self.n, 1))
        else:
            if state.shape[0] == 2:
                std_px = self.noise[0][0]
                std_py = self.noise[0][1]
                sample = self.rng.normal( [[0], [0]], scale=[[std_px], [std_py]])
            else:
                std_px = self.noise[0][0]
                std_py = self.noise[0][1]
                std_vx = self.noise[0][2]
                std_vy = self.noise[0][3]
                sample = self.rng.normal( [[0], [0], [0], [0]], scale=[[std_px], [std_py], [std_vx], [std_vy]]) # different sample for each time step, but same for each running main code
            return sample.reshape(-1,1)

    
    def gen_pmf(self, control_input, state, noise, S):
        # 1) Decide std_ux and std_uy
        std_ux_ux, std_ux_uy, std_uy_ux, std_uy_uy = noise[1]
        
        std_ux = np.sqrt(std_ux_ux*(control_input[0,0]**2) +
                        std_ux_uy*(control_input[1,0]**2))
        std_uy = np.sqrt(std_uy_ux*(control_input[0,0]**2) +
                        std_uy_uy*(control_input[1,0]**2))
        samples_u = np.random.normal([[0],[0]], [[std_ux], [std_uy]], size=(S,2,1))
        pdf_ux = _pdf_or_ones(samples_u[:,0], std_ux)
        pdf_uy = _pdf_or_ones(samples_u[:,1], std_uy)
            
        if state.shape[0] == 2:
            std_px, std_py = noise[0]
            if std_px <= 1e-5 and std_py <= 1e-5:
                std_px = 0.001
                std_py = 0.001
            # std_px = 0.05
            # std_py = 0.05
            samples_x = np.random.normal([[0],[0]],
                                    [[std_px], [std_py]],
                                    size=(S,2,1))
            pdf_px = _pdf_or_ones(samples_x[:,0], std_px)
            pdf_py = _pdf_or_ones(samples_x[:,1], std_py)
            
            # 5) Multiply => joint pdf => normalize => pmf
            joint_pdf = pdf_ux * pdf_uy * pdf_px * pdf_py 

        else:
            std_px, std_py, std_vx, std_vy = noise[0]

            samples_x = np.random.normal([[0],[0],[0],[0]],
                                    [[std_px], [std_py], [std_vx], [std_vy]],
                                    size=(S,4,1))
            # 4) Compute PDFs using a helper function
            pdf_px = _pdf_or_ones(samples_x[:,0], std_px)
            pdf_py = _pdf_or_ones(samples_x[:,1], std_py)
            pdf_vx = _pdf_or_ones(samples_x[:,2], std_vx)
            pdf_vy = _pdf_or_ones(samples_x[:,3], std_vy)
        
            # 5) Multiply => joint pdf => normalize => pmf
            joint_pdf = pdf_ux * pdf_uy * pdf_px * pdf_py * pdf_vx * pdf_vy
        
        pmf = joint_pdf / np.sum(joint_pdf)

        return pmf, samples_u, samples_x


    def rvo_neighbors(self, object_list):
        """
        Get the list of RVO neighbors.
        Returns:
            list: List of RVO neighbor states [x, y, vx, vy, radius].
        """
        return [
            obj.rvo_neighbor_state() for obj in object_list if self.id != obj.id
        ]
    
    def rvo_neighbor_state(self):
        """
        Get the RVO state for this object.

        Returns:
            list: State [x, y, vx, vy, radius].
        """
        return [
            self.x_curr[0, 0],
            self.x_curr[1, 0],
            self.velocity_xy[0, 0],
            self.velocity_xy[1, 0],
            self.radius,
        ]
    
    def rvo_state(self):
        """
        Get the full RVO state including desired velocity.

        Returns:
            list: State [x, y, vx, vy, radius, vx_des, vy_des, theta].
        """
        self.update_target()
        
        _, v_des = self.nominal_input(self.x_curr, self.target) # acceleration
        vx_des = v_des[0, 0]
        vy_des = v_des[1, 0]
        
        return [
            self.x_curr[0, 0],
            self.x_curr[1, 0],
            self.velocity_xy[0, 0],
            self.velocity_xy[1, 0],
            self.radius,
            vx_des,
            vy_des,
        ]
    

    def sfm_obstacles(self, obstacles):
        """
        Get the list of SFM obstacles.
        Returns:
            list: List of SFM obstacle states [x, y, vx, vy, radius].
        """
        return [
            obj.sfm_obstacle_state() for obj in obstacles
        ]
        
    def sfm_obstacle_state(self):
        """
        Get the SFM state for this object.

        Returns:
            list: State [x, y, vx, vy, radius].
        """
        return ObservableState(
            px=self.x_curr[0, 0],
            py=self.x_curr[1, 0],
            vx=self.velocity_xy[0, 0],
            vy=self.velocity_xy[1, 0],
            radius=self.radius,
        )
        
    def sfm_state(self):
        """
        Get the full SFM state including desired velocity.

        Returns:
            list: State [x, y, vx, vy, radius, vx_des, vy_des].
        """
        self.update_target()
        
        # _, v_des = self.nominal_input(self.x_curr, self.target)
        v_des = 2.0 # TODO u max is useless if set vdes as 2.0
            
        theta = np.arctan2(self.velocity_xy[1, 0], self.velocity_xy[0, 0])
        return FullState(
            px=self.x_curr[0, 0],
            py=self.x_curr[1, 0],
            vx=self.velocity_xy[0, 0],
            vy=self.velocity_xy[1, 0],
            radius=self.radius,
            v_pref=np.linalg.norm(v_des),
            gx=self.target[0, 0],
            gy=self.target[1, 0],
            theta=theta,
            # omega=None,
        )
        
    def update_target(self, proximity_threshold=0.2):
        current_pos = self.x_curr[:2]
        target_pos = self.target[:2]
        distance = np.linalg.norm(current_pos - target_pos)
        if distance < proximity_threshold:
            self.target = self.x0.copy()
            # print(f"Target updated: current position {current_pos.flatten()} is within {distance:.4f} of target; "
            #     f"new target is set to x0: {self.x0.flatten()}")
        
    def update_local_objects(self, obstacles, robots, R_sensing=20,max_num_obs=5,use_one_obs=True):
        local_objects = []
        obj_pos = self.x_curr[:2]  # Assuming x_curr is at least 2D (x, y)

        # --- Process obstacles ---
        obstacles_in_range = []
        for obs in obstacles:
            # If the object itself is an obstacle, skip itself.
            if self.type == 'singleint_obs' and obs.id == self.id:
                continue
            obs_pos = obs.x_curr[:2]
            distance = np.linalg.norm(obj_pos - obs_pos)
            if distance <= R_sensing:
                obstacles_in_range.append((obs, distance))

        # If no obstacles are in range, fall back to the single nearest obstacle
        if not obstacles_in_range and use_one_obs:
            nearest_obs = min(obstacles, key=lambda o: np.linalg.norm(obj_pos - o.x_curr[:2]))
            obstacles_in_range.append((nearest_obs, np.linalg.norm(obj_pos - nearest_obs.x_curr[:2])))

        # Sort obstacles by distance and select the five closest ones.
        obstacles_in_range.sort(key=lambda tup: tup[1])
        if len(obstacles_in_range) > max_num_obs:
            closest_obstacles = [obs for obs, _ in obstacles_in_range[:5]]
        else:   
            closest_obstacles = [obs for obs, _ in obstacles_in_range]

        # --- Process robots (if applicable) ---
        if self.behavior_type == 'all_obj':
            for robot in robots:
                # For specific robot types, skip adding the object itself.
                if self.type in ['doubleint', 'unicycle_v2', 'doubleint_v1', 'unicycle_v1', 'singleint']:
                    if robot.id == self.id:
                        continue

                robot_pos = robot.x_curr[:2]
                distance = np.linalg.norm(obj_pos - robot_pos)
                if distance <= R_sensing:
                    local_objects.append(robot)

        # Add the (up to) five closest obstacles.
        local_objects.extend(closest_obstacles)

        return local_objects
      

    def observation(self, all_obstacles, all_robots, method='sfm'):            
        if method == 'rvo':   
            self_state = self.rvo_state()
            loc_objects = self.update_local_objects(all_obstacles, all_robots, R_sensing=5)
            loc_obj_state_list = self.rvo_neighbors(loc_objects)
        # elif method == 'rl_v2':
        #     self_state = self.state()
        #     loc_objects = self.update_local_objects(all_obstacles, all_robots, R_sensing=5)
        #     loc_obj_state_list = self.full_obstacles(loc_objects)
        else:
            self_state = self.sfm_state()
            loc_objects = self.update_local_objects(all_obstacles, all_robots, R_sensing=5)
            loc_obj_state_list = self.sfm_obstacles(loc_objects)

        return [self_state, loc_obj_state_list]

