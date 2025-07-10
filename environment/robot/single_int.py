from ..object_base import ObjectBase
import numpy as np
import casadi as ca

def is_casadi(x,u=None):
    if u is None:
        return isinstance(x, ca.SX) or isinstance(x, ca.MX)
    else:
        return isinstance(x, ca.SX) or isinstance(x, ca.MX) or isinstance(u, ca.SX) or isinstance(u, ca.MX)
    
class SingleIntegratorRobot(ObjectBase):
    def __init__(self, x0, u_max, u_min, mapsize, radius, dt, noise, id, target, seed):
        assert len(x0) == 2, "SingleIntegratorRobot requires a 2-dimensional x0"
        super().__init__(x0, radius, dt, noise, id, target, seed)
        
        self.type = 'singleint'
        self.behavior_type = 'only_obs'

        # Define specific dynamics for the single integrator
        self.A = np.array([
            [1, 0],
            [0, 1]
        ])
        self.B = np.array([
            [dt, 0],
            [0, dt]
        ])
        self.G = np.array([[1], [0]])  

        self.n = 2
        self.m = 2

        self.u_min = np.array([u_min[0], u_min[1]]).reshape(-1, 1)
        self.u_max = np.array([u_max[0], u_max[1]]).reshape(-1, 1)
        self.x_min = np.array([mapsize[0,0], 
                               mapsize[2,0]]).reshape(-1, 1)
        self.x_max = np.array([mapsize[1,0],
                               mapsize[3,0]]).reshape(-1, 1)  
        
        self.xlog = [self.x0]
        self.ulog = [np.zeros((self.m, 1))]

        self.u = np.zeros((self.m, 1)).reshape(-1, 1)
        # self.target = np.array(target).reshape(-1, 1)
        
        self.P = np.eye(self.n) # state cost
        self.Q = np.eye(2) 
        self.R = np.eye(self.m) # control cost


    def step(self, x_k1=None, uk=None):
        if x_k1 is None:
            x_k1 = self.dynamics(self.x_curr, uk)
        self.x_curr = x_k1
        self.xlog.append(self.x_curr)
        self.ulog.append(uk)
        # self.u_cost += uk.T @ uk * self.dt
        self.velocity_xy = uk
        
    def reset(self):
        self.x_curr = self.x0
        self.xlog = [self.x0]
        self.ulog = [np.zeros((self.m, 1))]
        self.u_cost = 0

    def dynamics(self, x, u): # real dynamics, execute the control input, not used for calculate cvar cbf constraint
        # check if x and u are nx1 and mx1 vectors
        assert x.shape == (self.n, 1), f"x shape: {x.shape}"
        assert u.shape == (self.m, 1), f"u shape: {u.shape}"
        if max(self.noise[1]) > 1e-5: # control noise
            u_noise = self.u_disturbance(u) # dont add noise for x_next in optimization process, add noise to control input for real movement
        else:
            u_noise = np.zeros((self.m,1))
        if max(self.noise[0]) > 1e-5: # statte noise
            x_noise = self.x_disturbance(x)
        else:
            x_noise = np.zeros((self.n,1))
        return self.A @ x + self.B @ (u + u_noise) + x_noise  
    
    
    def dynamics_uncertain(self, x, u, wu =np.zeros((2,1)), wx= np.zeros((2,1))): # used for calculate cvar cbf constraint
        return self.A @ x + self.B @ (u + wu) + wx
    
    def nominal_input(self, X, G, d_min=0.05, k_v=0.5):
        G = np.copy(G.reshape(-1, 1))
        v_max = np.sqrt(self.u_max[0]**2 + self.u_max[1]**2) # TODO
        pos_errors = G[0:2,0] - X[0:2,0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)
        v_des = k_v * pos_errors
        # v_des = np.clip(v_des.reshape(-1,1), self.u_min, self.u_max)
        v_mag = np.linalg.norm(v_des)
        if v_mag > v_max:
            v_des = v_des * v_max / v_mag
        return v_des.reshape(-1, 1), v_des.reshape(-1, 1)
    
    def agent_barrier_dt(self, x_k, x_k1, obs_state = None, obs_traj = None, radius=0.1, H=np.zeros((4, 1)), L=-np.inf, htype='dist', obs_fov=None):
        # x_k1 = self.dynamics(x_k, u_k)
        if htype == 'dist':
            obs_pos = obs_state[0:2].reshape(-1, 1)
            # obs_curr_pos = obs_traj[-1][0:2].reshape(-1, 1)
            h_k1 = self.h_dist(x_k1, obs_pos, radius)
            h_k = self.h_dist(x_k, obs_pos, radius)
        elif htype == 'linear':
            h_k1 = self.h_linear(x_k1, H = H, L = L)
            h_k = self.h_linear(x_k, H=H, L=L)
        elif htype == 'vel':
            raise NotImplementedError
        elif htype == 'cone':
            raise NotImplementedError
        elif htype == 'dist_cone':
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown htype: {htype}")

        d_h = h_k1 - h_k

        return h_k, d_h
    
    


    
    def h_dist(self, x_k, obs_pos, radius, beta=1.01):
        '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
        if is_casadi(x_k):
            h = ca.mtimes((x_k[0:2] - obs_pos[0:2]).T, (x_k[0:2] - obs_pos[0:2])) - beta*radius**2
        else:
            h = (x_k[0, 0] - obs_pos[0, 0])**2 + (x_k[1, 0] - obs_pos[1, 0])**2 - beta*radius**2
        return h
    
    def h_linear(self, x_k, H = np.array([0,0,0,1]).reshape(-1,1), L=-1):
        if is_casadi(x_k):
            h = ca.mtimes(x_k.T, H) + L 
        else:
            h = x_k.T @ H + L
        return h
    

    

        
