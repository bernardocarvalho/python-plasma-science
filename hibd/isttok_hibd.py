# https://scicomp.stackexchange.com/questions/43032/eulers-method-for-fast-moving-particle-trajectory
# https://github.com/scipython/plasma-projects/blob/master/gyromotion/Gyromotion.ipynb
# http://implicit-layers-tutorial.org/neural_odes/
#
import numpy as np
from scipy import constants
from scipy.integrate import solve_ivp
# from scipy.integrate import odeint
# c = constants.unit('speed_of_light')
# c = -299792458
c = constants.speed_of_light
mu_0 = constants.mu_0
amc = constants.physical_constants['atomic mass constant'][0]
# Cs+ Molecular weight: 132.9049033
# m = 0.002
# v_x = 0
# ISTTOK Major radius
R0 = 0.46
Rchamber = 0.12
r_coils_int = 0.20
r_coils_ext = 0.22
r_plasma_max = 0.08
Rdetector2 = 0.2 * 0.2
Ydetector = -0.2

# HIBD initial entrey position, angle
hib_pos = np.array((-0.2, 0.38, 0))
# Inition HIB angle
theta = -0.3 * np.pi
hib_dir = np.array([np.cos(theta), np.sin(theta), 0])
# E_cs+ = 20 kV  = 1/2 M_cs * V^2

# radius = 0.1

# B_r = 12
B_tor = 0.3  # Mag field in axis (Tesla)
# Vol = 0.790524
# B = lambda x, y, z : (B_r*Vol) / (4*np.pi*np.sqrt(x**3 + y**2 + z**2))


# def B(x, y, z):
#     return (B_r*Vol) / (4*np.pi*np.sqrt(x**2 + y**2 + z**2))
# Pure Toroidal field
# Magnetic field, electric field
# No electric field
E = np.array((0.0, 0.0, 0.0))
#
def calc_ion_speed(Ecin, mass):
    """
    """
    return np.sqrt(2 * Ecin / mass)


def plasma_density(X, rmax=r_plasma_max):
    rpla = np.linalg.norm(X[:2, :], axis=0)
    return rpla < rmax


def Btor(X, R0=0.46, B0=0.5):
    if np.linalg.norm(X[:2]) < 0.15:
        Bz = B0 * (R0 + X[0]) / R0
    else:
        Bz = 0.0
    return np.array((0, 0, Bz))


def lorentz(t, X, q_over_m):
    """
    The equations of motion for the Lorentz force on a particle with
    q/m given by q_over_m. X=[x,y,z,vx,vy,vz] defines the particle's
    position and velocity at time t: F/m = a = (q/m)[E + v×B].
    """
    v = X[3:]
    drdt = v
    dvdt = q_over_m * (E + np.cross(v, Btor(X[:2])))
    return np.hstack((drdt, dvdt))


def calc_trajectory(q, m, tf, X):
    """
    Calculate the particle's trajectory.
    q, m are the particle charge and mass;
    X=[x,y,z,vx,vy,vz] are its initial position and velocity vector.
    """
    # Final time, number of time steps, time grid.
    # tf = 5e-6
    N = int(1e5)
    t = np.linspace(0, tf, num=N)
    # Initial positon and velocity components.
    y = X  # np.hstack((r0, v0))
    # Events to terminate odeint
    def leave_chamber(t, y, qm): return y[1] - Ydetector
    leave_chamber.terminal = True
    leave_chamber.direction = -1  # going down

    # def hit_detector(t, y, qm): return np.linalg.norm(y[:1]) - Rdetector
    def hit_detector(t, y, qm): return y[0] * y[0] + y[1] * y[1] - Rdetector2
    hit_detector.terminal = True
    hit_detector.direction = 1  # going out

    # Do the numerical integration of the equation of motion.
    # def leave_chamber(t, y): return np.linalg.norm(y[:1]) - 0.2
    # Integration method to use:
    # ‘RK45’ (default): Explicit Runge-Kutta method of order 5(4) [1].
    return solve_ivp(lorentz, [0, tf], y, t_eval=t,
                     events=hit_detector, args=(q/m,))
    # dense_output=True)


a = np.linspace(0, 2 * np.pi, 50)
chamberXY = Rchamber * np.array([np.cos(a), np.sin(a)])
