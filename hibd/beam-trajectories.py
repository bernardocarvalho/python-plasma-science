#!/usr/bin/env python3
"""
 https://scicomp.stackexchange.com/questions/43032/eulers-method-for-fast-moving-particle-trajectory
 https://github.com/scipython/plasma-projects/blob/master/gyromotion/Gyromotion.ipynb
 http://implicit-layers-tutorial.org/neural_odes/
"""
#
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from isttok_hibd import (chamberXY,
                         calc_ion_speed,
                         calc_trajectory,
                         hib_pos,
                         hib_dir,
                         plasma_density)
import argparse
from time import perf_counter
# from threading import Thread
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# from scipy.integrate import odeint
# c = constants.unit('speed_of_light')
# c = -299792458
q = -1
c = constants.speed_of_light
me = constants.electron_mass
e = constants.e
eV = constants.electron_volt
# mu_0 = constants.mu_0
amc = constants.physical_constants['atomic mass constant'][0]
# Cs+ Molecular weight: 132.9049033
q_csp = +e
m_csp = 132.9049033 * amc


def plot_trajectories2D(ax, sol):
    """Produce a static plot of the trajectories.
    """
    # plt.title("Position")
    ax.plot(chamberXY[0], chamberXY[1], linestyle='dashed')
    ax.plot(sol.y[0, :], sol.y[1, :])
    # plt.gca().set_aspect('equal', adjustable='box')
    ax.set_aspect('equal', adjustable='box')
    # ax.gridlines.set_visible(True)
    # plt.show()
# return sol


# def calc_task(EkeCV, X0):
def calc_task(beam):
    X0 = beam['Pos']
    Q = beam['Charge'] * q_csp
    # E_cs = EkeCV * 1e3 * eV
    # v_cs = calc_ion_speed(E_cs, m_csp)
    # v_i = v_cs * hib_dir
    # X0 = np.hstack((hib_pos, v_i))
    return calc_trajectory(Q, m_csp, tf=4.4e-6, X=X0)


#  if __name__ == '__main__':
def main():
    print("Number of processors: ", mp.cpu_count())
    parser = argparse.ArgumentParser(
            description='Compute trajectory of hib ions')

    parser.add_argument('-e', '--energy', type=float,
                        help='Cs+ HIB Energy in keV', default=20)
    parser.add_argument('-p', '--plots', help='Show beam plots',
                        action='store_true')  # on/off op_flags
    args = parser.parse_args()

    E_cs = args.energy * 1e3 * eV
    # v_cs = np.sqrt(2 * E_cs / m_csp)
    v_cs = calc_ion_speed(E_cs, m_csp)
    print(f"E_cs={E_cs:.2g} J, v_cs={v_cs:.2g} m/s, {v_cs / c:.3g}")

    v_i = v_cs * hib_dir

    v_xi, v_yi, v_zi = v_i
    print(f"v_xi={v_xi:.2g}, v_yi={v_cs:.2g} m/s")
    # exit()
    X0 = np.hstack((hib_pos, v_i))
    sol0 = calc_trajectory(q_csp, m_csp, tf=4.4e-6, X=X0)
    print(f'Max time_eval was {sol0.t.max():.2g} second(s).')
    print(f't.shape {sol0.t.shape} y.shape {sol0.y.shape} ')

    secondaryBeams = []

    pdIdx = plasma_density(sol0.y)
    Xin = sol0.y[:, pdIdx]
    lastPoint = Xin[:2, 0]
    # with np.nditer(Xin, flags=['multi_index'], op_flags=["readwrite"]) as it:
    #    for row_index in it:
    # row_data = Xin[:, it.multi_index[1]]
    dist = 0.0
    secSep = 0.005  # 5 mm
    for col in Xin.T:
        dp = np.linalg.norm(lastPoint - col[:2])
        lastPoint = col[:2]
        dist += dp
        # np.linalg.norm(lastPoint - col[:2])
        if dist > secSep:
            print(f"Pos: {col[0]:.3f}, {col[1]:.3f}, dist = {dist:.4f}")
            dist = 0.0
            b = {
                'Pos': col,
                'Charge': +2,
                'EnergykeV': args.energy
                }
            secondaryBeams.append(b)

    print(f'pd   {pdIdx.shape} {Xin.shape} Sec. Beams:{len(secondaryBeams)} ')
    # breakpoint()
    # for i, x in enumerate(pd):
    #     if pd > 0:
    # print(f"time {i}, density {pd}")
    start_time = perf_counter()
    """
    threads = []
    for n in range(80):
        t = Thread(target=calc_task, args=(args.energy + n,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    """
    TASKS = len(secondaryBeams)
    """
    beams = []
    for i in range(TASKS):
        E_cs = (args.energy + i) * 1e3 * eV
        v_cs = calc_ion_speed(E_cs, m_csp)
        v_i = v_cs * hib_dir
        Xi = np.hstack((hib_pos, v_i))
        b = {
                'Pos': Xi,
                'Charge': +2,
                'EnergykeV': args.energy + i
                }:Warning
        beams.append(b)
    """
    # https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
    solutions = []
    with ProcessPoolExecutor(max_workers=TASKS) as executor:
        sol_map = executor.map(calc_task, secondaryBeams)
        for b, sol in zip(secondaryBeams, sol_map):
            solutions.append(sol)
            # print(f"calc_task({b['EnergykeV']}) = {sol.t.max():2g}")
    end_time = perf_counter()

    if args.plots:
        fig = plt.figure()
        ax0 = fig.add_subplot(122)  # , projection='3d')
        for sol in solutions:
            ax0.plot(sol.y[0, :], sol.y[1, :])
        ax0.plot(chamberXY[0], chamberXY[1], linestyle='dashed')
        ax0.set_aspect('equal', adjustable='box')

        ax1 = fig.add_subplot(121)  # , projection='3d')
        plot_trajectories2D(ax1, sol0)
        plt.show()

    print(f'It took {end_time - start_time: 0.2f} second(s) to complete all taks.')

if __name__ == '__main__':
    main()
