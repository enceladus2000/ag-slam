import matplotlib.pyplot as plt
from helpers import Robot, Pose2D
import pyroomacoustics as pra

def plot_map(room: pra.Room, robot: Robot):
    room.plot()
    robot_size = robot.get_radius()
    # ax.legend()
    plt.plot(*robot.pose.to_tuple()[0:2], 'ko')
    plt.grid(True)
    plt.legend()
    plt.show()