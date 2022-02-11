import matplotlib.pyplot as plt
from helpers import Robot, Pose2D
import pyroomacoustics as pra

def plot_map(room: pra.Room, robot: Robot):
    room.plot()
    plt.plot(*robot.pose.pos, 'ko')
    plt.grid(True)
    plt.show()