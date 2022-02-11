from re import X
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from helpers import Pose2D, Robot, extract_TOAs
from plotting import plot_map

room_corners = np.transpose([
	[0, 0], [5, 0],
	[7, 6], [0, 6]
])
fs = 16000

def main():
	room = pra.Room.from_corners(
					room_corners, absorption=0.1,
					fs=fs, max_order=4
				)

	robot = Robot(
		pose=Pose2D(2, 3, np.pi/2),
		source_pose=Pose2D(-0.1, 0),
		mic_pose=Pose2D(0.1, 0)
	)
	room.add_microphone(robot.mic_pos)
	room.add_source(robot.source_pos)

	# plot_map(room, robot)

	room.compute_rir()
	# room.plot_rir()
	# plt.show()

	extract_TOAs(room.rir[0][0], fs, 8)
	
if __name__ == '__main__':
	main()