from re import X

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

from helpers import Pose2D, Point2D, Robot, extract_TOAs
from plotting import plot_map

room_corners = np.transpose([
	[0, 0], [5, 0],
	[7, 6], [0, 6]
])
fs = 16000

def main():
	room = pra.Room.from_corners(
					room_corners, fs=fs, max_order=4,
					materials=pra.Material(0.5),
				)

	robot = Robot(
		pose=Pose2D.from_xytheta(2, 3, np.pi/2),
		source_pos=Point2D(-0.1, 0),
		mic_pos=Point2D(0.1, 0)
	)
	print(robot.mic_global_pos, robot.src_global_pos)

	room.add_microphone(robot.mic_global_pos.to_tuple())
	room.add_source(robot.src_global_pos.to_tuple())

	plot_map(room, robot)

	# room.compute_rir()
	# room.plot_rir()
	# plt.show()

	# times, amps = extract_TOAs(room.rir[0][0], fs, 8, True)
	
if __name__ == '__main__':
	main()
