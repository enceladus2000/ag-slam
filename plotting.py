import matplotlib.pyplot as plt
from helpers import Robot, Pose2D
import pyroomacoustics as pra

"""
TODO:
* zoom out a lil DONE
* circle for robot DONE
* legend ??
"""

def plot_map(room: pra.Room, robot: Robot):
	room.plot()

	ax = plt.gca()
	ax.plot(*robot.pose.to_tuple()[0:2], 'ko')
	robot_size = robot.get_radius()
	circle_patch = plt.Circle(robot.pose.to_tuple(), robot_size, fill=False)
	ax.add_artist(circle_patch)

	plt.grid(True)
	plt.legend()

	xlims, ylims = get_room_lims(room)
	ax.set_xlim(xlims[0], xlims[1])
	ax.set_ylim(ylims[0], ylims[1])
	ax.set_aspect('equal', adjustable='box')

	plt.show()

def get_room_lims(room: pra.Room, factor=1.2):
	bbox = room.get_bbox()
	# pra.Room.get_bbox returns [[x1, x2], [y1, y2]]
	x1, x2 = bbox[0]
	y1, y2 = bbox[1]

	dx = x2 - x1
	kx = dx * (factor-1) / 2
	x1 -= kx
	x2 += kx

	dy = y2 - y1
	ky = dy * (factor-1) / 2
	y1 -= ky
	y2 += ky
	
	return [x1, x2], [y1, y2]
