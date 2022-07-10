from __future__ import annotations

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

class Pose2D:
	def __init__(self, transformation_matrix) -> None:
		self.T = transformation_matrix

	def __add__(self, other) -> Pose2D:
		return Pose2D(self.T+other.T)

	def inverse(self) -> Pose2D:
		return Pose2D(np.linalg.inv(self.T))

	def to_point(self) -> Point2D:
		return Point2D(self.x, self.y)

	def to_tuple(self) -> Tuple:
		return (self.x, self.y, self.theta)

	@staticmethod
	def _gen_trans_matrix(x, y, theta):
		return np.array([
			[np.cos(theta),	-np.sin(theta),	x],
			[np.sin(theta), np.cos(theta),	y],
			[0,				0,				1],
		])

	@classmethod
	def from_xytheta(cls, x, y, theta) -> Pose2D:
		return cls(Pose2D._gen_trans_matrix(x, y, theta))
		
	@property
	def x(self):
		return self.T[0][2]

	@property
	def y(self):
		return self.T[1][2]

	@property
	def theta(self):
		return np.arccos(self.T[0][0])

	def __str__(self):
		return f'Pose2D({self.x}, {self.y}, {self.theta})'

class Point2D:
	def __init__(self, x, y):
		self.x = float(x)
		self.y = float(y)

	def transform(self, poses: Union[Pose2D, List[Pose2D]]) -> Point2D:
		if isinstance(poses, Pose2D):
			pn = poses.T @ np.array([self.x, self.y, 1]).T
			return Point2D(pn[0], pn[1])
		elif isinstance(poses, list):
			pcpy = Point2D(self.x, self.y)
			for p in poses:
				pcpy = pcpy.transform(p)
			return pcpy
		else:
			raise TypeError('Only Pose2D can be applied as transformations.')

	def dist_from(self, point: Point2D):
		return np.sqrt((self.x-point.x)**2 + (self.y-point.y)**2)

	def to_tuple(self) -> Tuple:
		return (self.x, self.y)

	def __str__(self):
		return f'Point2D({self.x}, {self.y})'
	
class Robot:

	def __init__(self, pose: Pose2D, source_pos: Point2D, mic_pos: Point2D) -> None:
		self.pose = pose                    # global pose of robot
		self._source_pos = source_pos	    # rel pose of speaker 
		self._mic_pos = mic_pos	        	# rel pose of mic

	@property
	def src_global_pos(self):
		"""Returns global pos [x,y] of source."""
		return self._source_pos.transform(self.pose)

	@property
	def mic_global_pos(self):
		"""Returns global pos [x,y] of mic."""
		return self._mic_pos.transform(self.pose)

	def get_radius(self):
		robot_pos = self.pose.to_point()
		return max(
				robot_pos.dist_from(self.mic_global_pos),
				robot_pos.dist_from(self.src_global_pos)
			)

def extract_TOAs(rir, fs, num=None, plot=False):
	"""extract TOAs from an impulse response.

	Args:
		rir (arraylike): Room impulse response.
		fs (int): sampling rate
		num (int, optional): If specified, returns only the `num` highest peaks. Defaults to None.
		plot (bool, optional): Plot if true. Defaults to False.

	Returns:
		px: Peak x positions.
		py: Peak y positions or heights.
	"""
	envelop = get_envelope(rir, bottom_envelop=False)	# bottom envelop doesn't make much difference
	px, py = findpeaks(envelop)

	# sort peaks by py in descending order
	px = np.flip( np.take(px, np.argsort(py)) )
	py = np.flip(np.sort(py))
	if num is not None:
		px = px[:num]
		py = py[:num]
	
	# right now assuming peak prominence = y_peak
	if plot:
		plt.plot(rir)
		plt.plot(envelop)
		plt.plot(px, py, 'r+')
		plt.show()

	# px is in indices, convert to timestamps
	px = px.astype(float) / fs

	return px, py

def get_envelope(signal, bottom_envelop=False):
	"""Returns envelop of given signal.

	Args:
		signal (arraylike): Signal to find envelop of.
		bottom_envelop (bool, optional): If True, will return average of top and bottom envelop.
			Otherwise just top envelop. Defaults to False.

	Returns:
		arraylike: envelop signal, same length.
	"""
	x, y = findpeaks(signal)
	# add anchor points
	x = [0] + x + [len(signal)-1]
	y = [0] + y + [signal[-1]]

	f_int = interp1d(x, y, 'linear')
	envelop = f_int(np.arange(0, len(signal)))

	if bottom_envelop:
		x, y = findpeaks(signal)

		x = [0] + x + [len(signal)-1]
		y = [0] + y + [signal[-1]]

		f_int = interp1d(x, y, 'linear')
		bottom_envelop = f_int(np.arange(0, len(signal)))
		envelop = (envelop+bottom_envelop)/2

	return envelop

def findpeaks(signal):
	"""Returns x and y coordinates of peaks in signal.

	Args:
		signal (arraylike): Signal to find peaks in.

	Returns:
		xvals: Indices of peaks.
		yvals: Y coords of peaks.
	"""
	xvals = []
	yvals = []
	for i in range(1, len(signal)-1):
		if signal[i-1] <= signal[i] >= signal[i+1]:
			xvals.append(i)
			yvals.append(signal[i])

	return xvals, yvals

def findtroughs(signal):
	xvals = []
	yvals = []
	for i in range(1, len(signal)-1):
		if signal[i-1] >= signal[i] <= signal[i+1]:
			xvals.append(i)
			yvals.append(signal[i])

	return xvals, yvals
