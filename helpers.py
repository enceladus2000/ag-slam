import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class Pose2D:
	def __init__(self, x, y, theta=0) -> None:
		self.x = x 
		self.y = y 
		self.theta = theta

	@property
	def pos(self):
		"""Returns [x, y]"""
		return [self.x, self.y]

	def __add__(self, other):
		x = self.x + other.x*np.cos(self.theta) - other.y*np.sin(self.theta) 
		y = self.y + other.x*np.sin(self.theta) - other.y*np.cos(self.theta)
		return Pose2D(
			x, y,
			self.theta + other.theta    # TODO: mod?
		)

class Robot:

	def __init__(self, pose, source_pose, mic_pose) -> None:
		self.pose = pose                    # global pose of robot
		self._source_pose = source_pose     # rel pose of speaker 
		self._mic_pose = mic_pose           # rel pose of mic

	@property
	def source_pos(self):
		"""Returns global pos [x,y] of source."""
		return (self.pose + self._source_pose).pos

	@property
	def mic_pos(self):
		return (self.pose + self._mic_pose).pos

def extract_TOAs(rir, fs, num=None, plot=False):
	envelop = get_envelope(rir, False)
	px, py = findpeaks(envelop)

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

	return px, py

def get_envelope(signal, bottom_envelop=False):
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
