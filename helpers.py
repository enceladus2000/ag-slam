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

	def __init__(self, pose: Pose2D, source_pose: Pose2D, mic_pose: Pose2D) -> None:
		self.pose = pose                    # global pose of robot
		self._source_pose = source_pose     # rel pose of speaker 
		self._mic_pose = mic_pose           # rel pose of mic

	@property
	def source_pos(self):
		"""Returns global pos [x,y] of source."""
		return (self.pose + self._source_pose).pos

	@property
	def mic_pos(self):
		"""Returns global pos [x,y] of mic."""
		return (self.pose + self._mic_pose).pos

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
