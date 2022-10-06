from collections import namedtuple

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import X

from helpers import Point2D, Pose2D


class Plane2D:
	def __init__(self, normal, dist_from_origin) -> None:
		self.normal = np.array(normal)
		self.normal = self.normal / np.sqrt(self.normal.dot(self.normal))
		self.dfo = float(dist_from_origin)

	def inplane_point(self, point: Point2D) -> Point2D:
		point_np = point.to_numpy()
		dot = point_np.dot(self.normal)
		ipp_np = point_np + (self.dfo - dot) * self.normal
		return Point2D.from_arraylike(ipp_np)

	def plot(self):
		p = self.dfo * self.normal
		alpha = np.arctan2(self.normal[1], self.normal[0])
		ax = plt.gca()

		if alpha in (0, np.pi):
			p2 = p + np.array((0, 1))
			ax.axline(p, p2)
		else:
			slope = np.tan(alpha + np.pi / 2)
			assert -np.inf < slope < np.inf
			ax.axline(p, slope)

"""Range-bearing measurement class. Convention is same as Pose2D, 
	i.e. use theta measured from x axis counterclockwise. """
RB_mmnt = namedtuple("RB_mmnt", ["range", "bearing"])

def main():
	# plt.ion()
	np.random.seed(62646)

	odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array((0.2, 0.2, 0.1)))
	prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array((0.5, 0.5, 0.2)))
	traj_noise = np.diag((0.01, 0.01, 0.001))
	mmnt_noise = np.diag((0.001, 0.001))

	centre = np.array((5, 2))
	radius = 1
	traj_base = circular_traj(
		centre, radius, num=4
	)  # 'desired' trajectory, used to compute odometry estimates
	traj_real = gen_noisy_traj(traj_base, traj_noise)  # actual trajectory (noisy)
	plot_traj(traj_base, color="b")
	plot_traj(traj_real, color="g", marker='o')

	# ground truth plane wall
	normal = np.array((1, 0))
	dfo = 0.5
	wall_gt = Plane2D(normal, dfo)
	wall_gt.plot()

	# calculate point cloud, both ground truth and actual (noisy)
	pcl_gt = calc_inplane_points(wall_gt, [pose.to_point() for pose in traj_real])
	rbms_gt = calc_rbms(pcl_gt, traj_real)
	rbms_noisy = gen_noisy_rbms(rbms_gt, mmnt_noise)	# the actual measured values corrupted with noise
	pcl_noisy = reproject_rbms(rbms_noisy, traj_real)

	# pcl, rbms, ippcl = noisy_point_mmnts(traj_real, wall_gt, mmnt_noise)
	plot_pcl(pcl_noisy, color="g", marker="x")
	plot_pcl(pcl_gt, color="r", marker=".")

	graph = gtsam.NonlinearFactorGraph()

	odom_ests = [traj_base[i + 1] - traj_base[i] for i in range(len(traj_base) - 1)]
	prior_est = traj_base[0]
	add_traj_to_graph(graph, odom_ests, prior_est, prior_noise, odom_noise)

	# wall_est = fit_plane(point_cloud)

	plt.gca().set_aspect("equal")
	plt.tight_layout()
	plt.grid(True)
	plt.show()

	# add IPPs, wall_est to graph

	# cost function
	initial_estimate = create_initial_est(traj_base)
	params = gtsam.LevenbergMarquardtParams()
	optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate,
												  params)

	result = optimizer.optimize()
	print(f'Final result: {result}')

	# save graph to file
	gtsam.writeG2o(graph, result, 'test.g2o')
	print('Written to file.')



def create_initial_est(traj_est):
	pose_keys = gen_keys(len(traj_est), X)
	est = gtsam.Values()
	for i in range(len(traj_est)):
		est.insert(pose_keys[i], gtsam.Pose2(*traj_est[i].to_tuple()))

	return est

def reproject_rbms(rbms, traj):
	assert len(rbms) == len(traj)

	pcl_reprojected = []
	for i in range(len(rbms)):
		m = rbms[i]
		t = Point2D.from_polar(*m)
		r = t.transform(traj[i])

		pcl_reprojected.append(r)

	return pcl_reprojected

def gen_noisy_rbms(rbms_gt, covariance_mat):
	assert covariance_mat.shape == (2, 2)

	rbms_noisy = []
	for rbm in rbms_gt:
		rbm_n = np.random.multivariate_normal(tuple(rbm), covariance_mat)
		rbm_n = RB_mmnt(rbm_n[0], rbm_n[1])
		rbms_noisy.append(rbm_n)

	return rbms_noisy


def calc_rbms(pointcloud, trajectory):
	"""returns range-bearing mmnt from one point to one pose. """
	assert len(pointcloud) == len(trajectory)

	rbms = []
	for i in range(len(trajectory)):
		bearing = trajectory[i].bearing_to_point(pointcloud[i])
		dist = trajectory[i].to_point().dist_from(pointcloud[i])
		rbm = RB_mmnt(dist, bearing)
		rbms.append(rbm)

	return rbms
	
def calc_inplane_points(plane: Plane2D, points):
	return [plane.inplane_point(p) for p in points]


def gen_noisy_traj(traj_base, traj_noise):
	traj_noisy = []
	for pose in traj_base:
		pn = np.random.multivariate_normal(pose.to_tuple(), traj_noise)
		pose_noisy = Pose2D.from_arraylike(pn)
		traj_noisy.append(pose_noisy)

	return traj_noisy


def add_traj_to_graph(graph, odom_ests, prior_est: Pose2D, prior_noise, odom_noise):
	keys = gen_keys(len(odom_ests)+1, X)

	graph.add(
		gtsam.PriorFactorPose2(keys[0], gtsam.Pose2(*prior_est.to_tuple()), prior_noise)
	)

	for i in range(len(odom_ests)):
		graph.add(
			gtsam.BetweenFactorPose2(
				keys[i], keys[i + 1], gtsam.Pose2(*odom_ests[i].to_tuple()), odom_noise
			),
		)

def gen_keys(num, symbol):
	return [symbol(i) for i in range(num)]

def plot_pcl(pcl, **kwargs):
	for point in pcl:
		plt.scatter(point.x, point.y, **kwargs)


def noisy_point_mmnts(traj, wall, noise_factor):
	"""Generate noisy range-bearing mmnts and associated noisy pointcloud.

	Args:
			traj (list of Pose2D)
			wall (Plane2D)
			noise_factor (float): Scalar value

	Returns:
			pointcloud, rb_mmnts, inplane-pointcloud
	"""

	points = []
	rbms = []  # range bearing measurements
	ippoints = []
	for pose in traj:
		# calc inplane point
		IPP = wall.inplane_point(pose.to_point())
		d = IPP.dist_from(pose.to_point())
		# generate noisy point mmnt by perturbing ipp
		noisy_point = np.random.normal(IPP.to_tuple(), d * noise_factor, size=(2,))
		noisy_point = Point2D.from_arraylike(noisy_point)

		points.append(noisy_point)
		ippoints.append(IPP)

		# calculate noisy range and bearing mmnts
		bearing = pose.bearing_to_point(noisy_point)
		dist = pose.to_point().dist_from(noisy_point)
		rbm = RB_mmnt(dist, bearing)
		rbms.append(rbm)

	return points, rbms, ippoints


def circular_traj(centre, radius, num):
	traj = []
	for i in range(num):
		alpha = i * 2 * np.pi / num
		x = np.cos(alpha)
		y = np.sin(alpha)
		theta = alpha + np.pi / 2
		p = centre + radius * np.array((x, y))
		pose = Pose2D.from_xytheta(p[0], p[1], theta)
		traj.append(pose)

	return traj


def plot_pose2D(pose: Pose2D):
	plt.scatter(*pose.to_tuple(), c="k")


def plot_lineseg(point1, point2, **kwargs):
	x = point1[0], point2[0]
	y = point1[1], point2[1]
	plt.plot(x, y, **kwargs)


def plot_traj(traj, **kwargs):
	for i in range(len(traj) - 1):
		plot_lineseg(
			traj[i].to_point().to_tuple(),
			traj[i + 1].to_point().to_tuple(),
			**kwargs
		)


if __name__ == "__main__":
	main()
