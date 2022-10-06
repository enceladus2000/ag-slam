""" 
Extremely basic example which reads g2o file and optimises using LM.
"""

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.utils import plot as gtsam_plot

# (find and) load datafile, noisetoygraph or something
datafile = 'example_graph.txt'
graph, initial = gtsam.readG2o(datafile, False)

# add prior on pose having index 0
def vector3(x, y, z):
	"""Create 3d double numpy array."""
	return np.array([x, y, z], dtype=float)

prior_model = gtsam.noiseModel.Diagonal.Variances(vector3(1e-6, 1e-6, 1e-8))
graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(), prior_model))

# init optimizer (gauss newton or levenberg marquedt)
params = gtsam.LevenbergMarquardtParams()
params.setVerbosity("Termination")
params.setMaxIterations(100)
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

# optimize
result = optimizer.optimize()

# print results
print('Done optimization')
print("initial error = ", graph.error(initial))
print("final error = ", graph.error(result))

print("\nFactor Graph:\n{}".format(graph))
print("\nInitial Estimate:\n{}".format(initial))
print("Final Result:\n{}".format(result))

# plot graph
resultPoses = gtsam.utilities.extractPose2(result)
for i in range(resultPoses.shape[0]):
	gtsam_plot.plot_pose2(1, gtsam.Pose2(resultPoses[i, :]))
plt.show()

# output optimized graph results to file
outputfile = "example_graph_output.txt"
print("Writing results to file: ", outputfile)
graphNoKernel, _ = gtsam.readG2o(datafile, False)
gtsam.writeG2o(graphNoKernel, result, outputfile)
print ("Done!")


