import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import itertools
import math
import sys

output_directory = 'output/bayes_filter_sim_discrete/'

# Define multi-dimensional version of np.random.choice()

def rnd_choice_multi_dim(indices_array, probability_array):
	return list(itertools.product(*indices_array))[
		np.random.choice(
			probability_array.size,
			p = probability_array.flatten()
		)
	]

# Define variables

x_names = [
	'pos_x',
	'pos_y'
]

x_names_long = [
	'$x$ position',
	'$y$ position'
]

x_values = [
	[10,20,30,40,50],
	[10,20,30,40,50]
]

y_names = [
	'LL',
	'UL',
	'LR',
	'UR'
]

y_names_long = [
	'Lower left',
	'Upper left',
	'Lower right',
	'Upper right'
]

y_values = [
	[0,10,20,30,40,50,60,70,80,90,100,110,120],
	[0,10,20,30,40,50,60,70,80,90,100,110,120],
	[0,10,20,30,40,50,60,70,80,90,100,110,120],
	[0,10,20,30,40,50,60,70,80,90,100,110,120]
]

# Calculate and store properties of the variable structure that we will need in
# multiple places


num_x_vars = len(x_names)
num_y_vars = len(y_names)

num_x_values_array = map(len, x_values)
num_y_values_array = map(len, y_values)

x_indices_array = map(range, num_x_values_array)
y_indices_array = map(range, num_y_values_array)

x_values_product = np.array(list(itertools.product(*x_values))).reshape(num_x_values_array + [num_x_vars])
x_values_squared_product = np.square(x_values_product)

# Define initial and conditional probabilities

# Define initial x probabilities

print '\nGenerating P(x_initial)...'

p_x_initial = np.empty(num_x_values_array, dtype = 'float')

for x_index in itertools.product(*x_indices_array):
	p_x_initial[x_index] = 1
p_x_initial = p_x_initial/np.sum(p_x_initial)

# Define x conditional on previous x

print '\nGenerating P(x | x_previous)...'

# Set size of movement at each time step
drift = 10.0

p_x_bar_x_previous = np.zeros(
	num_x_values_array + num_x_values_array,
	dtype = 'float'
)

for x_previous_index in itertools.product(*x_indices_array):
	for x_index in itertools.product(*x_indices_array):
		distance_squared = 0
		for x_var_index in range(num_x_vars):
			distance_squared += (
				x_values[x_var_index][x_index[x_var_index]] -
				x_values[x_var_index][x_previous_index[x_var_index]]
			)**2
		p_x_bar_x_previous[x_previous_index][x_index] = math.exp(
			-distance_squared/drift**2
		)
	p_x_bar_x_previous[x_previous_index] = p_x_bar_x_previous[x_previous_index]/np.sum(p_x_bar_x_previous[x_previous_index])

# Define y conditional on x

print '\nGenerating P(y | x)...'

# Quantify sensor noise
noise = 20.0

# Set positions of sensors
sensor_positions = [
	[0,0],
	[0,60],
	[60,0],
	[60,60]
]

# Generate probability distribution of sensor readings for every x position. We
# have sensors measure L(1) distance because it has good scaling properties and
# it's natural for this discrete case

p_y_bar_x = np.ones(num_x_values_array + num_y_values_array, dtype = 'float')

for x_index in itertools.product(*x_indices_array):
	sys.stdout.write(
		'\rGenerating P(y | x) for x_index = {}'.format(
			x_index
		)
	)
	sys.stdout.flush()
	for y_index in itertools.product(*y_indices_array):
		for y_var_index in range(num_y_vars):
			distance = 0
			for x_var_index in range(num_x_vars):
				distance += abs(
					x_values[x_var_index][x_index[x_var_index]] -
					sensor_positions[y_var_index][x_var_index]
				)
			p_y_bar_x[x_index][y_index] *= math.exp(
				-(y_values[y_var_index][y_index[y_var_index]] -
					distance)**2/noise**2
			)
	p_y_bar_x[x_index] = p_y_bar_x[x_index]/np.sum(p_y_bar_x[x_index])

# Analyze and visualize conditional probabilities

# Analyze and visualize P(x_initial)

print '\n\nAnalyzing P(x_initial)...'

print	'\nP(x_initial):'
print pd.DataFrame(
	p_x_initial,
	index = pd.MultiIndex.from_product([x_values[0]], names=[x_names[0]]),
	columns = pd.MultiIndex.from_product([x_values[1]], names=[x_names[1]])
)

print '\nSigma_x_initial P(x_initial) = {}'.format(np.sum(p_x_initial))

# Analyze and visulize P(x | x_previous)

print '\nAnalyzing P(x | x_previous)...'

# Choose random sample for x_previous
x_previous_index = (
	np.random.choice(x_indices_array[0]),
	np.random.choice(x_indices_array[1])
)
print '\nShowing results for ({}_previous, {}_previous) = ({}, {})'.format(
	x_names[0],
	x_names[1],
	x_values[0][x_previous_index[0]],
	x_values[1][x_previous_index[1]]
)

# Print P(x | x_previous)
print '\nP(x | {}_previous = {}, {}_previous = {}):'.format(
	x_names[0],
	x_values[0][x_previous_index[0]],
	x_names[1],
	x_values[1][x_previous_index[1]]
)

print pd.DataFrame(
	p_x_bar_x_previous[x_previous_index],
	index = pd.MultiIndex.from_product([x_values[0]], names=[x_names[0]]),
	columns = pd.MultiIndex.from_product([x_values[1]], names=[x_names[1]])
)

print '\nSigma_x P(x | {}_previous = {}, {}_previous = {}) = {}'.format(
	x_names[0],
	x_values[0][x_previous_index[0]],
	x_names[1],
	x_values[1][x_previous_index[1]],
	np.sum(p_x_bar_x_previous[x_previous_index])
)

# Plot P(x | x_previous)

plt.figure(figsize=(10,6))
fig = plt.gcf()
fig.subplots_adjust(right=0.7) # Make room on the right for the legend
plt.pcolor(
	p_x_bar_x_previous[x_previous_index].T,
	vmin = 0.0,
	vmax = 1.0,
	cmap=plt.cm.Blues,
	edgecolors='black'
)
plt.plot(
	x_previous_index[0] + 0.5,
	x_previous_index[1] + 0.5,
	'go',
	label = 'Previous position'
)
cb = plt.colorbar()
cb.set_label('P(position | previous position)')
plt.legend(
	loc = 'upper left',
	bbox_to_anchor = (1.25, 1) # Put the legend to the right of the main plot
)
plt.xlabel(x_names_long[0])
plt.ylabel(x_names_long[1])
plt.title('P(position | previous position)')
ax = plt.gca()
ax.set_yticks(
	np.arange((p_x_bar_x_previous[x_previous_index].shape[0])) + 0.5,
	minor=False
)
ax.set_xticks(
	np.arange((p_x_bar_x_previous[x_previous_index].shape[1])) + 0.5,
	minor=False
)
ax.set_xticklabels(x_values[1], minor=False)
ax.set_yticklabels(x_values[0], minor=False)
for tck in ax.xaxis.get_major_ticks():
    tck.tick1On = False
    tck.tick2On = False
for tck in ax.yaxis.get_major_ticks():
    tck.tick1On = False
    tck.tick2On = False

fig.savefig(output_directory + 'p_x_bar_x_previous.pdf')
fig.savefig(output_directory + 'p_x_bar_x_previous.png')
plt.close(fig)

# Analyze and visualizae P (y | x)

print '\nAnalyzing P(y | x)...'

# Choose random sample for x
x_index = (
	np.random.choice(x_indices_array[0]),
	np.random.choice(x_indices_array[1])
)
print '\nShowing results for ({}, {}) = ({}, {})'.format(
	x_names[0],
	x_names[1],
	x_values[0][x_index[0]],
	x_values[1][x_index[1]]
)

# Calculate marginal P(y | x) for each y variable
p_y_bar_x_marginal = np.zeros([num_y_vars, np.array(y_indices_array).shape[1]])
distance = np.zeros(num_y_vars)

for y_var_index in range(num_y_vars):
	for x_var_index in range(num_x_vars):
		distance[y_var_index] += abs(
			x_values[x_var_index][x_index[x_var_index]] -
			sensor_positions[y_var_index][x_var_index]
		)
	p_y_bar_x_marginal[y_var_index] = np.sum(
		p_y_bar_x[x_index],
		axis=tuple(
			range(num_y_vars)[:y_var_index] + range(num_y_vars)[y_var_index+1:]
		)
	)

# Print marginal P(y | x) for each y variable
for y_var_index in range(num_y_vars):
	print '\nSensor: {} ({}, {})'.format(
		y_names[y_var_index],
		sensor_positions[y_var_index][0],
		sensor_positions[y_var_index][1]
	)
	print 'Distance = {}'.format(distance[y_var_index])
	print '\nP({} | {} = {}, y_pos = {}):'.format(
		y_names[y_var_index],
		x_names[0],
		x_values[0][x_index[0]],
		x_values[1][x_index[1]]
	)
	print pd.DataFrame(
		p_y_bar_x_marginal[y_var_index],
		index = pd.MultiIndex.from_product([y_values[y_var_index]], names=[y_names[y_var_index]]),
		columns = ['P']
	)
	print '\nSigma_{} P({} | {} = {}, y_pos = {}) = {}'.format(
		y_names[y_var_index],
		y_names[y_var_index],
		x_names[0],
		x_values[0][x_index[0]],
		x_values[1][x_index[1]],
		np.sum(p_y_bar_x[x_index])
	)

# Plot marginal P(y | x) for each y variable
plt.figure(figsize=(10,6))
fig = plt.gcf()
fig.subplots_adjust(right=0.7) # Make room on the right for the legend
for y_var_index in range(num_y_vars):
	plt.subplot(4, 1, y_var_index + 1)
	pc = plt.pcolor(
		np.expand_dims(p_y_bar_x_marginal[y_var_index].T, axis=0),
		vmin = 0.0,
		vmax = 1.0,
		cmap=plt.cm.Blues,
		edgecolors='black'
	)
	plt.plot(
		distance[y_var_index]/10 + 0.5,
		0.5,
		'go',
		label = 'Actual distance'
	)
	if y_var_index == 0:
		plt.legend(
			loc = 'upper left',
			bbox_to_anchor = (1.16, 1) # Put the legend to the right of the main plot
		)
	plt.ylabel(y_names_long[y_var_index])
	ax = plt.gca()
	ax.set_yticks([])
	if y_var_index == num_y_vars - 1:
		plt.xlabel('Measured distance')
		ax.set_xticks(
			np.arange((p_y_bar_x_marginal[y_var_index].shape[0])) + 0.5,
			minor=False
		)
		ax.set_xticklabels(y_values[y_var_index], minor=False)
		for tck in ax.xaxis.get_major_ticks():
		    tck.tick1On = False
		    tck.tick2On = False
	else:
		ax.set_xticks([])
plt.suptitle('P(measured distances | position)')
cbar_ax = fig.add_axes([0.72, 0.12, 0.03, 0.75])
cb = fig.colorbar(pc, cax=cbar_ax)
cb.set_label('P(measured distance | position)')

fig.savefig(output_directory + 'p_y_bar_x.pdf')
fig.savefig(output_directory + 'p_y_bar_x.png')
plt.close(fig)

# Generate simulated x data and y data

print '\nGenerating simulated x data and y data...'

# Set time variable parameters

tinitial = 0
deltat = 1
num_timesteps = 50

# Initialize data objects

t = np.zeros(num_timesteps, dtype = 'float')
x_index_t = np.zeros([num_timesteps, num_x_vars], dtype = 'int')
y_index_t = np.zeros([num_timesteps, num_y_vars], dtype = 'int')

# Set random seed (for reproducibility)
# np.random.seed(654321)

# Define initial values

t[0] = tinitial
x_index_t[0] = rnd_choice_multi_dim(
	x_indices_array,
	p_x_initial
)
y_index_t[0] = rnd_choice_multi_dim(
	y_indices_array,
	p_y_bar_x[tuple(x_index_t[0])]
)

# Generate simulated data

for t_index in range(1,num_timesteps):
	sys.stdout.write(
		'\rGenerating timestep {}/{}'.format(
			t_index +1,
			num_timesteps
		)
	)
	sys.stdout.flush()
	t[t_index] = t[t_index- 1] + deltat
	x_index_t[t_index] = rnd_choice_multi_dim(
		x_indices_array,
		p_x_bar_x_previous[tuple(x_index_t[t_index - 1])]
	)
	y_index_t[t_index] = rnd_choice_multi_dim(
		y_indices_array,
		p_y_bar_x[tuple(x_index_t[t_index])]
	)

# Analyze and visualize simulated x data and y data

print '\n\nAnalyzing simulated x data and y data...'

x_value_t = np.zeros([num_timesteps, num_x_vars], dtype = 'float')
y_value_t = np.zeros([num_timesteps, num_y_vars], dtype = 'float')

# Calculate x and y values from indices
for t_index in range(num_timesteps):
	for x_var_index in range(num_x_vars):
		x_value_t[t_index][x_var_index] = x_values[x_var_index][x_index_t[t_index][x_var_index]]
	for y_var_index in range(num_y_vars):
		y_value_t[t_index][y_var_index] = y_values[y_var_index][y_index_t[t_index][y_var_index]]

# Print a sample of the data

print_start = 0
print_stop = 30

print '\n'
print pd.concat([
	pd.DataFrame(t, columns = ['t']),
	pd.DataFrame(x_value_t, columns = x_names),
	pd.DataFrame(y_value_t, columns = y_names)
], axis = 1)[print_start:print_stop]

# Calculate and print sampled distribution of x steps
p_step = np.zeros(
	[(num_x_values_array[0] - 1)*2 + 1, (num_x_values_array[1] - 1)*2 + 1],
	dtype='float'
)
for t_index in range(1, num_timesteps):
	p_step[
		(num_x_values_array[0] - 1) + x_index_t[t_index][0] - x_index_t[t_index-1][0],
		(num_x_values_array[0] - 1)+ x_index_t[t_index][1] - x_index_t[t_index-1][1]
	] += 1
p_step = p_step/np.sum(p_step)
print '\n p(x step):'
print pd.DataFrame(
	p_step,
	index = range(-(num_x_values_array[0] - 1), num_x_values_array[0]),
	columns = range(-(num_x_values_array[1] - 1), num_x_values_array[1])
)

# Calculate actual distances and compare with y values
distance_t = np.zeros([num_timesteps, num_y_vars], dtype='int')
error_t = np.zeros([num_timesteps, num_y_vars], dtype='int')
error_distribution = np.zeros(25, dtype='float')

for t_index in range(num_timesteps):
	for y_var in range(num_y_vars):
		distance_t[t_index][y_var] = int(
			abs(x_values[0][x_index_t[t_index][0]] - sensor_positions[y_var][0]) +
			abs(x_values[1][x_index_t[t_index][1]] - sensor_positions[y_var][1])
		)
		error_t[t_index][y_var] = int(y_value_t[t_index][y_var]) - distance_t[t_index][y_var]
		error_distribution[
			(num_y_values_array[0] - 1) + error_t[t_index][y_var]/10
		] += 1

# Print distribution of sensor errors
print '\nSensor error distribution:'
print pd.DataFrame(
	error_distribution/np.sum(error_distribution),
	index = range(
		-(num_y_values_array[0] - 1),
		num_y_values_array[0]
	)
)

# Use Bayesian filter to calculate posterior probability distribution of x
# conditional on y data

print '\nCalculating posterior probability distribution of x conditional on simulated y data...'

print '\nCalculating exact solution P(x)(t)...'

print '\nInitializing P(x)(t)...'
# Initialize data objects

p_x = np.zeros([num_timesteps] + num_x_values_array, dtype = 'float')

# Calculate posterior probability distribution of x at t=0

print '\nCalculating P(x)(t=0)...'

for x_index in itertools.product(*x_indices_array):
	p_x[0][x_index] = p_y_bar_x[x_index][tuple(y_index_t[0])]*p_x_initial[x_index]
p_x[0] = p_x[0]/np.sum(p_x[0])

print '\nCalculating approximate solution using particle filter...'

# Generate particles for t=0

num_particles = 50

print '\nInitializing particle filter with {} particles...'.format(num_particles)

particle_indices = np.zeros((num_timesteps, num_particles, num_x_vars), dtype = 'int')
particle_values = np.zeros((num_timesteps, num_particles, num_x_vars), dtype = 'float')
particle_weights = np.zeros((num_timesteps, num_particles), dtype = 'float')

# print '\nnum_particles = {}'.format(num_particles)
# print 'type(num_particles) = {}'.format(type(num_particles))
#
# print '\ntype(particle_indices) = {}'.format(type(particle_indices))
# print 'particle_indices.dtype = {}'.format(particle_indices.dtype)
# print 'particle_indices.shape = {}'.format(particle_indices.shape)
# print 'particle_indices.size = {}'.format(particle_indices.size)
#
# print '\ntype(particle_values) = {}'.format(type(particle_values))
# print 'particle_values.dtype = {}'.format(particle_values.dtype)
# print 'particle_values.shape = {}'.format(particle_values.shape)
# print 'particle_values.size = {}'.format(particle_values.size)
#
# print '\ntype(particle_weights) = {}'.format(type(particle_weights))
# print 'particle_weight.dtype = {}'.format(particle_weights.dtype)
# print 'particle_weight.shape = {}'.format(particle_weights.shape)
# print 'particle_weight.size = {}'.format(particle_weights.size)

print '\nGenerating particles for t = 0...'

for particle_index in range(num_particles):
	particle_indices[0, particle_index] = rnd_choice_multi_dim(
		x_indices_array,
		p_x_initial
	)
	for var_index in range(num_x_vars):
		particle_values[0, particle_index, var_index] = x_values[var_index][particle_indices[0, particle_index, var_index]]
	particle_weights[0, particle_index] = p_y_bar_x[tuple(particle_indices[0,particle_index])][tuple(y_index_t[0])]
particle_weights[0] = particle_weights[0]/np.sum(particle_weights[0])

# Analyze and visualize posterior probability distribution of x at t=0

print '\nAnalyzing and visualizing solution for t = 0...'

# Print posterior probability distribution of x at t=0 (exact solution)

print '\nx(t=0) = ({}, {})'.format(
	x_values[0][x_index_t[0][0]],
	x_values[1][x_index_t[0][1]]
)

print '\nExact solution P(x)(t=0):'
print pd.DataFrame(
	p_x[0],
	index = pd.MultiIndex.from_product([x_values[0]], names=[x_names[0]]),
	columns = pd.MultiIndex.from_product([x_values[1]], names=[x_names[1]])
)

print '\nSigma_x P(x)(t=0) = {}'.format(np.sum(p_x[0]))

x_mean_exact = np.zeros([num_timesteps, num_x_vars], dtype = 'float')
x_sd_exact = np.zeros([num_timesteps, num_x_vars], dtype = 'float')

prob_reshaped = np.repeat(p_x[0], 2).reshape(p_x[0].shape + tuple([2]))

x_mean_exact[0] = np.average(x_values_product, axis=(0,1), weights=prob_reshaped)
x_squared_mean_exact = np.average(x_values_squared_product, axis=(0,1), weights=prob_reshaped)
x_sd_exact[0] = np.sqrt(x_squared_mean_exact - np.square(x_mean_exact[0]))

print '\nExact solution mean = {}'.format(x_mean_exact[0])
print 'Exact solution SD = {}'.format(x_sd_exact[0])

print '\nParticles for t=0:'
print pd.concat([
	pd.DataFrame(particle_indices[0], columns = x_names),
	pd.DataFrame(particle_values[0], columns = x_names),
	pd.DataFrame(particle_weights[0], columns = ['weight'])],
	axis = 1)

print '\nSigma_i weight_i = {}'.format(np.sum(particle_weights[0]))

x_mean_particle = np.zeros((num_timesteps, num_x_vars), dtype = 'float')
x_sd_particle = np.zeros((num_timesteps, num_x_vars), dtype = 'float')

x_mean_particle[0] = np.average(particle_values[0], axis=0, weights=particle_weights[0])
x_squared_mean_particle = np.average(np.square(particle_values[0]), axis=0, weights=particle_weights[0])
x_sd_particle[0] = np.sqrt(x_squared_mean_particle - np.square(x_mean_particle[0]))

print '\nParticle mean = {}'.format(x_mean_particle[0])
print 'Particle SD = {}'.format(x_sd_particle[0])

print '\nPlotting solution for t = 0...'
# Plot posterior probability distribution of x at t=0
plt.figure(figsize=(10,6))
fig = plt.gcf()
ax = plt.gca()
fig.subplots_adjust(right=0.7) # Make room on the right for the legend
plt.pcolormesh(
	p_x[0].T,
	vmin = 0.0,
	vmax = 1.0,
	cmap=plt.cm.Blues,
	edgecolors='black'
)
plt.plot(
	x_index_t[0][0] + 0.5,
	x_index_t[0][1] + 0.5,
	'go',
	label = 'Actual position'
)
# plt.plot(
# 	x_mean_exact[0][0]/10 - 1 + 0.5,
# 	x_mean_exact[0][1]/10 - 1 + 0.5,
# 	'bo',
# 	label = 'Conf region: exact'
# )
cr_exact = Ellipse(xy=x_mean_exact[0]/10 - 1 + 0.5, width=x_sd_exact[0][0]*2/10, height=x_sd_exact[0][1]*2/10)
ax.add_artist(cr_exact)
cr_exact.set_clip_box(ax.bbox)
cr_exact.set_alpha(0.3)
cr_exact.set_facecolor('blue')
# plt.plot(
# 	x_mean_particle[0][0]/10 - 1 + 0.5,
# 	x_mean_particle[0][1]/10 - 1 + 0.5,
# 	'yo',
# 	label = 'Conf region: particle'
# )
cr_particle = Ellipse(xy=x_mean_particle[0]/10 - 1 + 0.5, width=x_sd_particle[0][0]*2/10, height=x_sd_particle[0][1]*2/10)
ax.add_artist(cr_particle)
cr_particle.set_clip_box(ax.bbox)
cr_particle.set_alpha(0.3)
cr_particle.set_facecolor('yellow')
cb = plt.colorbar()
cb.set_label('P(position| measured distances)')
plt.legend(
	loc = 'upper left',
	bbox_to_anchor = (1.25, 1) # Put the legend to the right of the main plot
)
plt.xlabel(x_names_long[0])
plt.ylabel(x_names_long[1])
plt.title('P(position($t = 0$) | measured distances ($t = 0$))')
ax = plt.gca()
ax.set_yticks(
	np.arange((p_x[0].shape[0])) + 0.5,
	minor=False
)
ax.set_xticks(
	np.arange((p_x[0].shape[1])) + 0.5,
	minor=False
)
ax.set_xticklabels(x_values[1], minor=False)
ax.set_yticklabels(x_values[0], minor=False)
for tck in ax.xaxis.get_major_ticks():
    tck.tick1On = False
    tck.tick2On = False
for tck in ax.yaxis.get_major_ticks():
    tck.tick1On = False
    tck.tick2On = False

fig.savefig(output_directory + 'p_x_0_bar_y_0.pdf')
fig.savefig(output_directory + 'p_x_0_bar_y_0.png')
plt.close(fig)

# Calculate posterior probability distribution of x for t > 0

print '\nCalculating exact solution P(x)(t>0)...'

for t_index in range(1, num_timesteps):
	sys.stdout.write(
		'\rCalculating timestep {}/{}'.format(
			t_index + 1,
			num_timesteps
		)
	)
	sys.stdout.flush()
	for x_index in itertools.product(*x_indices_array):
		for x_previous_index in itertools.product(*x_indices_array):
			p_x[t_index][x_index] += np.product([
				p_y_bar_x[x_index][tuple(y_index_t[t_index])],
				p_x_bar_x_previous[x_previous_index][x_index],
				p_x[t_index-1][x_previous_index]
			])
	p_x[t_index] = p_x[t_index]/np.sum(p_x[t_index])
	prob_reshaped = np.repeat(p_x[t_index], 2).reshape(p_x[t_index].shape + tuple([2]))
	x_mean_exact[t_index] = np.average(x_values_product, axis=(0,1), weights=prob_reshaped)
	x_squared_mean_exact = np.average(x_values_squared_product, axis=(0,1), weights=prob_reshaped)
	x_sd_exact[t_index] = np.sqrt(x_squared_mean_exact - np.square(x_mean_exact[t_index]))

# Calculating particles for t > 0

print '\n\nGenerating particles for t > 0...'

for t_index in range(1, num_timesteps):
	for particle_index in range(num_particles):
		previous_particle = particle_indices[t_index-1, np.random.choice(num_particles, p=particle_weights[t_index-1])]
		# if t_index==1:
		# 	print 'previous_particle = {}'.format(previous_particle)
		particle_indices[t_index, particle_index] = rnd_choice_multi_dim(
			x_indices_array,
			p_x_bar_x_previous[tuple(previous_particle)]
			)
		# if t_index==1 :
		# 	print 'particle_indices[{}, {}] = {}'.format(t_index, particle_index, particle_indices[t_index, particle_index])
		for var_index in range(num_x_vars):
			particle_values[t_index, particle_index, var_index] = x_values[var_index][particle_indices[t_index, particle_index, var_index]]
		# if t_index==1 :
		# 	print 'particle_values[{}, {}] = {}'.format(t_index, particle_index, particle_values[t_index, particle_index])
		particle_weights[t_index, particle_index] = p_y_bar_x[tuple(particle_indices[t_index,particle_index])][tuple(y_index_t[t_index])]
	particle_weights[t_index] = particle_weights[t_index]/np.sum(particle_weights[t_index])
	x_mean_particle[t_index] = np.average(particle_values[t_index], axis=0, weights=particle_weights[t_index])
	x_squared_mean_particle = np.average(np.square(particle_values[t_index]), axis=0, weights=particle_weights[t_index])
	x_sd_particle[t_index] = np.sqrt(x_squared_mean_particle - np.square(x_mean_particle[t_index]))

# Analyze and visualize posterior probability distribution of x conditional on
# y data

print '\nAnalyzing and visualizing solution for t > 0...'

# Calculate posterior mean and standard deviation of x variables

# for t_index in range(1, num_timesteps):
# 	for x_var in range(num_x_vars):
# 		p_marginal = np.sum(
# 			p_x[t_index],
# 			axis = tuple(range(num_x_vars)[:x_var] + range(num_x_vars)[x_var+1:])
# 		)
# 		x_mean_exact[t_index][x_var] = np.inner(p_marginal, x_values[x_var])
# 		x_sd_exact[t_index][x_var] = math.sqrt(
# 			np.inner(
# 				p_marginal,
# 				np.square(np.subtract(x_values[x_var], x_mean_exact[t_index][x_var]))
# 			)
# 		)

# Print sample of data with posterior mean and standard deviation of x variables

# print pd.concat([
# 	pd.DataFrame(x_value_t, columns = x_names),
# 	pd.DataFrame(x_mean_exact, columns = [x_name + '_mean_exact' for x_name in x_names]),
# 	# pd.DataFrame(x_sd_exact, columns = [x_name + '_sd' for x_name in x_names]),
# 	pd.DataFrame(x_mean_particle, columns = [x_name + '_mean_part' for x_name in x_names])
# ], axis = 1)[print_start:print_stop]
for x_var_index in range(num_x_vars):
	print '\n{}:'.format(x_names[x_var_index])
	print pd.concat([
		pd.DataFrame(x_value_t[:,x_var_index], columns = ['Actual']),
		pd.DataFrame(x_mean_exact[:,x_var_index], columns = ['Mean_exact']),
		pd.DataFrame(x_mean_particle[:,x_var_index], columns = ['Mean_particle']),
		pd.DataFrame(x_sd_exact[:,x_var_index], columns = ['SD_exact']),
		pd.DataFrame(x_sd_particle[:,x_var_index], columns = ['SD_particle'])
	], axis = 1)[print_start:print_stop]

# Plot sample of mean and standard deviation of x variables

plot_start = 0
plot_end = 30

plt.figure(figsize=(7.5,10))
fig = plt.gcf()
fig.subplots_adjust(right=0.7) # Make room on the right for the legend
for x_var_index in range(num_x_vars):
	plt.subplot(num_x_vars + num_y_vars, 1, x_var_index + 1)
	actual_line = plt.plot(
		t[plot_start:plot_end],
		x_value_t[plot_start:plot_end, x_var_index],
		'g-',
		label = 'Actual position'
	)
	# mean_exact_line = plt.plot(
	# 	t[plot_start:plot_end],
	# 	x_mean_exact[plot_start:plot_end, x_var_index],
	# 	'b-',
	# 	label = 'Exact mean'
	# )
	confidence_region_exact = plt.fill_between(
		t[plot_start:plot_end],
		np.subtract(
			x_mean_exact[plot_start:plot_end, x_var_index],
			x_sd_exact[plot_start:plot_end, x_var_index]
		),
		np.add(
			x_mean_exact[plot_start:plot_end, x_var_index],
			x_sd_exact[plot_start:plot_end, x_var_index]
		),
		color = 'blue',
		alpha = 0.3,
		label = 'Confidence region: exact'
	)
	# mean_particle_line = plt.plot(
	# 	t[plot_start:plot_end],
	# 	x_mean_particle[plot_start:plot_end, x_var_index],
	# 	'y-',
	# 	label = 'Particle mean'
	# )
	confidence_region_particle = plt.fill_between(
		t[plot_start:plot_end],
		np.subtract(
			x_mean_particle[plot_start:plot_end, x_var_index],
			x_sd_particle[plot_start:plot_end, x_var_index]
		),
		np.add(
			x_mean_particle[plot_start:plot_end, x_var_index],
			x_sd_particle[plot_start:plot_end, x_var_index]
		),
		color = 'yellow',
		alpha = 0.3,
		label = 'Confidence region: particle'
	)
	plt.ylabel(x_names_long[x_var_index])
	ax =plt.gca()
	ax.set_xticks([])
	if x_var_index == 0:
		plt.title('P(position | measured distances)')
		plt.legend(loc = 'upper left', bbox_to_anchor = (1.05, 1))
for y_var_index in range(num_y_vars):
	plt.subplot(
		num_x_vars + num_y_vars,
		1,
		num_x_vars + y_var_index + 1
	)
	actual_distance = plt.plot(
		t[plot_start:plot_end],
		distance_t[plot_start:plot_end, y_var_index],
		'g-',
		label = 'Actual distance'
	)
	measured_distance = plt.plot(
		t[plot_start:plot_end],
		y_value_t[plot_start:plot_end, y_var_index],
		'bo',
		label = 'Measured distance'
	)
	plt.ylabel(y_names_long[y_var_index])
	ax =plt.gca()
	if y_var_index == 0:
		plt.title('Sensor readings')
		plt.legend(loc = 'upper left', bbox_to_anchor = (1.05, 1))
	if y_var_index == num_y_vars - 1:
		plt.xlabel('Time (s)')
	else:
		ax.set_xticks([])

fig.savefig(output_directory + 'p_x_t_bar_y_t.pdf')
fig.savefig(output_directory + 'p_x_t_bar_y_t.png')
plt.close(fig)

# Generate animation of P(x(t)| y(t))

print '\nGenerating animation of P(x(t) | y(0),...,y(t))'
animation_start = 0
animation_end = 30

plt.figure(figsize=(10,6))
fig = plt.gcf()
fig.subplots_adjust(right=0.7) # Make room on the right for the legend
pcm = plt.pcolormesh(
	p_x[0].T,
	vmin = 0.0,
	vmax = 1.0,
	cmap=plt.cm.Blues,
	edgecolors='black'
)
ln, = plt.plot(
	[],
	[],
	'go',
	label = 'Actual position'
)
cb = plt.colorbar()
cb.set_label('P(position | measured distances)')
plt.legend(
	loc = 'upper left',
	bbox_to_anchor = (1.25, 1) # Put the legend to the right of the main plot
)
plt.xlabel('$x$ position')
plt.ylabel('$y$ position')
title_text = plt.title('P(position($t = 0$) | measured distances ($t = 0$))')
ax = plt.gca()
ax.set_yticks(
	np.arange((p_x[0].shape[0])) + 0.5,
	minor=False
)
ax.set_xticks(
	np.arange((p_x[0].shape[1])) + 0.5,
	minor=False
)
ax.set_xticklabels(x_values[1], minor=False)
ax.set_yticklabels(x_values[0], minor=False)
for tck in ax.xaxis.get_major_ticks():
    tck.tick1On = False
    tck.tick2On = False
for tck in ax.yaxis.get_major_ticks():
    tck.tick1On = False
    tck.tick2On = False

# plt.show()
# sys.exit()

def init():
	pcm.set_array([])
	ln.set_data([],[])
	title_text.set_text('')
	return pc, ln, title_text

def animate(i):
	pcm.set_array(p_x[i].T.ravel())
	ln.set_data(x_index_t[i][0] + 0.5, x_index_t[i][1] + 0.5)
	title_text.set_text('P(position($t = {}$) | measured distances ($t = 0,\ldots, {}$))'.format(int(t[i]),int(t[i])))
	return pc, ln, title_text

anim = animation.FuncAnimation(
	fig,
	animate,
	init_func = init,
	frames = animation_end,
	interval = 500,
	blit = False,
	repeat = False)

plt.rcParams['animation.ffmpeg_path'] = u'C:\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = animation.FFMpegWriter(fps = 2)
anim.save(output_directory + 'p_x_t_bar_y_t.mp4', writer = FFwriter)

# plt.show()
plt.close(fig)
