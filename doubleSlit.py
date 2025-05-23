from manim import *
from complexSurface import ComplexSurface
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from opengl3d import OpenGLPrism


class Domain:
	"""
	Describes a 2D domain on which the finite difference solver can solve the SE.
	If the domain is discretized, a regular grid is generated in the bounding box of the domain.
	For each point, is_contained is used to test if it is inside of the domain.
	Only points inside the domain are used for the numerics.

	attributes:
		xlim (iterable of two floats): edges of the bounding box of the domain in x-direction
		ylim (iterable of two floats): edges of the bounding box of the domain in y-direction
		is_contained (callable float, float --> bool): takes in x and y from inside the bounding box and returns if (x, y) is contained in the domain
		is_discretized (bool): tracks if discretize() has been called yet
		to_flat_dict (dictionary): maps i, j indices on the regular grid inside the bounding box to flattened indices in the domain
		from_flat_dict (dictionary): maps flattened indices from the domain to i, j indices on the regular grid inside the bounding box
		x_space (numpy array): x coordinates of the regular grid in the bounding box
		y_space (numpy array): y coordinates of the regular grid in the bounding box
	"""
	def __init__(self, xlim, ylim, is_contained):
		self.xlim = xlim
		self.ylim = ylim
		self.is_contained = is_contained
		self.is_discretized = False


	def discretize(self, samples_x, samples_y):
		"""
		Initializes a grid on the domain.

		Args:
			samples_x (int): The number of grid points in the x-direction
			samples_y (int): The number of grid points in the y-direction
		"""
		x_space = np.linspace(self.xlim[0], self.xlim[1], samples_x)
		y_space = np.linspace(self.ylim[0], self.ylim[1], samples_y)

		to_flat_dict = {}
		from_flat_dict = {}

		counter = 0
		for i, x in enumerate(x_space):
			for j, y in enumerate(y_space):
				if self.is_contained(x, y):
					to_flat_dict[(i, j)] = counter
					from_flat_dict[counter] = (i, j)
					counter = counter + 1

		self.is_discretized = True
		self.to_flat_dict = to_flat_dict
		self.from_flat_dict = from_flat_dict
		self.x_space = x_space
		self.y_space = y_space

	
	def sample(self, func):
		"""
		Samples a function on the grid inside the domain
		
		Args:
			func (callable float, float --> float or complex): The function to be sampled

		Returns:
			numpy array: Flat numpy array of the function values sampled on the grid inside the domain
		"""
		if not self.is_discretized:
			raise ValueError("Can only sample on discretized domains")
		
		vec = []

		res_x = self.x_space.shape[0]
		res_y = self.y_space.shape[0]

		x_step = (self.xlim[1] - self.xlim[0]) / (res_x - 1)
		y_step = (self.ylim[1] - self.ylim[0]) / (res_y - 1)

		for x in self.x_space:
			for y in self.y_space:
				if self.is_contained(x, y):
					if (self.is_contained(x + x_step, y) and self.is_contained(x - x_step, y) and
						self.is_contained(x, y + y_step) and self.is_contained(x, y - y_step)):
						vec.append(func(x, y))
					else:
						vec.append(0)

		return np.array(vec)
	

	def data_to_regular_grid(self, data):
		"""
		Transforms a flat vector of function values to the corresponding 2D grid of values
		
		Args:
			data (numpy array): Flat numpy array of values for each grid point in the domain

		Returns:
			numpy array: 2D numpy array of the values from data aranged on the regular grid.
						 Values outside the domain are taken to be zero.
		"""
		res_x = self.x_space.shape[0]
		res_y = self.y_space.shape[0]
		out = np.zeros(shape=(res_x, res_y))

		for i in range(data.shape[0]):
			x, y = self.from_flat_dict[i]
			out[x, y] = data[i]
		
		return out


def build_finite_difference_kernel(domain, res_x, res_y):
	"""
	Builds the kernel for the finite difference solver.
	
	Args:
		domain (Domain): The domain on which the SE should be solved
		res_x (int): The number of samples in x-direction
		res_y (int): The number of samples in y-direction

	Returns:
		scipy sparse coo matrix: Matrix A, such that psi(t) = e^{At} psi(0)
	"""
		
	domain.discretize(res_x, res_y)

	x_step = (domain.xlim[1] - domain.xlim[0]) / (res_x - 1)
	y_step = (domain.ylim[1] - domain.ylim[0]) / (res_y - 1)

	to_flat_dict = domain.to_flat_dict

	kernel_rows = []
	kernel_cols = []
	kernel_vals = []
	
	for i in range(res_x):
		for j in range(res_y):

			# Check if interior point
			if (i-1, j) in to_flat_dict and (i+1, j) in to_flat_dict and (i, j-1) in to_flat_dict and (i, j+1) in to_flat_dict:

				# del_x^2
				kernel_rows.append(to_flat_dict[(i, j)])
				kernel_cols.append(to_flat_dict[(i+1, j)])
				kernel_vals.append(-1/2j*1/(x_step**2))

				kernel_rows.append(to_flat_dict[(i, j)])
				kernel_cols.append(to_flat_dict[(i, j)])
				kernel_vals.append(1j*1/(x_step**2))

				kernel_rows.append(to_flat_dict[(i, j)])
				kernel_cols.append(to_flat_dict[(i-1, j)])
				kernel_vals.append(-1/2j*1/(x_step**2))

				# del_y^2
				kernel_rows.append(to_flat_dict[(i, j)])
				kernel_cols.append(to_flat_dict[(i, j+1)])
				kernel_vals.append(-1/2j*1/(y_step**2))

				kernel_rows.append(to_flat_dict[(i, j)])
				kernel_cols.append(to_flat_dict[(i, j)])
				kernel_vals.append(1j*1/(y_step**2))

				kernel_rows.append(to_flat_dict[(i, j)])
				kernel_cols.append(to_flat_dict[(i, j-1)])
				kernel_vals.append(-1/2j*1/(y_step**2))

	
	kernel = coo_matrix((kernel_vals, (kernel_rows, kernel_cols)), shape=(len(domain.to_flat_dict), len(domain.to_flat_dict)))
	return kernel


def gaussian_wave_packet(x, x_0, p_0, delta_x, t):
	"""
	Defines the wavefunction of a Gaussian wavepacket in 1D
	"""
	return (
			1/np.sqrt(np.sqrt(np.pi/2)*(2 * delta_x + 1j*t/delta_x)) 
		 	* np.exp(-((x-x_0-p_0*t)**2)/(4*delta_x**2*(1 + 1j*t/(2*delta_x**2))))
		   	* np.exp(1j*p_0*(x - x_0 - p_0*t/2))
			)


def gaussian_wave_packet_2D(x, x_0, p_0, delta_x, t):
	"""
	Defines the wavefunction of a Gaussian wavepacket in 2D
	"""
	return gaussian_wave_packet(x[0], x_0[0], p_0[0], delta_x[0], t) * gaussian_wave_packet(x[1], x_0[1], p_0[1], delta_x[1], t)


def bilinear_interpolation(x, y, data, domain):
	"""
	Uses bilinear interpolation to get function values between the gridpoints.

	Args:
		x (float): x-coordinate at which the function should be evaluated. Must be inside bounding box of the domain.
		y (float): y-coordinate at which the function should be evaluated. Must be inside bounding box of the domain.
		data (numpy array): Flat array of function values on the gridpoints inside the domain
		domain (Domain): The domain on which the data was sampled.
	"""
	
	if not domain.is_discretized:
		raise ValueError("Data does not come from domain, since domain is not discretized!")

	res_x = domain.x_space.shape[0]
	res_y = domain.y_space.shape[0]

	x_step = (domain.xlim[1] - domain.xlim[0]) / (res_x - 1)
	y_step = (domain.ylim[1] - domain.ylim[0]) / (res_y - 1)

	x_1 = int((x - domain.xlim[0]) // x_step)
	x_2 = int((x - domain.xlim[0]) // x_step + 1)

	y_1 = int((y - domain.ylim[0]) // y_step)
	y_2 = int((y - domain.ylim[0]) // y_step + 1)

	f_11 = data[domain.to_flat_dict[(x_1, y_1)]] if (x_1, y_1) in domain.to_flat_dict else 0
	f_12 = data[domain.to_flat_dict[(x_1, y_2)]] if (x_1, y_2) in domain.to_flat_dict else 0
	f_21 = data[domain.to_flat_dict[(x_2, y_1)]] if (x_2, y_1) in domain.to_flat_dict else 0
	f_22 = data[domain.to_flat_dict[(x_2, y_2)]] if (x_2, y_2) in domain.to_flat_dict else 0

	x_1 = x_1 * x_step + domain.xlim[0]
	x_2 = x_2 * x_step + domain.xlim[0]
	y_1 = y_1 * y_step + domain.ylim[0]
	y_2 = y_2 * y_step + domain.ylim[0]

	return 1/(x_step * y_step) * (f_11*(x_2 - x)*(y_2 - y) + f_12*(x_2 - x)*(y - y_1)
							    + f_21*(x - x_1)*(y_2 - y) + f_22*(x - x_1)*(y - y_1))


class SESolverAnimation(Animation):
	"""
	Animates the dynamics given by the Schrödinger equation (SE)

	attributes:
		complexSurface (ComplexSurface): the surface mesh representing the wavefunction
		kernel (scipy sparse matrix): the finite difference kernel
		domain (Domain): the domain on which the SE is solved
		t_end (float): The final time of the evolution
	"""
	def __init__(self, complexSurface, kernel, domain, initial_cond, t_end=1, **kwargs):

		super().__init__(complexSurface, **kwargs)

		self.kernel = kernel
		self.domain = domain
		self.initial_cond = initial_cond
		self.t_end = t_end


	def interpolate_mobject(self, alpha):
		"""
		Calculates the state of the complexSurface for values of alpha [0, 1] corresponding to times [0, t_end]
		This method is called by the manim renderer to determine the state of the Scene for each frame.

		Args:
			alpha (float between 0 and 1): Clock of the animation. Interpolates between 0 (start of animation) and 1 (end of animation)
		"""

		data = expm_multiply(self.kernel*alpha*self.t_end, self.initial_cond)
		self.mobject.set_complex_func(lambda u,v: bilinear_interpolation(u, v, data, self.domain))



class DoubleSlitExperiment(ThreeDScene):
	"""
	The manim scene for the double slit experiment
	"""
	def construct(self):
		self.set_camera_orientation(phi=15 * DEGREES, theta=35 * DEGREES)

		x_lim = [-10, 10, 600]
		y_lim = [-10, 10, 600]

		initial_wf = lambda u, v: gaussian_wave_packet_2D((u, v), (-2, 0), (20, 0), (0.1, 0.1), 0)

		axes = ThreeDAxes(x_range=(-5, 5, 1), y_range=(-5, 5, 1), z_range=(0, 1, 0.5))

		domain = Domain(x_lim, y_lim, lambda x,y: abs(x) > 0.125 or (0.25 < y < 0.5 or -0.5 < y < -0.25))
		domain.discretize(samples_x=x_lim[2], samples_y=y_lim[2])
		kernel = build_finite_difference_kernel(domain, x_lim[2], y_lim[2])

		upper_wall = OpenGLPrism(dimensions=axes.c2p(0.25, 4.5, 0.001)).move_to(axes.c2p(0, 2.75, 0.0005))
		lower_wall = OpenGLPrism(dimensions=axes.c2p(0.25, 4.5, 0.001)).move_to(axes.c2p(0, -2.75, 0.0005))
		central_wall = OpenGLPrism(dimensions=axes.c2p(0.25, 0.5, 0.001)).move_to(axes.c2p(0, 0, 0.0005))

		initial_data = domain.sample(initial_wf)

		surface_plane = ComplexSurface(
			lambda u,v: bilinear_interpolation(u, v, initial_data, domain), axes, resolution=(300, 300),
			v_range=[-5, 5], u_range=[-5, 5], colors=[RED, ORANGE, YELLOW, GREEN, BLUE],
			fill_opacity=1
		)

		se_dynamics = SESolverAnimation(surface_plane, domain=domain,
								  	kernel=kernel, initial_cond=initial_data, t_end=0.5)
		
		self.add(axes, surface_plane)
		self.add(axes, upper_wall)
		self.add(axes, lower_wall)
		self.add(axes, central_wall)

		self.play(se_dynamics, run_time=10, rate_func=linear)


class SingleSlitExperiment(ThreeDScene):
	"""
	The manim scene for diffraction on a single slit
	"""
	def construct(self):
		self.set_camera_orientation(phi=15 * DEGREES, theta=35 * DEGREES)

		x_lim = [-10, 10, 600]
		y_lim = [-10, 10, 600]

		initial_wf = lambda u, v: gaussian_wave_packet_2D((u, v), (-2, 0), (20, 0), (0.1, 0.1), 0)

		axes = ThreeDAxes(x_range=(-5, 5, 1), y_range=(-5, 5, 1), z_range=(0, 1, 0.5))

		domain = Domain(x_lim, y_lim, lambda x,y: abs(x) > 0.125 or abs(y) < 0.125)
		domain.discretize(samples_x = x_lim[2], samples_y= y_lim[2])
		kernel = build_finite_difference_kernel(domain, x_lim[2], y_lim[2])

		upper_wall = OpenGLPrism(dimensions=axes.c2p(0.25, 4.875, 0.001)).move_to(axes.c2p(0, 5.125/2, 0.0005))
		lower_wall = OpenGLPrism(dimensions=axes.c2p(0.25, 4.875, 0.001)).move_to(axes.c2p(0, -5.125/2, 0.0005))

		initial_data = domain.sample(initial_wf)

		surface_plane = ComplexSurface(
			lambda u,v: bilinear_interpolation(u, v, initial_data, domain), axes, resolution=(100, 100),
			v_range=[-5, 5], u_range=[-5, 5],
			fill_opacity=1
		)

		se_dynamics = SESolverAnimation(surface_plane, domain=domain,
								  	kernel=kernel, initial_cond=initial_data, t_end=0.5)
		
		self.add(axes, surface_plane)
		self.add(axes, upper_wall)
		self.add(axes, lower_wall)

		self.play(se_dynamics, run_time=10, rate_func=linear)
