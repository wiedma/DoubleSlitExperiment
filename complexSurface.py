from manim import Surface, config, RendererType, color_to_rgba, HSV, ManimColor
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
import numpy as np

class ComplexSurface(Surface, metaclass=ConvertToOpenGL):
	"""
	Surface that visualizes a complex function.
	Height of the surface corrsponds to the modulus of the complex function.
	Phase information is encoded in the color.
	"""

	def __init__(self, func, axes, u_range=[0, 1], v_range=[0, 1], resolution=32,
			   surface_piece_config={}, fill_opacity=1.0, should_make_jagged=False,
			   pre_function_handle_to_anchor_scale_factor=1e-05, **kwargs):
		
		surf_func = lambda u,v: axes.c2p(u, v, np.abs(func(u, v)))

		self.axes = axes
		self.complex_func = func

		if config.renderer == RendererType.OPENGL:
			super().__init__(surf_func, u_range=u_range, v_range=v_range, resolution=resolution,
					surface_piece_config=surface_piece_config, fill_opacity=fill_opacity,
					should_make_jagged=should_make_jagged, axes=axes,
					pre_function_handle_to_anchor_scale_factor=pre_function_handle_to_anchor_scale_factor,
					**kwargs)
		else:
			super().__init__(surf_func, u_range=u_range, v_range=v_range, resolution=resolution,
					surface_piece_config=surface_piece_config, fill_opacity=fill_opacity,
					should_make_jagged=should_make_jagged,
					pre_function_handle_to_anchor_scale_factor=pre_function_handle_to_anchor_scale_factor,
					**kwargs)
		

		self.set_fill_by_phase()
		
	
	def set_fill_by_phase(self):

		if config.renderer == RendererType.OPENGL:
			self.refresh_shader_data()
			return self

		for mob in self.family_members_with_points():
			axis_values = self.axes.point_to_coords(mob.get_midpoint())
			axis_value = np.angle(self.complex_func(axis_values[0], axis_values[1]))
			mob_color = HSV(((axis_value + np.pi)/(2*np.pi), 1, min(4*axis_values[2], 1, 1 if axis_values[2] > 1e-6 else 0)))
			mob.set_color(ManimColor(color_to_rgba(mob_color)), family=False)

		return self
	

	def set_complex_func(self, func, **kwargs):
		self.complex_func = func
		surf_func = lambda u,v: self.axes.c2p(u, v, np.abs(func(u, v)))

		if config.renderer == RendererType.OPENGL:
			self.passed_uv_func = surf_func
			self.init_points()
		else:
			self._func = surf_func
			self._setup_in_uv_space()
			self.apply_function(lambda p: surf_func(p[0], p[1]))
			if self.should_make_jagged:
				self.make_jagged()

		self.set_fill_by_phase()


	def _get_color_by_value(self, s_points):
		"""Matches each vertex to a color associated by its complex phase

		Only used for OpenGLSurface compatability

		Parameters
		----------
		s_points
			The vertices of the surface.

		Returns
		-------
		List
			A list of colors matching the vertex inputs.
		"""

		return_colors = []
		for point in s_points:
			axis_values = self.axes.point_to_coords(point)
			axis_value = np.angle(self.complex_func(axis_values[0], axis_values[1]))
			temp_color = HSV(((axis_value + np.pi)/(2*np.pi), 1, min(4*axis_values[2], 1, 1 if axis_values[2] > 1e-6 else 0)))
			return_colors.append(color_to_rgba(temp_color))

		return return_colors
	

	def get_shader_data(self):
		"""Called by parent Mobject to calculate and return
		the shader data.

		Returns
		-------
		shader_dtype
			An array containing the shader data (vertices and
			color of each vertex)
		"""
		s_points, du_points, dv_points = self.get_surface_points_and_nudged_points()
		shader_data = np.zeros(len(s_points), dtype=self.shader_dtype)
		shader_data["point"] = s_points
		shader_data["du_point"] = du_points
		shader_data["dv_point"] = dv_points

		self.color_by_val = self._get_color_by_value(s_points)
		shader_data["color"] = self.color_by_val

		return shader_data
		
