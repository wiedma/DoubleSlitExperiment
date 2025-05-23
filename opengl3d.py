from manim import Prism
import numpy as np

class OpenGLPrism(Prism):
	"""
	OpenGL compatible variant of the Prism class
	"""
	def __init__(
		self, dimensions: tuple[float, float, float] | np.ndarray = [3, 2, 1], **kwargs
	) -> None:
		super().__init__(dimensions=dimensions, **kwargs)

	def init_points(self) -> None:
		"""Creates the sides of the :class:`Prism`."""
		super().init_points()
		for dim, value in enumerate(self.dimensions):
			self.rescale_to_fit(value, dim, stretch=True)
