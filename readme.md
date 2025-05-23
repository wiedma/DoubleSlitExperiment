# Installation
If you use conda, then create a new environment from the conda shell with

	conda create --name <name> --file requirements_conda.txt

Where you replace <name> with the name you want to choose for your conda environment.  
This should automatically install the necessary packages.

If you prefer to use venv, you need to run the following commands in your command line.

	python3 -m venv <name>
	source <name>/bin/activate
	pip install -r requirements_pip.txt

If this does not work try to install the necessary packages manually.

You should only need to install

- manim
- scipy
- IPython

# Rendering
If you want manim to render the animations you need to execute the following command in your command line  

	manim -qm --renderer=opengl --disable_caching --write_to_movie doubleSlit.py DoubleSlitExperiment

The -qm flag tells manim to render in medium quality (720p 30fps).  
You could also render in higher quality (see manim documentation), but for some reason the colors look  
really ugly if you render in 60fps. I don't fully know why.

The last argument refers to the Scene that should be rendered.

For more details, refer to the manim documentation at https://www.manim.community/
