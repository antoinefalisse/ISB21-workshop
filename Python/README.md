Python version of the example.
==============================

# Install requirements:
- Install [Anaconda](https://docs.anaconda.com/anaconda/install/).
- Open an anaconda prompt.
- (Recommended) Create a specific conda environment for the workshop: `conda create -n isb2021 python=3.7 pip spyder` (Python > 3.7 will cause issues on Mac).
	- Activate the environment: `conda activate isb2021`.
- Navigate to the Python folder of ISB21-workshop: `cd <my_directory>/ISB21-workshop/Python` where `<my_directory>` is the path to the folder where ISB21-workshop is located.
- Install the required packages: `python -m pip install -r requirements.txt`.

# Run the code:
- Open an anaconda prompt.
- (Depending on your installlation, see above) Activate the environment: `conda activate isb2021`.
- Navigate to the Python folder of ISB21-workshop: `cd <my_directory>/ISB21-workshop/Python` where `<my_directory>` is the path to the folder where ISB21-workshop is located.
    - Option 1: You can run the code using the following command line: `python main.py`. You can edit the scripts in your favorite text editor (e.g., Notepad ++).
    - Option 2: You can run the code in your favorite IDE such as spyder. To launch spyder from the command line, type `spyder` in the anaconda prompt. You can then run `main.py`. To get interactive plots and see the animation, change the backend to automatic: Tools > preferences > IPython console > Graphics > Graphics backend > Backend: Automatic.
    - Option 3: You can also use the Jupyter notebook version of the code. To open the notebook from the command line, type `ipython notebook main.ipynb` in the anaconda prompt. Note that getting the animation to play properly might require you to install [FFmpeg](https://www.ffmpeg.org/) and have it in your environment path. By default, the animation will not play. 
- (Mac OS): you might run into security issues related to CasADi. If so, please follow the steps described under **If you want to open an app that hasn’t been notarized or is from an unidentified developer** on [this webpage](https://support.apple.com/en-us/HT202491).

# A couple of notes:
- `main.py` (or `main.ipynb`) contains the main script to formulate and solve the trajectory optimization problem. Running this script should take about 1-2s. 
- Different pre-defined walking styles can be generated by adjusting the variable `selected_gait`.
- Python > 3.7 will cause issues on Mac.
- To participate in our challenge, set the variable `challenge` to `True` (see line 48 in `main.py`).
- If you get troubles with getting the code to run, please post an issue on Github or contact Antoine (afalisse@stanford.edu).
