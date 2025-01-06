# ArcticBound: Ice Layer Detection System
### Overview
The program processes a given grayscale image and computes:

- Simple Bayesian Method: Identifies layer boundaries using local maxima in edge strength.
- Viterbi Algorithm (HMM): Employs probabilistic modeling to find the most likely sequence of boundary positions.
- Human-Traced Viterbi Algorithm: Improves Viterbi results by incorporating user-provided feedback points.

**The results include:**

Layer boundaries visualized with different colors:
- Yellow: Simple Bayesian
- Blue: Viterbi (HMM)
- Red: Human-Traced Viterbi

### Files Generated
- `air_ice_output.png`: Visualization of air-ice boundary predictions.
- `ice_rock_output.png`: Visualization of ice-rock boundary predictions.
- `layers_output.txt`: Contains numerical values for boundaries computed using the three methods.
### How It Works
* Edge Strength Map:
  - An edge detection algorithm computes the edge strength for each pixel in the input image.

* Simple Bayesian Method:
  - For each column, find the two highest edge strength values separated by a threshold distance.
  - Assign the upper boundary to air-ice and the lower to ice-rock.
* Viterbi Algorithm:
  - Uses emission probabilities derived from the edge strength map.
  - Models state transitions with decreasing probabilities for larger positional changes.
* Human-Traced Viterbi Algorithm:
  - Incorporates user-provided points for the air-ice or ice-rock boundaries.
  - Adjusts transition probabilities and state initialization accordingly.
### Usage

- Required libraries: Pillow, numpy, scipy, imageio

### Code Structure
- `edge_strength()`: Computes the edge strength map of the image.
- `draw_boundary()`: Plots a boundary line on the image.
- `viterbi()`: Implements the Viterbi algorithm to compute boundaries.
- `air_human_viterbi()`: Human-traced version of the Viterbi algorithm for air-ice boundary.
- `icerock_viterbi()`: Computes ice-rock boundary using Viterbi.
- `icerock_human_viterbi()`: Human-traced version of the Viterbi algorithm for ice-rock boundary.
### Example
Given an input image (input_image.png), the program identifies and visualizes the air-ice and ice-rock layers, saving the results as:
<br>
`air_ice_output.png` <br>
`ice_rock_output.png`<br>
Additionally, the coordinates for the boundaries are stored in layers_output.txt.
