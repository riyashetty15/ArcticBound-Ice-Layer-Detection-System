#!/usr/local/bin/python3
#
# Authors: Riya Shetty (rishett), Vanita Lalwani (vlalwan), Rohith Reddy (rohi)
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
# Discussed the idea and the approach with the seniors and fellow classmates. Approached the code with from the viterbi in-class activity.

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import numpy as np
import imageio
from copy import deepcopy

# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)

def neigbouring(r, c):
    neighbouring_pixels = [(r, c - 1), (r - 1, c - 1), (r + 1, c - 1), (r - 2, c - 1), (r + 2, c - 1), (r - 3, c - 1), (r + 3, c - 1), (r - 4, c - 1), (r + 4, c - 1)]      
    surrounding_edges = []
    for n in neighbouring_pixels:
        if (0 <= n[0] < hmm_row and 0 <= n[1] < hmm_col):
            surrounding_edges.append(n)
    return surrounding_edges

def emission_probablity(x):
    res = (x - min(x)) / (max(x) - min(x))
    return res

def viterbi(hmm_col, hmm_row, edge_strength, emissions, backtracking, transition_probablities, states):
    for c in range(1, hmm_col):
        for r in range(hmm_row):
            maximum = -1
            trans = neigbouring(r, c)
            for i in range(len(trans)):
                if abs(trans[i][0] - r) == 4:
                    val = transition_probablities[4] * states[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 3:
                    val = transition_probablities[3] * states[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 2:
                    val = transition_probablities[2] * states[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 1:
                    val = transition_probablities[1] * states[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 0:
                    val = transition_probablities[0] * states[trans[i][0]][c - 1]
                else:
                    val = 0.000001
                if val > maximum:
                    maximum = val
                    backtracking[r][c] = trans[i][0]
            states[r][c] = emissions[r][c] * maximum

    air = zeros(hmm_col)
    rock = zeros(hmm_col)

    max_val = argmax(states[:, hmm_col - 1])
    for c in range(hmm_col - 1, -1, -1):
        air[c] = int(max_val)
        max_val = backtracking[int(max_val)][c]
    return deepcopy(air)

def air_human_viterbi(hmm_col, hmm_row, edge_strength, emissions, backtracking, transition_probablities, states, air_col, air_row):
    air = zeros(hmm_col)
    states = deepcopy(states)
    for row in range(hmm_row):
        states[row][air_col] = 0
    states[air_row][air_col] = 1
    for col in range(air_col + 1, hmm_col):
        for row in range(hmm_row):
            trans = neigbouring(row, col)
            maximum = -1
            for i in range(len(trans)):
                if abs(trans[i][0] - row) == 4:
                    val = transition_probablities[4] * states[trans[i][0]][col - 1]
                elif abs(trans[i][0] - row) == 3:
                    val = transition_probablities[3] * states[trans[i][0]][col - 1]
                elif abs(trans[i][0] - row) == 2:
                    val = transition_probablities[2] * states[trans[i][0]][col - 1]
                elif abs(trans[i][0] - row) == 1:
                    val = transition_probablities[1] * states[trans[i][0]][col - 1]
                elif abs(trans[i][0] - row) == 0:
                    val = transition_probablities[0] * states[trans[i][0]][col - 1]
                else:
                    val = 0.0000001
                if val > maximum:
                    maximum = val
                    backtracking[row][col] = trans[i][0]
            states[row][col] = emissions[row][col] * maximum
    max_val = argmax(states[:, hmm_col - 1])
    for col in range(hmm_col - 1, -1, -1):
        air[col] = int(max_val)
        max_val = backtracking[int(max_val)][col]
    return air

def icerock_viterbi(hmm_col, hmm_row, edge_strength, emissions, backtracking1, transition_probablities, states1, a):
    air = zeros(hmm_col)
    states1 = deepcopy(states1)
    t = int(a[0] + 10)
    states1[:t+1, 0] = zeros(t+1)
    for c in range(1, hmm_col):
        t = int(a[c] + 15)
        if(t < hmm_row):
            r = t
        states1[:r+1, c] = zeros(r+1)
        while(r < (hmm_row)):
            trans = neigbouring(r, c)
            maximum = -1
            for i in range(len(trans)):
                if abs(trans[i][0] - r) == 4:
                    val = transition_probablities[4] * states1[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 3:
                    val = transition_probablities[3] * states1[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 2:
                    val = transition_probablities[2] * states1[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 1:
                    val = transition_probablities[1] * states1[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 0:
                    val = transition_probablities[0] * states1[trans[i][0]][c - 1]
                else:
                    val = 0.000001
                if val > maximum:
                    maximum = val
                    backtracking1[r][c] = trans[i][0]
            states1[r][c] = emissions[r][c] * maximum
            r = r + 1 
    max_v = argmax(states1[t: , hmm_col - 1])
    max_v += t
    for c in range(hmm_col - 1, -1, -1):
        air[c] = int(max_v)
        max_v = backtracking1[int(max_v)][c]
    return air

def icerock_human_viterbi(hmm_col, hmm_row, edge_strength, emissions, backtracking1, transition_probablities, states1, ice_col, ice_row, a):
    air = zeros(hmm_col)
    states1 = deepcopy(states1)
    states[:, ice_col] = zeros(hmm_row)
    t = int(a[0] + 10)
    states1[:t+1, 0] = zeros(t+1)
    states1[ice_row, ice_col] = 1
    for c in range(1, hmm_col):
        t = int(a[c] + 10)
        if(t < hmm_row):
            r = t
        states1[:r+1, c] = zeros(r+1)
        while(r < (hmm_row)):
            trans = neigbouring(r, c)
            maximum = -1
            for i in range(len(trans)):
                if abs(trans[i][0] - r) == 4:
                    val = transition_probablities[4] * states1[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 3:
                    val = transition_probablities[3] * states1[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 2:
                    val = transition_probablities[2] * states1[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 1:
                    val = transition_probablities[1] * states1[trans[i][0]][c - 1]
                elif abs(trans[i][0] - r) == 0:
                    val = transition_probablities[0] * states1[trans[i][0]][c - 1]
                else:
                    val = 0.000001
                if val > maximum:
                    maximum = val
                    backtracking1[r][c] = trans[i][0]
            states1[r][c] = emissions[r][c] * maximum
            r = r + 1 
    max_v = argmax(states1[t: , - 1])
    max_v += t
    for c in range(hmm_col - 1, -1, -1):
        air[c] = int(max_v)
        max_v = backtracking1[int(max_v)][c]
    return air

# main program
# Yellow - Bayes, Red - Feedback, Blue - Viterbi
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    air_row = int(gt_airice[0])     # row coordinate of air_ice
    air_col = int(gt_airice[1])     # col coordinate of air_ice
    ice_row = int(gt_icerock[0])    # row coordinate of ice_rock
    ice_col = int(gt_icerock[1])    # row coordinate of ice_rock

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
    hmm_col = edge_strength.shape[1]
    hmm_row = edge_strength.shape[0]
    airice_simple = [] 
    icerock_simple = []


    # AIR_ICE

    # For simple bayes we just need to find the maximum of per row and column
    for c in range(1, hmm_col):
        edges_bayes = edge_strength[:, c].tolist()
        firstmax = max(edges_bayes)
        firstmax = edges_bayes.index(firstmax)
        edges_bayes[firstmax] = 0
        while(1):
            secondmax = max(edges_bayes)
            secondmax = edges_bayes.index(secondmax)
            if abs(firstmax - secondmax) >= 10:
                break
            edges_bayes[secondmax] = 0
        val_min = min(firstmax, secondmax)
        val_max = max(firstmax, secondmax)
        icerock_simple.append(val_max)
        airice_simple.append(val_min)
    
    # initial values of emission, transition and the states 
    emissions = [emission_probablity(edge_strength[:, i]) for i in range(hmm_col)]
    emissions = array(emissions).T
    backtracking = zeros((hmm_row, hmm_col))
    transition_probablities = [1, 0.5, 0.1, 0.05, 0.01]
    states = zeros((hmm_row, hmm_col))
    states[:, 0] = copy(emissions[:, 0])

    # Implementation for HMM
    airice_hmm = zeros(hmm_col)
    # calculating viterbi
    airice_hmm = viterbi(hmm_col, hmm_row, edge_strength, emissions, backtracking, transition_probablities, states)
    # Human Tracing: similar to the above.
    airice_feedback = zeros(hmm_col)
    airice_feedback = air_human_viterbi(hmm_col, hmm_row, edge_strength, emissions, backtracking, transition_probablities, states, air_col, air_row)

    # ICE_ROCK

    # Values for icerock
    emissions1 = [emission_probablity(edge_strength[:, i]) for i in range(hmm_col)]
    emissions1 = array(emissions1).T
    backtracking1 = zeros((hmm_row, hmm_col))
    states1 = zeros((hmm_row, hmm_col))
    states1[:, 0] = copy(emissions1[:, 0])
    icerock_hmm = zeros(hmm_col)
    icerock_feedback = zeros(hmm_col)

    # Implementation for HMM
    icerock_hmm = icerock_viterbi(hmm_col, hmm_row, edge_strength, emissions, backtracking1, transition_probablities, states1, airice_hmm)
    # Implementation for Human HMM
    icerock_feedback = icerock_human_viterbi(hmm_col, hmm_row, edge_strength, emissions, backtracking1, transition_probablities, states1, ice_col, ice_row, airice_feedback)


    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
