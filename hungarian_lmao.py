"""
This is just Q-learning.

TODO: Make it stop at a good place
		Maybe use best genetic mutation of current permutation as stop signal?

TODO: Add distances/coordinates into input state
TODO: Make things identify by coordinates, not ID
		RN, the only thing in the input state is ID
		It takes a lot of swaps before the AI actually realizes where everything is
		save on swaps and also make it more robust by adding coords

TODO: add a NN instead of a Q-Table to make a DQN in PyTorch.

TODO: add Multi Agent Reinforcement learning, where each car has its own agent

TODO: add GNN instead of normal NN or Q-Table

TODO: sell to Uber
"""

import random
import time
from dataclasses import dataclass
from typing import List

import pygame
import pygame.display
import pygame.draw

from collections import defaultdict
import itertools
import matplotlib
import matplotlib.style
import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
import math

@dataclass
class Location:
	x: int
	y: int

rider_color = (255, 0, 0)
car_color = (0, 0, 255)

max_grid_size = (500, 500)
grid_square_size = 600/(max_grid_size[1] + 4)
node_size = 5
corner_size = 3

background_color = (0, 0, 0)
screen_width = 800
screen_height = 600

def sc(x, y):
	# converts to screen coords
	if type(x) == tuple:
		x, y = x
	return (x * grid_square_size + 20, y * grid_square_size + 20)

class Scene:
	def __init__(self):
		self.riders: List[Location] = []
		self.cars: List[Location] = []

	def draw_cars_and_riders(self, screen):
		# draw all riders on the screen scaled to the max grid size
		for rider in self.riders:
			pygame.draw.circle(
				screen,
				rider_color,
				sc(rider.x, rider.y),
				node_size
			)

		# draw all cars on the screen scaled to the max grid size
		for car in self.cars:
			pygame.draw.circle(
				screen,
				car_color,
				sc(car.x, car.y),
				node_size
			)

	def calculate_manhattan_distance(self, car_id: int, rider_id: int) -> int:
		"""
		Returns the manhattan distance between the car and the rider.
		This is the weighting on the graph.
		"""
		return abs(self.cars[car_id].x - self.riders[rider_id].x) + abs(self.cars[car_id].y - self.riders[rider_id].y)

	def calculate_euclidean_distance(self, car_id: int, rider_id: int) -> float:
		"""
		Returns the euclidean distance between the car and the rider.
		This is the weighting on the graph.
		"""
		return ((self.cars[car_id].x - self.riders[rider_id].x) ** 2 + (self.cars[car_id].y - self.riders[rider_id].y) ** 2) ** 0.5

	def draw_permutation(self, permutation: List[int], screen):
		"""
		Draws the pathways from each car to their respective riders.
		"""

		path_vert = True

		for rider_id in range(len(permutation)):
			car_id = permutation[rider_id]

			car_x = self.cars[car_id].x
			car_y = self.cars[car_id].y
			rider_x = self.riders[rider_id].x
			rider_y = self.riders[rider_id].y
			vert_color= (0, 255, 255)
			horz_color = (255, 0, 255)

			path_vert = not path_vert

			pygame.draw.line(
				screen,
				vert_color if path_vert else horz_color,
				sc(car_x, car_y),
				sc(rider_x, rider_y),
			)

			continue

			if path_vert:
				# vertical line from car y to rider y
				pygame.draw.line(
					screen,
					vert_color,
					sc(car_x, car_y),
					sc(car_x, rider_y),
				)
				# horizontal line from car x to rider x
				pygame.draw.line(
					screen,
					horz_color,
					sc(car_x, rider_y),
					sc(rider_x, rider_y),
				)
				# corner
				pygame.draw.circle(
					screen,
					vert_color,
					sc(car_x, rider_y),
					corner_size
				)
			else:
				# horizontal line from car x to rider x
				pygame.draw.line(
					screen,
					horz_color,
					sc(car_x, car_y),
					sc(rider_x, car_y),
				)
				# vertical line from car y to rider y
				pygame.draw.line(
					screen,
					vert_color,
					sc(rider_x, car_y),
					sc(rider_x, rider_y),
				)
				# corner
				pygame.draw.circle(
					screen,
					horz_color,
					sc(rider_x, car_y),
					corner_size
				)
			# path_vert = not path_vert

	def evaluate_permutation(self, permutation: List[int]) -> int:
		"""
		Returns the total distance of the permutation.
		The permutation is a list of integers, where each integer is the index of a car.
		"""

		total_distance = 0
		for rider_id in range(len(permutation)):
			car_id = permutation[rider_id]
			total_distance += 0 * self.calculate_manhattan_distance(car_id, rider_id) + self.calculate_euclidean_distance(car_id, rider_id)

		return total_distance

"""
class CarRoutingEnv:
	def __init__(self):
		self.cars = 10
		self.riders = 10
		self.scene = get_random_scene(cars, riders)


	def reset()
"""

screen = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption("ML NP-complete AV Simulator")
pygame.display.flip()

def get_random_scene(ncars, nriders):
	scene = Scene()
	for _ in range(ncars):
		scene.cars.append(Location(random.randint(0, max_grid_size[0]), random.randint(0, max_grid_size[1])))
	for _ in range(nriders):
		scene.riders.append(Location(random.randint(0, max_grid_size[0]), random.randint(0, max_grid_size[1])))
	return scene

nnodes = 10

scene = get_random_scene(nnodes, nnodes)


def swap_two(permutation: List[int], a:int, b:int, scene: Scene) -> List[int]:
	"""
	Returns a swapped permutation.
	"""
	new_permutation = permutation.copy()

	new_permutation[a], new_permutation[b] = new_permutation[b], new_permutation[a]
	return new_permutation




def createEpsilonGreedyPolicy(Q, epsilon, num_actions):

    def policyFunction(state):

        Action_probabilities = np.ones(num_actions,
                dtype = float) * epsilon / num_actions

        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction


best_permutation = list(range(len(scene.riders)))


adjmat = [ [-999999999999999 for i in range(len(best_permutation)*2 )] for j in range(len(best_permutation)*2 ) ]

for i in range(len(best_permutation)):
    adjmat[i][best_permutation[i]+len(best_permutation)] = scene.calculate_euclidean_distance(i, best_permutation[i])
    adjmat[best_permutation[i]+len(best_permutation)][i] = scene.calculate_euclidean_distance(i, best_permutation[i])


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


car_pts = [ [scene.cars[i].x, scene.cars[i].y] for i in range(len(scene.cars))]
rider_pts = [ [scene.riders[i].x, scene.riders[i].y] for i in range(len(scene.riders))]

C = cdist(rider_pts, car_pts)
_, assignment = linear_sum_assignment(C)

print(assignment)

print(best_permutation)

print(scene.evaluate_permutation(assignment.tolist()))


def mutate_permutation(permutation: List[int], swaps: int, scene: Scene) -> List[int]:
	"""
	Returns a mutated permutation.
	"""
	new_permutation = permutation.copy()
	for _ in range(swaps):
		# swap two random indices
		possible_riders = [i for i in range(len(permutation))]
		weights = [
			# distance from car to rider
			scene.calculate_euclidean_distance(new_permutation[rider_id], rider_id) for rider_id in range(len(permutation))
		]
		weights_total = sum(weights)
		weights = [w / weights_total for w in weights]
		rider_id_a, rider_id_b = random.choices(possible_riders, weights=weights, k=2)
		new_permutation[rider_id_a], new_permutation[rider_id_b] = new_permutation[rider_id_b], new_permutation[rider_id_a]
	return new_permutation

best_permutation = list(range(len(scene.riders)))

population_size = 10
nrand = 0

x = 0

def random_permutation(n):
	permutation = list(range(n))
	random.shuffle(permutation)
	return permutation

while True:
	screen.fill(background_color)
	scene.draw_permutation(assignment, screen)
	scene.draw_cars_and_riders(screen)

	pygame.display.flip()

	time.sleep(5)

	break

running = True
k = 0
while running:
	for event in pygame.event.get():
		if event.type == pygame.constants.QUIT:
			running = False

	x += 1

	prev  = scene.evaluate_permutation(best_permutation)

	new_population = [*[
		mutate_permutation(best_permutation, random.randint(1, 4), scene)
		for _ in range(population_size)
	], best_permutation, *[random_permutation(len(best_permutation)) for _ in range(nrand)]]

	prev_best_permutation = best_permutation

	best_permutation = min(new_population, key=lambda p: scene.evaluate_permutation(p))
	if scene.evaluate_permutation(best_permutation) == scene.evaluate_permutation(prev_best_permutation):
		best_permutation = prev_best_permutation

	evaluation = scene.evaluate_permutation(best_permutation)
	if x % 100 == 1:

		print(evaluation)


	screen.fill(background_color)
	scene.draw_permutation(best_permutation, screen)
	scene.draw_cars_and_riders(screen)

	pygame.display.flip()

print(x)
