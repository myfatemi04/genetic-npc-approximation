"""
Simulating the delays created by electric vehicles.
We do this by creating a weighted directed graph, where each node is a destination.
"""

import random
import time
from dataclasses import dataclass
from typing import List

import pygame
import pygame.display
import pygame.draw


@dataclass
class Location:
	x: int
	y: int

rider_color = (255, 0, 0)
car_color = (0, 0, 255)

max_grid_size = (20, 20)
grid_square_size = 600/24
node_size = 10
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
			path_vert = not path_vert

	def evaluate_permutation(self, permutation: List[int]) -> int:
		"""
		Returns the total distance of the permutation.
		The permutation is a list of integers, where each integer is the index of a car.
		"""

		total_distance = 0
		for rider_id in range(len(permutation)):
			car_id = permutation[rider_id]
			total_distance += self.calculate_manhattan_distance(car_id, rider_id)

		return total_distance

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

scene = get_random_scene(10, 10)

def mutate_permutation(permutation: List[int], swaps: int) -> List[int]:
	"""
	Returns a mutated permutation.
	"""
	new_permutation = permutation.copy()
	for _ in range(swaps):
		# swap two random indices
		i = random.randint(0, len(permutation) - 1)
		j = random.randint(0, len(permutation) - 1)
		new_permutation[i], new_permutation[j] = new_permutation[j], new_permutation[i]
	return new_permutation

best_permutation = list(range(len(scene.riders)))

population_size = 100

running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.constants.QUIT:
			running = False

	new_population = [*[
		mutate_permutation(best_permutation, 10)
		for _ in range(population_size)
	], best_permutation]

	best_permutation = min(new_population, key=lambda p: scene.evaluate_permutation(p))

	evaluation = scene.evaluate_permutation(best_permutation)
	print(evaluation)

	screen.fill(background_color)
	scene.draw_permutation(best_permutation, screen)
	scene.draw_cars_and_riders(screen)

	time.sleep(0.5)

	pygame.display.flip()
