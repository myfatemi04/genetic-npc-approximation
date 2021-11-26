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
path_color = (0, 255, 0)
max_grid_size = (24, 24)
grid_square_size = 800/24
circle_size = 10

background_color = (0, 0, 0)
screen_width = 800
screen_height = 600

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
				(rider.x * grid_square_size, rider.y * grid_square_size),
				circle_size
			)
			
		# draw all cars on the screen scaled to the max grid size
		for car in self.cars:
			pygame.draw.circle(
				screen,
				car_color,
				(car.x * grid_square_size, car.y * grid_square_size),
				circle_size
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

		for rider_id in range(len(permutation)):
			car_id = permutation[rider_id]

			car_x = self.cars[car_id].x
			car_y = self.cars[car_id].y
			rider_x = self.riders[rider_id].x
			rider_y = self.riders[rider_id].y
			# vertical line from car y to rider y
			pygame.draw.line(
				screen,
				path_color,
				(car_x * grid_square_size, car_y * grid_square_size),
				(car_x * grid_square_size, rider_y * grid_square_size)
			)
			# horizontal line from car x to rider x
			pygame.draw.line(
				screen,
				path_color,
				(car_x * grid_square_size, rider_y * grid_square_size),
				(rider_x * grid_square_size, rider_y * grid_square_size)
			)
		

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

scene = Scene()
scene.cars.append(Location(10, 10))
scene.cars.append(Location(10, 11))
scene.riders.append(Location(12, 12))
scene.riders.append(Location(14, 12))

evaluation = scene.evaluate_permutation([0, 1])
print(evaluation)

running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.constants.QUIT:
			running = False

	screen.fill(background_color)
	scene.draw_permutation([0, 1], screen)
	scene.draw_cars_and_riders(screen)

	time.sleep(0.01)

	pygame.display.flip()
