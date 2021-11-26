"""
Simulating the delays created by electric vehicles.
We do this by creating a weighted directed graph, where each node is a destination.
"""

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
max_grid_size = (24, 24)
circle_size = 10

background_color = (0, 0, 0)
screen_width = 800
screen_height = 600

class Scene:
	def __init__(self):
		self.riders: List[Location] = []
		self.cars: List[Location] = []

	def draw(self, screen):
		# draw all riders on the screen scaled to the max grid size
		for rider in self.riders:
			pygame.draw.circle(
				screen,
				rider_color,
				(rider.x * screen_width / max_grid_size[0],
					rider.y * screen_height / max_grid_size[1]),
				circle_size
			)
			
		# draw all cars on the screen scaled to the max grid size
		for car in self.cars:
			pygame.draw.circle(
				screen,
				car_color,
				(car.x * screen_width / max_grid_size[0],
					car.y * screen_height / max_grid_size[1]),
				circle_size
			)

	def calculate_manhattan_distance(self, car_id: int, rider_id: int) -> int:
		"""
		Returns the manhattan distance between the car and the rider.
		This is the weighting on the graph.
		"""
		return abs(self.cars[car_id].x - self.riders[rider_id].x) + abs(self.cars[car_id].y - self.riders[rider_id].y)


screen = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption("ML NP-complete AV Simulator")
pygame.display.flip()

scene = Scene()
scene.cars.append(Location(10, 10))
scene.cars.append(Location(10, 11))
scene.riders.append(Location(12, 12))
scene.riders.append(Location(14, 12))

running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.constants.QUIT:
			running = False

	screen.fill(background_color)
	scene.draw(screen)

	time.sleep(0.01)

	pygame.display.flip()
