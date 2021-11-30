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

class Car:
	def __init__(self, loc):
		self.loc: Location = loc
		self.dest = None

class Scene:
	def __init__(self):
		self.riders: List[Location] = []
		self.cars: List[Car] = []

	def step(self):
		for car in self.cars:
			if car.dest is not None:
				# if the car is assigned a destination, move towards it
				dest = self.riders[car.dest]
				if car.loc.x < dest.x:
					car.loc.x += 1
				elif car.loc.x > dest.x:
					car.loc.x -= 1
				if car.loc.y < dest.y:
					car.loc.y += 1
				elif car.loc.y > dest.y:
					car.loc.y -= 1
			else:
				# if the car is not assigned a destination, randomly assign one
				# get a list of all destinations without a car
				taken_destinations = set(car.dest for car in self.cars)
				available_destinations = [i for i in range(len(self.riders)) if i not in taken_destinations]
				car.dest = random.choice(available_destinations)

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
				sc(car.loc.x, car.loc.y),
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

def get_random_scene(ncars, nriders):
	scene = Scene()
	for _ in range(ncars):
		loc = Location(random.randint(0, max_grid_size[0]), random.randint(0, max_grid_size[1]))
		scene.riders.append(loc)
	for _ in range(nriders):
		loc = Location(random.randint(0, max_grid_size[0]), random.randint(0, max_grid_size[1]))
		car = Car(loc)
		scene.cars.append(car)
	return scene

screen = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption("ML AV Simulator")
pygame.display.flip()

scene = get_random_scene(10, 10)

x = 0

running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.constants.QUIT:
			running = False

	x += 1

	screen.fill(background_color)
	scene.draw_cars_and_riders(screen)
	scene.step()
	time.sleep(0.0)

	pygame.display.flip()
