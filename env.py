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

__s = 1000
max_grid_size = (__s, __s)
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
		self.reached_dest_in = None

class Scene:
	def __init__(self):
		self.riders: List[Location] = []
		self.cars: List[Car] = []

	def step(self, timer):
		for car in self.cars:
			if car.dest is not None:
				if car.reached_dest_in is not None:
					continue
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
				if car.loc.x == dest.x and car.loc.y == dest.y:
					car.reached_dest_in = timer
			else:
				method = 'random'

				if method == 'random':
					# if the car is not assigned a destination, randomly assign one
					# get a list of all destinations without a car
					taken_destinations = set(car.dest for car in self.cars)
					available_destinations = [i for i in range(len(self.riders)) if i not in taken_destinations]
					car.dest = random.choice(available_destinations)
				
				elif method == 'greedy':
					# choose the closest destination that is not assigned to a car
					# get a list of all destinations without a car
					taken_destinations = set(car.dest for car in self.cars)
					available_destinations = [i for i in range(len(self.riders)) if i not in taken_destinations]
					# choose the closest destination that is not assigned to a car
					closest_destination = None
					closest_distance = None
					for dest_id in available_destinations:
						dest = self.riders[dest_id]
						distance = abs(car.loc.x - dest.x) + abs(car.loc.y - dest.y)
						if closest_destination is None or distance < closest_distance:
							closest_destination = dest_id
							closest_distance = distance
					car.dest = closest_destination

		reached_dest_in = [car.reached_dest_in for car in self.cars]
		if None in reached_dest_in:
			return
		
		# all cars reached their destination
		if max(reached_dest_in)	== timer:
			# all cars reached their destination
			print("all cars reached their destination")
			print("average delay:", sum(reached_dest_in)/len(reached_dest_in))
			print("max delay:", max(reached_dest_in))
			print()

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

timer = 0

running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.constants.QUIT:
			running = False

	timer += 1

	screen.fill(background_color)
	scene.draw_cars_and_riders(screen)
	scene.step(timer)
	time.sleep(0.0)

	pygame.display.flip()
