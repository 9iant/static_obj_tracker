import numpy as np
import csv
from collections import defaultdict

class bis:
    def __init__(self, W_distance_drone_track, W_distance_object_track, W_YOLO_confidence):
        self.W_distance_drone_track = W_distance_drone_track
        self.W_distance_object_track = W_distance_object_track
        self.W_YOLO_confidence = W_YOLO_confidence

    def Distance_calculator(self,Distance_1, Distance_2):
        #TODO check the distance calculator
        Distance = np.linalg.norm(Distance_1-Distance_2,axis = 0)
        return Distance

    def best_image_selector(self,data_per_track):
        for track in data_per_track:
            #compute distance between drone and object position
            drone_x_y = np.array([data_per_track[track]['drone_position_x'], data_per_track[track]['drone_position_y']])
            image_x_y = np.array([data_per_track[track]['object_global_position_2d_x'], data_per_track[track]['object_global_position_2d_y']])
            track_x_y = np.random.rand(2,1)
            distance_drone_track = self.Distance_calculator(drone_x_y,track_x_y)
            distance_object_track = self.Distance_calculator(image_x_y,track_x_y)
            #compute the fitness
            #TODO division by zero needs to be handled
            yolo_confidence = np.array(data_per_track[track]['drone_position_x'])
            print(yolo_confidence.shape)
            fitness = self.W_distance_drone_track*1/distance_drone_track + self.W_distance_object_track*1/distance_object_track + self.W_YOLO_confidence * yolo_confidence
            best_image = np.argmax(fitness)
            print('best image for track {0} is {1}'.format(track,data_per_track[track]['seq_tw'][best_image]))