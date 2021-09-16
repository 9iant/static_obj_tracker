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

    def best_image_selector(self,data_per_track, objects_positions):
        index = 0
        best_image_list = []
        yolo_bbox_list = []
        object_position_list = []
        print(len(data_per_track))
        for track in data_per_track:
            #compute distance between drone and object position
            drone_x_y = np.array([data_per_track[track]['drone_position_x'], data_per_track[track]['drone_position_y']])
            image_x_y = np.array([data_per_track[track]['object_global_position_2d_x'], data_per_track[track]['object_global_position_2d_y']])
            #track_x_y = np.random.rand(2,1)
            track_x_y = objects_positions[index][:,None]
            distance_drone_track = self.Distance_calculator(drone_x_y,track_x_y)
            distance_object_track = self.Distance_calculator(image_x_y,track_x_y)
            #compute the fitness
            #TODO division by zero needs to be handled
            yolo_confidence = np.array(data_per_track[track]['drone_position_x'])
            fitness = self.W_distance_drone_track*1/distance_drone_track + self.W_distance_object_track*1/distance_object_track + self.W_YOLO_confidence * yolo_confidence
            
            best_image = np.argmax(fitness)
            print('best image for track <{0}> is <{1}>'.format(track,data_per_track[track]['seq_tw'][best_image]))
            index += 1 
            best_image_list.append(data_per_track[track]['seq_tw'][best_image])
            yolo_bbox_list.append(\
                [data_per_track[track]['yolo_xmin'][best_image],\
                data_per_track[track]['yolo_xmax'][best_image],\
                data_per_track[track]['yolo_ymin'][best_image],\
                data_per_track[track]['yolo_ymax'][best_image]])

            object_position_list.append(track_x_y)
        
        return best_image_list, object_position_list, yolo_bbox_list

    def run(self, trackers, name):
        tracks_dic = {}
        list_object_pos = []
        
        for idx, tracker in enumerate(trackers):
            #create tracker
            tracks_dic[name+'_id_{0}'.format(tracker.id)] = {}
            list_object_pos.append(tracker.position)
            #fill in the keys
            for index,value in tracker.dets[0].items():
                tracks_dic[name+'_id_{0}'.format(tracker.id)][index] = []
            for i in range(0,len(tracker.dets)):
                for index,value in tracker.dets[i].items():
                    tracks_dic[name+'_id_{0}'.format(tracker.id)][index].append(value)
        best_image_list, object_position_list, yolo_bbox_list = self.best_image_selector(tracks_dic,list_object_pos)

        return best_image_list, object_position_list,yolo_bbox_list
