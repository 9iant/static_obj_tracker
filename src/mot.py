#!/usr/bin/env python
from __future__ import print_function
# import roslib
import sys

#----------------for offline process---------
import numpy 
import matplotlib.pyplot as plt
import pandas as pd
import time
#-----------------from SORT------------------
import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.patches as patches
import glob
import time
import argparse
#---------------------------------------------
class mot:
    def __init__(self):
        # self.depth_sub = rospy.Subscriber("/depth_estimator/depth_infoarray", depth_infoArray, self.detection_cb)
        # publisher or srv server 
        self.distance_threshold = 1.0 # [m]
        self.mot_tracker = Sort(dist_threshold=self.distance_threshold) ## 
        # self.mot_Array_pub = rospy.Publisher("/mot/tracks", MarkerArray, queue_size = 1)

    def detection_cb(self, data):
        # TODO: 
        if len(data.results)>0:
        # only dog now.
            only_dog_indice = []
        for idx in range(len(data.results)):
            if data.results[idx].label == "dog":
                only_dog_indice.append(idx)
        if len(only_dog_indice):
            print("{} dogs are detected! ".format(len(only_dog_indice)))
            dets = np.zeros((len(only_dog_indice), 2))
            for i in only_dog_indice:
                dets[i][0] = data.results[i].x
                dets[i][1] = data.results[i].y
            self.mot_tracker.update(dets)##
        self.publish_markerArray()


class Sort(object):
  def __init__(self, dist_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    # self.max_age = max_age
    # self.min_hits = min_hits
    self.dist_threshold = dist_threshold
    self.trackers = [] ## important
    self.frame_count = 0

  def update(self, dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,score],[x,y,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 2))
    for idx, tracker in enumerate(self.trackers):
    	trks[idx] = self.trackers[idx].position # no predition
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(np.array(dets), np.array(trks), self.dist_threshold) ###########3
    
    # update matched trackers with assigned detections
    for m in matched:
      print("matching : {}".format(m)) #################### m[0] / m[1] exchaged!
      print(dets[m[0], :])
      print(self.trackers[m[1]].position)
      self.trackers[m[1]].update(dets[m[0], :])
    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = Tracker(dets[i,:])
        self.trackers.append(trk)
        print("new Track generated! current total {} tracks".format(Tracker.count))
    i = len(self.trackers)

    # if self.frame_count%150==1: # per 100 frames, delete tracker with few detections 
    #   del_trk = []
    #   for idx, tracker in enumerate(self.trackers):
    #     if tracker.hits < 10:
    #       del_trk.append(idx)
    #   for i in del_trk:
    #   	self.trackers.pop(i) # remove tracker with few detections
    #   	print("track dead because few detections")

    # for trk in reversed(self.trackers):
    #     d = trk.get_state()[0]
    #     if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
    #       ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
    #     i -= 1
    #     # remove dead tracklet
    #     if(trk.time_since_update > self.max_age):
    #       self.trackers.pop(i)
    # if(len(ret)>0):
    #   return np.concatenate(ret)
    # return np.empty((0,5))

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) 

  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def get_distancemat(dets, trks):
   """
   make distance matrix 
   """
   dets = np.expand_dims(dets,0)
   trks = np.expand_dims(trks,1)
   distance = np.linalg.norm(dets - trks, axis = 2)
   return distance

class Tracker(object):
  count = 0
  def __init__(self, position):
    self.id = Tracker.count
    self.position = np.array(position) # [x, y]
    Tracker.count += 1
    self.hits = 1
  def update(self, position): # just simple moving avg
    self.hits += 1
    self.position = (self.position*(self.hits-1) + np.array(position))/self.hits
    
def associate_detections_to_trackers(detections,trackers,distance_threshold = 1.0):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        print("There is no trackers")
        return np.zeros((0,2),dtype=int), np.arange(len(detections)), np.zeros((0,2),dtype=int)
    distance_matrix = get_distancemat(detections, trackers)

    distance_matrix = np.where(distance_matrix>distance_threshold, 999, distance_matrix) # like gating
    if min(distance_matrix.shape) > 0:
        matched_indices = linear_assignment(distance_matrix)
    else:
        matched_indices = np.zeros((0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    #filter out matched with far distance
        matches = []
    
    for m in matched_indices:
        if(distance_matrix[m[0], m[1]] > distance_threshold): # if distance is too far, don't match
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def main(args):
    
    # read data
    data = pd.read_csv("./rec/image_server_for_test.csv")
    print(len(data)) # 1185 for dataset1 
    data = data.drop(data.index[296:len(data)]) # use [0:295] only for dataset1

    # stamp = data['stamp'].to_numpy()
    # stamp = stamp - stamp[0] # Start with 0.0 s
    # seq_tw = data['seq_tw'].to_numpy()
    # drone_position_x = data['drone_position_x'].to_numpy()
    # drone_position_y = data['drone_position_y'].to_numpy()
    # drone_position_z = data['drone_position_z'].to_numpy()
    # drone_position_qx = data['drone_position_qx'].to_numpy() 
    # drone_position_qy = data['drone_position_qy'].to_numpy()
    # drone_position_qz = data['drone_position_qz'].to_numpy()
    # drone_position_qw = data['drone_position_qw'].to_numpy()
    # yolo_label = data['yolo_label'].to_numpy()
    # yolo_xmin = data['yolo_xmin'].to_numpy()
    # yolo_xmax = data['yolo_xmax'].to_numpy()
    # yolo_ymin = data['yolo_ymin'].to_numpy()
    # yolo_ymax = data['yolo_ymax'].to_numpy()
    # yolo_confidence = data['yolo_confidence'].to_numpy()
    # object_global_position_2d_x = data['object_global_position_2d_x'].to_numpy()
    # object_global_position_2d_y = data['object_global_position_2d_y'].to_numpy()
    # object_global_position_2d_z = data['object_global_position_2d_z'].to_numpy()

    ## MOT initialization ##
    mot_tracker = mot()

    data_list = []
    for key, value in data.iterrows():
        data_list.append(value)
        pass
    '''                   Draw (batch)           '''
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)

    # ax.scatter(drone_position_x, drone_position_y ,label='drone_traj')
    # ax.set_aspect('equal', 'box')
    # ax.legend()
    # # ax.set(xlim=[0., 1.], ylim=[-0.5, 2.5], title='Example', xlabel='xAxis', ylabel='yAxis')
    # obj = fig.add_subplot(1,1,1)
    # obj.scatter(object_global_position_2d_x, object_global_position_2d_y, label = 'obj')
    # obj.legend()

    # plt.show()

    '''                   Draw (iter)            '''
    # plt.ioff()

    # fig_iter = plt.figure(2)
    # ax_iter = fig_iter.add_subplot(1,1,1)
    # ax_iter.set(xlim = [-1.0, 11], ylim = [-1, 4], xlabel = 'x [m]', ylabel = 'y [m]')
    # traj_iter, = ax_iter.plot(drone_position_x[0:1], drone_position_y[0:1] ,'x')
    # obj_iter, = ax_iter.plot(object_global_position_2d_x[0:1], object_global_position_2d_y[0:1] ,'o')

    # plt.ion()
    # for idx in range(len(stamp)):
    #     traj_iter.set_data(drone_position_x[0:idx], drone_position_y[0:idx])
    #     obj_iter.set_data(object_global_position_2d_x[0:idx], object_global_position_2d_y[0:idx])
    #     plt.pause(0.01)

    #############################################################################################################

if __name__ == '__main__':
    main(sys.argv)