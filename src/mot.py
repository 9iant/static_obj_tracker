#!/usr/bin/env python
from __future__ import print_function
# import roslib
import sys

#----------------for offline process---------
import numpy as np
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
        self.distance_threshold = 1.5 # [m]
        self.soldier_tracker = Sort(dist_threshold=self.distance_threshold) ## 
        self.dog_tracker = Sort(dist_threshold=self.distance_threshold)
        # self.mot_Array_pub = rospy.Publisher("/mot/tracks", MarkerArray, queue_size = 1)

    def detection_cb(self, ls_data_item):
        # if len(ls_data_item)>0:
        # # only soldier now.
        #     only_soldier_indice = []
        # for idx in range(len(ls_data_item)):
        #     if ls_data_item[idx].yolo_label == "soldier":
        #         only_soldier_indice.append(idx)
        # if len(only_soldier_indice):
        #     # print("{} soldier is detected! ".format(len(only_soldier_indice)))
        #     dets = np.zeros((len(only_soldier_indice), 2))
        #     for i in only_soldier_indice:
        #         dets[i][0] = ls_data_item[i].object_global_position_2d_x
        #         dets[i][1] = ls_data_item[i].object_global_position_2d_y
        #     # print(dets)
        
        only_soldier = filter(lambda item: item.yolo_label=='soldier',ls_data_item)
        only_dog = filter(lambda item: item.yolo_label=='dog',ls_data_item)
        print("current seq id : ", ls_data_item[0].seq_tw)
        if len(only_soldier)!=0:
            print("update soldier")
            self.soldier_tracker.update(only_soldier)

        if len(only_dog)!=0:
            print("update dog")
            self.dog_tracker.update(only_dog)

    def print_result(self):
        print("-"*10)
        print("the tracking result <soldier>")
        print("-"*10)
        for i in range(len(self.soldier_tracker.trackers)):
            print('{}th track'.format(i))
            print(self.soldier_tracker.trackers[i].position)
            print('number of assigned data :', len(self.soldier_tracker.trackers[i].dets))
        print("-"*10)
        print("the tracking result <dog>")
        print("-"*10)
        for i in range(len(self.dog_tracker.trackers)):
            print('{}th track'.format(i+1))
            print(self.dog_tracker.trackers[i].position)
            print('number of assigned data :', len(self.dog_tracker.trackers[i].dets))
        
class Sort(object):
  def __init__(self, dist_threshold=0.3, min_hits = 10):
    """
    Sets key parameters for SORT
    """
    # self.max_age = max_age
    self.window_size = 10
    self.min_hits = 5
    self.dist_threshold = dist_threshold
    self.trackers = [] ## important
    self.frame_count = 0

  def update(self, data):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,score],[x,y,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets = np.zeros((len(data), 2))
    for i in range(len(data)):
        dets[i][0] = data[i].object_global_position_2d_x
        dets[i][1] = data[i].object_global_position_2d_y

    self.frame_count += 1 # all data has same seq_tw number. thus choose 0 indice

    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 2))
    for idx, tracker in enumerate(self.trackers):
    	trks[idx] = self.trackers[idx].position # no predition
    # print('dets', dets)
    # print('trks', trks)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(np.array(dets), np.array(trks), self.dist_threshold) ##
    '''
        matched : Assignment matrix,  [<trks>, <dets>] <...> is column matrix. = 2d array
        unmatched_dets : unmatched dets, [dets1, dets2, ... ] = 1d array
        unmatched_trks : unmatched trks, [trks1, trks2, ... ] = 1d array
    '''
    # print('-'*10)
    # print('matching result : ')
    # print(matched)
    # print('unmatched dets : ')
    # print(unmatched_dets)
    # print('unmatched tracks : ')
    # print(unmatched_trks)
    # print('-'*10)

    # update matched trackers with assigned detections
    # print("<"*10+"start update"+">"*10)
    for m in matched:
    #   print("matching : {}".format(m)) #################### m[0] / m[1] exchaged!
    #   print(dets)

    #   print(dets[m[1], :])
    #   print(self.trackers[m[0]].position)
      self.trackers[m[0]].update(data[m[1]])#(dets[m[1], :])
    # create and initialise new trackers for unmatched detections
    
    # print(data)
    for i in unmatched_dets:
        trk = Tracker(data[i])
        self.trackers.append(trk)
        print("new Track generated! current total {} tracks".format(Tracker.count))
    i = len(self.trackers)

    

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
  def __init__(self, data): #v dets[m[1], :] -> data[m[1]]
    self.id = Tracker.count
    self.position = np.array([data.object_global_position_2d_x, data.object_global_position_2d_y]) # [x, y]
    Tracker.count += 1
    self.hits = 1
    self.dets = list(data)
  def update(self, data): # just simple moving avg
    self.hits += 1
    new_position = np.array([data.object_global_position_2d_x, data.object_global_position_2d_y])
    self.position = (self.position*(self.hits-1) + new_position)/self.hits
    self.dets.append(data)

    
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
        if(d not in matched_indices[:,1]):
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,0]):
            unmatched_trackers.append(t)
    #filter out matched with far distance
    
    matches = []
    
    for m in matched_indices:
        if(distance_matrix[m[0], m[1]] > distance_threshold): # if distance is too far, don't match
            unmatched_detections.append(m[1])
            unmatched_trackers.append(m[0])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def main(args):
    # read data
    data = pd.read_csv("./rec/image_server_out1.csv")
    # cut data
    data = data.drop(data.index[296:len(data)]) # use [0:295] only for dataset1 (total 1185)

    # change it to list type data container
    ls_data = []
    result = []
    index = 0
    for key, value in data.iterrows():
        if (index+2)> len(data): # to deal with last item
            break
        result.append(value)
        if value.seq_tw != data.loc[index+1,'seq_tw']: # if the seq_tw is same, store them into one item.
            ls_data.append(result)
            result = []    
        index += 1
    
    ## MOT initialization ##
    mot_tracker = mot()
    ## pseudo topic generation and run callback
    for ls_data_item in ls_data:
        mot_tracker.detection_cb(ls_data_item)


    # TODO: delete tracks..
    
    for idx, tracker in enumerate(mot_tracker.dog_tracker.trackers):
        if tracker.hits < mot_tracker.dog_tracker.min_hits :
            mot_tracker.dog_tracker.trackers.remove(tracker)
            print("track dead because few detections")

    mot_tracker.print_result()
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