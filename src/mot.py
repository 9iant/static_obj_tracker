#!/usr/bin/env python
from __future__ import print_function
# import roslib
import sys
import math
import datetime
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
from bis import bis
#------------------for visualization---------------------
import cv2 # for drawing

'''------------------------------------------------------------------------------------------------------------'''
class mot:
    def __init__(self):
        self.distance_threshold = 1.2 # [m]
        self.soldier_tracker = Sort(dist_threshold=self.distance_threshold)
        self.dog_tracker = Sort(dist_threshold=self.distance_threshold)
        self.civilian_tracker = Sort(dist_threshold=self.distance_threshold)

    def detection_cb(self, ls_data_item):
        only_soldier = filter(lambda item: item.yolo_label=='soldier',ls_data_item)
        only_dog = filter(lambda item: item.yolo_label=='dog',ls_data_item)
        only_civilian = filter(lambda item: item.yolo_label=='civilian',ls_data_item)

        if len(only_soldier)!=0:
            self.soldier_tracker.update(only_soldier)

        if len(only_dog)!=0:
            self.dog_tracker.update(only_dog)

        if len(only_civilian)!=0:
            self.civilian_tracker.update(only_civilian)

    def print_result(self):
        print("-"*30)
        print("the tracking result <soldier>")
        print("-"*30)
        for i in range(len(self.soldier_tracker.trackers)):
            print('{}th track'.format(i+1))
            print(self.soldier_tracker.trackers[i].position)
            print('number of assigned data :', len(self.soldier_tracker.trackers[i].dets))
            print('number of assigned data :', self.soldier_tracker.trackers[i].dets)
        print("-"*30)
        print("the tracking result <dog>")
        print("-"*30)
        for i in range(len(self.dog_tracker.trackers)):
            print('{}th track'.format(i+1))
            print(self.dog_tracker.trackers[i].position)
            print('number of assigned data :', len(self.dog_tracker.trackers[i].dets))
        print("-"*30)
        print("the tracking result <civilian>")
        print("-"*30)
        for i in range(len(self.civilian_tracker.trackers)):
            print('{}th track'.format(i+1))
            print(self.civilian_tracker.trackers[i].position)
            print('number of assigned data :', len(self.civilian_tracker.trackers[i].dets))
    # TODO: visualization
    def plot_result(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal', 'box')
        ax.set(xlim = [-1.0, 11], ylim = [-1, 4], xlabel = 'x [m]', ylabel = 'y [m]')
        for i in range(len(self.soldier_tracker.trackers)):
            ax.plot(self.soldier_tracker.trackers[i].position[0],self.soldier_tracker.trackers[i].position[1],'bx',label='sol{}'.format(i+1))
        for i in range(len(self.dog_tracker.trackers)):
            ax.plot(self.dog_tracker.trackers[i].position[0],self.dog_tracker.trackers[i].position[1],'rx',label='dog{}'.format(i+1))
        for i in range(len(self.civilian_tracker.trackers)):
            ax.plot(self.civilian_tracker.trackers[i].position[0],self.civilian_tracker.trackers[i].position[1],'rx',label='civilian{}'.format(i+1))
        ax.legend()
        plt.show()
'''------------------------------------------------------------------------------------------------------------'''  
class Sort(object):
  def __init__(self, dist_threshold=0.8, min_hits = 5):
    """
    Sets key parameters for SORT
    """
    # self.max_age = max_age
    self.window_size = 10
    self.min_hits = 8
    self.dist_threshold = dist_threshold
    self.trackers = [] ## important
    self.frame_count = 0

  def update(self, data):
    """
    Params:
      data - a list of detections in same sequence ID, in the format [object,object,...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 2)) for frames without detections).
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    # Change the data type. // List --> 2D array
    dets = np.zeros((len(data), 2))
    for i in range(len(data)):
        dets[i][0] = data[i].object_global_position_2d_x
        dets[i][1] = data[i].object_global_position_2d_y 

    self.frame_count += 1 # TODO: for online track deletion

    # Change the data type. // List --> 2D array
    trks = np.zeros((len(self.trackers), 2))
    for idx, tracker in enumerate(self.trackers):
    	trks[idx] = self.trackers[idx].position # static object, no predition
    
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(np.array(dets), np.array(trks), self.dist_threshold) ##
    '''
        IMPORTANT!
        <The shape of return variables>
        matched : Assignment matrix,  [<trks>, <dets>] <...> is column matrix. = 2d array
        unmatched_dets : unmatched dets, [dets1, dets2, ... ] = 1d array
        unmatched_trks : unmatched trks, [trks1, trks2, ... ] = 1d array
    '''
    # update matched detections to tracks
    for m in matched:
      self.trackers[m[0]].update(data[m[1]])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = Tracker(data[i])
        self.trackers.append(trk)
        if data[0].yolo_label=='soldier':
            print("SOLDIER :: new track generated at {}! current total {} tracks".format(data[0].seq_tw, Tracker.soldier_count))
        if data[0].yolo_label=='dog':
            print("DOG     :: new track generated at {}! current total {} tracks".format(data[0].seq_tw, Tracker.dog_count))
        if data[0].yolo_label=='civilian':
            print("CIVILIAN:: new track generated at {}! current total {} tracks".format(data[0].seq_tw, Tracker.civilian_count))

    i = len(self.trackers)

    
    # TODO: online track deletion
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
'''------------------------------------------------------------------------------------------------------------'''
class Tracker(object):
  soldier_count = 0
  dog_count = 0
  civilian_count = 0
  def __init__(self, data): #v dets[m[1], :] -> data[m[1]]
    
    if data.yolo_label == 'soldier':
        self.id = Tracker.soldier_count
        Tracker.soldier_count += 1

    if data.yolo_label == 'dog':
        self.id = Tracker.dog_count
        Tracker.dog_count += 1

    if data.yolo_label == 'civilian':
        self.id = Tracker.civilian_count
        Tracker.civilian_count += 1
    
    self.position = np.array([data.object_global_position_2d_x, data.object_global_position_2d_y]) # [x, y]
    self.hits = 1
    self.dets = list([data])
  def update(self, data): # just simple moving avg
    self.hits += 1
    new_position = np.array([data.object_global_position_2d_x, data.object_global_position_2d_y])
    self.position = (self.position*(self.hits-1) + new_position)/self.hits
    self.dets.append(data)
'''------------------------------------------------------------------------------------------------------------'''
def linear_assignment(cost_matrix):
  """
  Hungarian Algorithm , DO NOT CHANGE!
  """
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
   make distance matrix , DO NOT CHANGE!
   """
   dets = np.expand_dims(dets,0)
   trks = np.expand_dims(trks,1)
   distance = np.linalg.norm(dets - trks, axis = 2)
   return distance

def associate_detections_to_trackers(detections,trackers,distance_threshold = 1.0):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
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
'''------------------------------------------------------------------------------------------------------------'''
'''  Functions for final output  '''
# # TODO : save our result 
def save_result(data,whichlabel ,best_images_list, positions, bbox_list ,file_path):
    # read the image we choose
    # save image with bounding box
    if whichlabel == "soldier":
        color = (0,0,255)
    elif whichlabel == "civilian":
        color = (0,255,0)
    else:
        color = (255,0,0)
    # make output folder if it does not exist 
    if not os.path.exists(file_path+"/output"):
        os.makedirs(file_path+"/output")
    cnt = 0
    result_text = ""
    for i, seq_tw in enumerate(best_images_list):
        information = data.loc[data["seq_tw"] == seq_tw]
        # To distinguish which label in single image
        for j in range(len(information)):
            if information["yolo_label"].values[j] == whichlabel:
                index = j
        image = cv2.imread(file_path + "/" + str(seq_tw) + ".jpg")
    
        
        label = information["yolo_label"].values[index]
        yolo_xmin = bbox_list[i][0]
        yolo_xmax = bbox_list[i][1]
        yolo_ymin = bbox_list[i][2]
        yolo_ymax = bbox_list[i][3]
        #  = information["yolo_xmin"].values[index]
        #  = information["yolo_xmax"].values[index]
        #  = information["yolo_ymin"].values[index]
        #  = information["yolo_ymax"].values[index]
        time_stamp = information["stamp"].values[index]
        datetimeobj = datetime.datetime.fromtimestamp(time_stamp)

        real_time = datetimeobj.strftime("%m/%d/%Y, %H:%M:%S")


        # print(label, yolo_xmin, yolo_xmax)
        cv2.rectangle(image,(yolo_xmin, yolo_ymin), (yolo_xmax, yolo_ymax), color, 2)
        cv2.putText(image, label, (yolo_xmin,yolo_ymin-10), 2, 1.2, color)
        cv2.putText(image, real_time, (int(image.shape[1]/2), image.shape[0]-30), 1 , 1.2, (255,255,255))
        
        # print("images are saved for "+whichlabel+ " label ")
        # cv2.imshow("test", image)
        # cv2.waitKey(10)
        # print(positions[i][0], positions[i][0])
        cv2.imwrite(file_path + "/output/" + label + str(cnt) + ".jpg", image)
        with open(file_path + "/output/" + label + str(cnt) + ".txt" , "w") as file:
            file.write(label+str(cnt)+ " is at (x, y) = (" +str(positions[i][0]) + ", "+str(positions[i][1])+")\n")
            result_text+=label+str(cnt)+ " is at (x, y) = (" +str(positions[i][0]) + ", "+str(positions[i][1])+")\n"
        
        cnt += 1
    with open(file_path + "/output/"+ whichlabel+" result.txt","w") as file:
        file.write(result_text)



    
def main(args):
    # !!
    input_path = "/home/asl/buffer_system_data/offboard_position_long_dist"
    input_csv_file = "/kdarpa_image_server_objects_info.csv"
    # output_path = "/home/asl/buffer_system_data/exploration_debug_3_640/output"
    # output_text_file_name = "output.txt"
    # read data
    data = pd.read_csv(input_path+input_csv_file)
    # cut data
    # !! remove this code when you do real test !
    # data = data.drop(data.index[296:len(data)]) # use [0:295] only for dataset1 (total 1185)

    # Change it to list type data container
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

    ## bis initalization ##
    best_image_selector = bis(0.5, 0.2, 0.3)

    ## CORE LOOP ##
    for ls_data_item in ls_data:
        #print(ls_data_item)
        if not math.isnan(ls_data_item[0]['object_global_position_2d_x']):
            mot_tracker.detection_cb(ls_data_item)


    ### Offline Track deletion ##
    for idx, tracker in enumerate(mot_tracker.dog_tracker.trackers):
        if tracker.hits < mot_tracker.dog_tracker.min_hits :
            mot_tracker.dog_tracker.trackers.remove(tracker)
            print("(dog) track dead because few detections")
    for idx, tracker in enumerate(mot_tracker.soldier_tracker.trackers):
        if tracker.hits < mot_tracker.soldier_tracker.min_hits :
            mot_tracker.soldier_tracker.trackers.remove(tracker)
            print("(soldier) track dead because few detections")
    for idx, tracker in enumerate(mot_tracker.civilian_tracker.trackers):
        if tracker.hits < mot_tracker.civilian_tracker.min_hits :
            mot_tracker.civilian_tracker.trackers.remove(tracker)
            print("(civilian) track dead because few detections")
    

    #mot_tracker.print_result()
    #mot_tracker.plot_result()
    # TODO: we need to unify the data, many unnecessary transitions from one type to the other
    #get best image for dog, civilian and soldier
    best_images_soldier, soldier_position, soldier_bbox_list = best_image_selector.run(mot_tracker.soldier_tracker.trackers,'soldier')
    best_images_dog, dog_position, dog_bbox_list = best_image_selector.run(mot_tracker.dog_tracker.trackers,'dog')
    best_images_civilian, civilian_position, civilian_bbox_list = best_image_selector.run(mot_tracker.civilian_tracker.trackers,'civilian')
    
    save_result(data, "soldier", best_images_soldier,soldier_position, soldier_bbox_list, input_path)
    save_result(data, "dog" , best_images_dog,dog_position, dog_bbox_list, input_path)
    save_result(data, "civilian", best_images_civilian,civilian_position, civilian_bbox_list, input_path)
 
    
        
    # create data in BIS format
    # print(best_images_dog)
    # print(ls_data[0][0].seq_tw)

    ''' BIS visualization'''
    # for idx, tracker in enumerate(mot_tracker.dog_tracker.trackers):
    #     best_image = filter(lambda x: x.seq_tw == best_images_dog[idx],tracker.dets)
    #     print(best_image[0])
    #     virtual_img = np.zeros((800,600),dtype = 'uint8')
    #     virtual_img[best_image[0].yolo_xmin:best_image[0].yolo_xmax, best_image[0].yolo_ymin:best_image[0].yolo_ymax] = 255
    #     # virtual_img[best_image[0].yolo_xmin][best_image[0].yolo_ymax] = 1
    #     # virtual_img[best_image[0].yolo_xmax][best_image[0].yolo_ymin] = 1
    #     # virtual_img[best_image[0].yolo_xmax][best_image[0].yolo_ymax] = 1
    #     # fig = plt.figure()
    #     # ax = filg
    #     plt.imshow(virtual_img.transpose())
    #     plt.show()
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
