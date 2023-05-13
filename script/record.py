#! /usr/bin/env python

import sys
import os
import csv
import rospy
import rosbag
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

input_bag = '/home/developer/Downloads/out_FAST-LIO.bag'
bag_file = '/home/developer/Downloads/in_STD.bag'
outposes = '/home/developer/Downloads/outposes.txt'

def correct_topics():
    with rosbag.Bag(bag_file, 'w') as Y:
      for topic, msg, t in rosbag.Bag(input_bag):
          if topic == '/cloud_registered_body':
              new_msg = msg
              new_msg.header.frame_id = "camera_init"
              new_msg.header.seq = 0
              Y.write('/cloud_undistort', new_msg, t)
          # if topic in ['/Odometry', '/tf', '/tf_static', '/clock']:
              # Y.write(topic, msg, t)
          # Y.write(topic, msg, t)
    Y.close()


def log_poses():
  bag = rosbag.Bag(bag_file)

  with open(outposes, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    for topic, msg, t in bag.read_messages(topics=['/Odometry']):
      if msg.pose is not None:
        time = float("{0}.{1}".format(t.secs, t.nsecs))
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        data_writer.writerow([time, p.x, p.y, p.z, o.x, o.y, o.z, o.w])

  bag.close()

print "Correcting the topics..."
correct_topics()
print "Correction Done"

print "Logging the poses..."
log_poses()
print "Logging done"
print "Finished"