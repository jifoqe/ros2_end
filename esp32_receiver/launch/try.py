#!/usr/bin/env python3
#episode,reason,distance,elapsed_time,score,timestamp
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import os
import subprocess, threading
import xacro
import time
import pickle
import random
import csv
from ament_index_python.packages import get_package_share_directory
from datetime import datetime
import tf_transformations
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

grid = 0.25
x_q = math.floor(0 / grid) * grid
y_q = math.floor(3 / grid) * grid
print(x_q)
print(random.randrange(8))