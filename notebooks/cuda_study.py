import sys
import math
sys.path.append("..")
from numba import cuda
from planner.waypoint import Waypoint
import numpy as np
import cv2

print(cuda.gpus[0].name)


@cuda.jit
def kernel(frame, data):
    (x, y ) = cuda.grid(2)
    
    bev_width = data[0]
    bev_height = data[1]
    real_bev_size_width = data[2]
    real_bev_size_height = data[3]
    car_width = data[4]
    car_length = data[5]
    x_goal = data[6]
    y_goal = data[7]

    if x >= bev_width or y >= bev_height:  # Check array boundaries
        return
    
    x_ratio = (real_bev_size_width - car_width)/bev_width
    y_ratio = (real_bev_size_height - car_length)/bev_height

    xg = x_ratio * x - x_goal
    yg = y_ratio * y - y_goal

    frame[y, x, 1] = math.sqrt(xg*xg + yg*yg)
    frame[y , x, 2] = 0

    if x < car_width or x > bev_width - car_width:
        return

    i = int(x - car_width)
    feasible = 1

    c = 0
    while feasible == 1 and i < bev_width and c < car_width:
         pclass = frame[y,i,0]         
         feasible = pclass == 1 or pclass == 24 or pclass == 25 or pclass == 26 or pclass == 27
         i += 1
         c += 1

    if c < car_width:
        feasible = 0

    frame[y,x,2] = feasible

DIST_FRONT = 16.3
DIST_BACK = 13.5
DIST_LEFT = 17.9
DIST_RIGHT = 17.9

def pre_process_bev(frame: np.array, goal: Waypoint, car_width: int, car_length: int) -> any:
   d_frame = cuda.to_device(np.ascontiguousarray(frame))
   threadsperblock = (16, 16)
   blockspergrid_x = (frame.shape[1] - 1) // threadsperblock[0] + 1
   blockspergrid_y = (frame.shape[0] - 1) // threadsperblock[1] + 1
   blockspergrid = (blockspergrid_x, blockspergrid_y)

   d_data = cuda.to_device(np.ascontiguousarray([
      frame.shape[0],
      frame.shape[1],
      DIST_LEFT + DIST_RIGHT,
      DIST_FRONT + DIST_BACK,
      car_width, 
      car_length,
      goal.x,
      goal.z
   ], dtype='float'))

   kernel[blockspergrid, threadsperblock](d_frame, d_data)
   return d_frame.copy_to_host()

frame = cv2.imread('bev.png')
proc_frame = pre_process_bev(frame, Waypoint(200, 0), 4, 0)

print (proc_frame[int(sys.argv[1]), int(sys.argv[2]),:])

