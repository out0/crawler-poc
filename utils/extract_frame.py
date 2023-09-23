import cv2
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("use {sys.argv[0]} videofile frame_file [frame skip]\n")
        exit(0)

    input = cv2.VideoCapture(sys.argv[1])
    skip = 1

    if (len(sys.argv) > 3 and sys.argv[3] != None):
        skip = int(sys.argv[3])
    
    
    while skip > 0:
        reading, frame = input.read()
        skip = skip - 1
    
    if not reading:
        print ("no frame available\n")
    else:
        cv2.imwrite(sys.argv[2], frame)