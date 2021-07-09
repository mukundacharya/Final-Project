from background_task import background
import time
from .helper import *

detect_update_interval = 1500
last_epoch=0

@background(schedule=5)
def detection_phase():
    frame_wb=getWebcamFrame()
    takeDecisions(frame_wb)
    #frame_ip=getIPWebcamFrame()
    #takeDecisions(frame_ip)
    time.sleep(3)
