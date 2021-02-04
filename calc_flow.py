import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def frames():
    """
    Get frames from video
    """

    video = cv2.VideoCapture(VIDEO_PATH)
    success = True

    while(success):
        success, current = video.read()
        if not success:
            break
        yield current
    else:
        video.release()


def optical_flow_dense():
    """
    Detect and Draw Optical Flow by Dense
    """

    to_grayscale = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    
    previous_is_set = False
    p_frame = None
    hsv = []
    
    params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0
    )
    
    # edit
    i = 0 

    for frame in frames():
        glayed = to_grayscale(frame)

        if not previous_is_set:
            p_frame = glayed
            #f_frame = p_frame.copy()
            hsv = np.zeros_like(frame)
            hsv[...,1] = 255
            previous_is_set = True
        else:
            # calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(glayed, p_frame, None, **params)
            cv2.writeOpticalFlow('OpticalFlow/estimated_flow/flow/adjacent/shapes/swirl/%d.flo' % i , flow)
            
        
            if flow is None:
                continue
            else:
                # optical flow's magnitudes and angles
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude to 0-255 scale
                img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                cv2.imwrite('OpticalFlow/estimated_flow/flow_img/adjacent/shapes/swirl/%d.png' % i , img)
                
                yield img
                
                p_frame = glayed.copy()
        i += 1
        print(i)

def draw(flow_func):
    fig = plt.figure()
    ims = []

    for f in flow_func():
        ims.append([plt.imshow(f)])

    ani = animation.ArtistAnimation(fig, ims)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':

    VIDEO_PATH = "OpticalFlow/movie/shapes-processed/shapes/swirl.gif"

    # params for ShiTomasi corner detection
    FEATURE_COUNT = 100
    FEATURE_PARAMS = dict(
        maxCorners=FEATURE_COUNT,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # Parameters for lucas kanade optical flow
    LK_PARAMS = dict(
        winSize  = (15,15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03)
    )

    # color for drawing (create FEATURE_COUNT colors each of these is RGB color)
    COLOR = np.random.randint(0, 255,(FEATURE_COUNT, 3))

    draw(optical_flow_dense)
