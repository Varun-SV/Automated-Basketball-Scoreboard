import threading
import os
import cv2
from natsort import *
from pose_detector import *
import math

def get_the_minimum_distance_of_ball_from_person(p1,p2)->int:
    '''
    returns the minimum distance of the ball from the person
    ''' 
    return math.dist(p1,p2)

def get_the_thrower(frame,ballx,bally):
    '''
    for each frame check if the player is throwing the ball or not
    '''
    ball_coords = [ballx,bally]
    distsl = []
    distsr = []
    print("Ball coords are: ",ball_coords)
    try:
        left_wrist_info, right_wrists_info = get_straight_hand_coordinates_for_every_person(frame)
        for person in left_wrist_info:
            print("Left wrist info of the person %s is: %s"%(person,str(left_wrist_info[person])))
            distsl.append(get_the_minimum_distance_of_ball_from_person(ball_coords,left_wrist_info[person]))
        # if min(distsl)>10:
        #     return False
        for person in right_wrists_info:
            print("Right wrist info of the person %s is: %s"%(person,str(right_wrists_info[person])))
            distsr.append(get_the_minimum_distance_of_ball_from_person(ball_coords,right_wrists_info[person]))
        if (min(distsl)<min(distsr) and min(distsl)>5) or (min(distsr)<min(distsl) and min(distsr)>5):
            return False
        return True
    except:
        False

    
    

def store_frame_in_buffer(frame,frame_num):
    if not os.path.exists("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\"):
        os.system("mkdir E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\")
    cv2.imwrite("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\frame_"+str(frame_num)+".jpg",frame)

def store_the_shot(frame,frame_num,shot_num,ballx,bally,retrace=0): #this one will only store one frame.
    if not os.path.isdir("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\shot_%s"%(str(shot_num))):
        print("Creating a new folder for shot %s"%(str(shot_num)))
        os.system("mkdir E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\shot_%s"%(str(shot_num)))
    # frame_num = frame_name.split('_')[1]
    if retrace and frame_num>0:
        retrace_back_frames(frame,frame_num,shot_num,ballx,bally)
    cv2.imwrite("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\shot_%s\\frame_%s.jpg"%(str(shot_num),str(frame_num)),frame)

def reverse_analysis(frame,frame_num,shot_num,ballx,bally):
    if not os.path.isfile("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\frame_"+str(frame_num)+".jpg"):
        store_frame_in_buffer(frame,frame_num)
    cnt = 0
    framess = list(reversed(natsorted(os.listdir("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\"))))
    pth = "E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\"
    flg = 0
    print("Frames in buffer : ",len(framess))
    while not get_the_thrower(frame,ballx,bally) and cnt<=len(framess):
        cnt+=1
        print("Checking shot %s"%(str(shot_num)))
        for i in framess:
            print("Checking frame %s"%(len(framess)-framess.index(i)))
            cnt+=1
            fr = cv2.imread(pth+i)
            store_the_shot(fr,framess.index(i),shot_num,ballx,bally,0)
            if get_the_thrower(fr,ballx,bally):
                flg = 1
                break
            if cnt>=len(framess):
                break
        if flg ==1:
            break
    # else:
    #     for i in framess:
    #         os.system("move %s %s"%(pth+i,"E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\shot_%s\\"%(str(shot_num))))
    if len(framess)-cnt>=0:
        if len(framess)-cnt>=25:
            print("Shot %s is a 2 pointer"%(str(shot_num)))
        else:
            print("Shot %s is a 3 pointer"%(str(shot_num)))

    for i in framess:
        os.remove(pth+i)
        


def retrace_back_frames(frame,frame_num,shot_num,ballx,bally):
    if not os.path.isfile("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\frame_"+str(frame_num)+".jpg"):
        store_frame_in_buffer(frame,frame_num)
    framess = list(reversed(natsorted(os.listdir("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\"))))
    # print(framess)
    if len(framess)==0:
        return "No frames to process"
    else:
        # res_frame_num=0
        for i in range(len(framess)):
            shot_score=0
            # frame = cv2.imread("output\\images\\buffer_frames\\"+framess[i])
            # if get_the_thrower(frame,ballx,bally):
            res_frame_num=int(framess[i].split('_')[1].split('.')[0])
            if "E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\frame_%s.jpg"%(str(res_frame_num)) in framess:
                # print(framess)
                ind = framess.index("frame_%s.jpg"%(str(res_frame_num)))
                while not get_the_thrower(frame,ballx,bally) and j>=0:
                    for j in range(ind,0,-1):
                        fr = cv2.imread("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\"+framess[j])
                        store_the_shot(fr,j,shot_num,ballx,bally,1)
                else:
                    if j>=1:
                        fin = cv2.imread("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\buffer_frames\\"+framess[j-1])
                        store_the_shot(fin,j-1,shot_num,ballx,bally,1)
            try:
                new_framess = list(os.listdir("E:\\AI_basketball_games_video_editor\\AI_basketball_games_video_editor\\output\\images\\shot_%s\\"%(str(shot_num))))
                if len(new_framess)-i < 20:
                    shot_score=2
                else:
                    shot_score=3
                print("Shot %s is a %s pointer"%(shot_num,shot_score))
            except:
                pass
            for j in range(i,len(framess)):
                fr = cv2.imread("output\\images\\buffer_frames\\"+framess[j])
                store_the_shot(fr,j,shot_num,ballx,bally,0)
        for k in range(0,len(framess)):
            os.remove("output\\images\\buffer_frames\\"+framess[k])


# def classify_objects(frame):
#     cfg = get_cfg()
#     cfg.merge_from_file("detectron2\\configs\\COCO-InstanceSegmentation\\mask_rcnn_R_50_FPN_3x.yaml")
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation\\mask_rcnn_R_50_FPN_3x.yaml")
#     predictor = DefaultPredictor(cfg)
#     outputs = predictor(frame)
#     v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#     out = v.draw_instance_predictions(outputs["instances"].to("gpu"))
#     # cv2.imshow("Frame", out.get_image()[:, :, ::-1])
#     return out.get_image()[:,:,::-1]




# def vid2frame(vid_src:str,dest_fld:str):
#     if not os.path.exists(dest_fld):
#         os.system("md %s"%(dest_fld))
#     cap = cv2.VideoCapture(vid_src)
#     cnt=0
#     while cap.isOpened():
#         cnt+=1
#         strs = dest_fld+"frame"
#         ret, frame = cap.read()
#         strs += str(cnt)
#         cv2.imwrite(strs,frame)

#         if cv2.waitKey(10) & 0xFF==ord('q'):
#             break
#     cap.release()
    
            
# def fld_frame_detector(fld:str):
#     if not os.path.exists(fld):
#         raise "No Folder to process"
#     else:
#         frames = os.listdir(fld)
#         frames = natsorted(frames)
#         for frame in frames:
#             file_path = fld+frame
#             fr = cv2.imread(file_path)
#             fr = clas   sify_objects(fr)
#             cv2.imwrite(frame,fr)

# def main():
#     thread1 = threading.Thread(target=vid2frame(),args=(r"C:\\Users\\varun\\Videos\\Finaln year projects\\NBA 2k21 input.mp4","output\\images\\frames\\",))
#     thread2 = threading.Thread(target=fld_frame_detector(),args=("output\\images\\frames\\",))
#     thread1.start()
#     thread2.start()
#     thread1.join()
#     thread2.join()

# main()