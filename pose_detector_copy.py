# Check Pytorch installation
import torch, torchvision

print('torch version:', torch.__version__, torch.cuda.is_available())
print('torchvision version:', torchvision.__version__)

# Check MMPose installation
import mmpose

print('mmpose version:', mmpose.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version


import cv2
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector

pose_config = r'E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\mmpose\configs\body\2d_kpt_sview_rgb_img\topdown_heatmap\coco\hrnet_w48_coco_256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_config = r"E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\mmpose\demo\mmdetection_cfg\faster_rcnn_r50_fpn_coco.py"
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint)
# initialize detector
det_model = init_detector(det_config, det_checkpoint)


import math
def angle_between_lines(line1, line2):
    """
    Calculates the angle between two lines in radians.
    Each line is represented as a tuple (x1, y1, x2, y2)
    where (x1, y1) and (x2, y2) are the coordinates of
    two points on the line.
    """
    dx1 = line1[2] - line1[0]
    dy1 = line1[3] - line1[1]
    dx2 = line2[2] - line2[0]
    dy2 = line2[3] - line2[1]

    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    angle = angle2 - angle1

    return abs(angle*180/math.pi)%181
from collections import defaultdict as dd
def get_straight_hand_coordinates_for_every_person(frame):
    
    img = frame
    mmdet_results = inference_detector(det_model, img) # inference detection
    person_results = process_mmdet_results(mmdet_results, cat_id=1) # extract person (COCO_ID=1) bounding boxes from the detection results
    pose_results, returned_outputs = inference_top_down_pose_model(pose_model, img, person_results, bbox_thr=0.95, format='xyxy', dataset=pose_model.cfg.data.test.type) # inference pose
    # show pose estimation results
    vis_result = vis_pose_result(pose_model, img, pose_results, dataset=pose_model.cfg.data.test.type, show=False)
    # reduce image size
    vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
    
    persons=dd(dict)
    parts = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
    num_of_persons = len(pose_results)
    left_wrist_info = dd(dict)
    right_wrists_info = dd(dict)

    if num_of_persons != 0:
        # print(pose_results)
        # print(len(pose_results))
        # print(len(pose_results[0]['keypoints']))
        for person in range(num_of_persons):
            body=dict()
            for part in range(len(parts)):
                body[parts[part]]=pose_results[person]['keypoints'].tolist()[part]
            persons[person]=body

        templ=[]
        tempr=[]
        # print(persons)
        for i in persons:
            l1shouldert2elbow = []
            l1shouldert2elbow.extend(persons[i]['left_shoulder'][:2])
            l1shouldert2elbow.extend(persons[i]['left_elbow'][:2])
            l2wrist2elbow = []
            l2wrist2elbow.extend(persons[i]['left_wrist'][:2])
            l2wrist2elbow.extend(persons[i]['left_elbow'][:2])
            # print(l1shouldert2elbow,l2wrist2elbow,sep=' ')

            r1shouldert2elbow = []
            r1shouldert2elbow.extend(persons[i]['right_shoulder'][:2])
            r1shouldert2elbow.extend(persons[i]['right_elbow'][:2])
            r2wrist2elbow = []
            r2wrist2elbow.extend(persons[i]['right_wrist'][:2])
            r2wrist2elbow.extend(persons[i]['right_elbow'][:2])
            # print(r1shouldert2elbow,r2wrist2elbow,sep=' ')
            # print(img.shape)
            top_portionl = [persons[i]['left_wrist'][0],0]
            top_portionr = [persons[i]['right_wrist'][0],0]
            # print(i)
            print(angle_between_lines(l1shouldert2elbow, l2wrist2elbow),angle_between_lines(r1shouldert2elbow, r2wrist2elbow),sep=' ==== ')
            if angle_between_lines(l1shouldert2elbow, l2wrist2elbow) > 150:
                templ.append(math.dist(top_portionl,persons[i]['left_wrist'][:2]))
                left_wrist_info[i] = persons[i]['left_wrist'][:2]
                right_wrists_info[i] = persons[i]['right_wrist'][:2]
                # print("distance between left wrist and top_border = %s"%(math.dist(top_portionl,persons[i]['left_wrist'][:2])))
            # print('left hand is straight for person %s'%(str(i)))
            if angle_between_lines(r1shouldert2elbow, r2wrist2elbow) > 150:
                tempr.append(math.dist(top_portionr,persons[i]['right_wrist'][:2]))
                left_wrist_info[i] = persons[i]['left_wrist'][:2]
                right_wrists_info[i] = persons[i]['right_wrist'][:2]
                # print("distance between right wrist and top_border = %s"%(math.dist(top_portionr,persons[i]['right_wrist'][:2])))

            # print('left hand is straight for person %s'%(str(i)))

            # try:
            #     # for person in persons:
            #     # left_wrist_info[i] = persons[i]['left_wrist'][:2]
            #     # right_wrists_info[i] = persons[i]['right_wrist'][:2]
            #     # return left_wrist_info, right_wrists_info
            # except:
            #     pass
        if len(left_wrist_info)==0 and len(right_wrists_info)==0:
                return False
        return left_wrist_info, right_wrists_info
#         try:
#             if len(templ)!=0 and len(tempr)!=0:
#                 if templ.index(min(templ)) < tempr.index(min(tempr)):
#                     # print("person %s is trying to shoot with left hand"%(templ.index(min(templ))))
#                     return persons[templ.index(min(templ))]['left_wrist'][:2]
#                 else:
#                     # print("person %s is trying to shoot with right hand"%(tempr.index(min(tempr))))
#                     return persons[tempr.index(min(tempr))]['right_wrist'][:2]
#             else:
#                 if len(templ)==0:
#                     return persons[tempr.index(min(tempr))]['right_wrist'][:2]
#                 else:
#                     return persons[templ.index(min(templ))]['left_wrist'][:2]
#             # if templ.index(min(templ)) < tempr.index(min(tempr)):
#             #     print("person %s is trying to shoot with left hand"%(templ.index(max(templ))))
#             # else:
#             #     print("person %s is trying to shoot with right hand"%(tempr.index(max(tempr))))
#         except:
#             return False
#         # print("person %s is trying to shoot %s"%(templ.index(max(templ))))
#             # print("person %s is trying to shoot %s"%(tempr.index(max(tempr))))


#             # print("Person %s's left hand angle is %s"%(str(i), str(angle_between_lines(l1shouldert2elbow, l1wrist2elbow))))

# # im = cv2.imread(r'E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\input\images\frame196.jpg')
# # im = cv2.imread(r'E:\FinalYearProject\Input\Images\frame1.jpg')
# # im = cv2.imread(r'E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\mmpose\tests\data\coco\000000196141.jpg')
# # im = cv2.imread(r"E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\mmpose\tests\data\coco\000000000785.jpg")
# # print(get_straight_hand_coordinates_for_every_person(im))