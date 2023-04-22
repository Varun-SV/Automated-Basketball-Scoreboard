# # from pose_detector import *
# from program import *
# import cv2
# # # # # im = cv2.imread(r"E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\mmpose\tests\data\coco\000000000785.jpg")
# # # # # im = cv2.imread(r"E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\mmpose\tests\data\coco\000000040083.jpg")
# # im  = cv2.imread(r"E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\mmpose\tests\data\coco\000000197388.jpg")
# # # # # im = cv2.imread(r"E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\mmpose\tests\data\coco\000000196141.jpg")
# im = cv2.imread(r"E:\FinalYearProject\output\images\shot_1\frame161.jpg")
# # print(get_straight_hand_coordinates_for_every_person(im))
# print(get_the_thrower(im,190, 219))





import cv2
from program import *
import random
# Open the video file
# cap = cv2.VideoCapture(r'E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\result\demo\out_nba2k21.mp4')
cap = cv2.VideoCapture(r'E:\AI_basketball_games_video_editor\AI_basketball_games_video_editor\input\videos\vid3.mp4')

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error opening video file")
frame_num=0
shot_num=0
shot = 0
rand_frames=161

# Read the frames of the video
while cap.isOpened():
    # Read a frame from the video
    ret, frame_cv2_array = cap.read()
    # x1 = 618
    # y1 = 270
    # x2 = 628
    # y2 = 290
    # If the frame was not read, break out of the loop
    if not ret:
        break
    frame_num+=1
    # fr_mn = 198
    if frame_num==198:
        shot  = 1
    # bx = (x1+x2)//2
    # by = (y1+y2)//2
    bx = 190
    by = 219
    store_frame_in_buffer(frame_cv2_array, frame_num)

    # reverse_analysis(frame_cv2_array, frame_num,shot_num, bx,by)
    # Display the frame
    # cv2.imshow('Video', frame)
    if shot:
        shot_num+=1
        # store_the_shot(frame_cv2_array, frame_num, shot_num, bx,by,1)
        print("Shot Successfull")
        print("Frame num : %s"%(frame_num))
        reverse_analysis(frame_cv2_array, frame_num, shot_num, bx,by)
        # retrace_back_frames(frame_cv2_array,frame_num,shot_num,bx,by)
        shot = 0
    if frame_num>200:
        break

    # Wait for 25 milliseconds for a key press
    # If the 'q' key is pressed, exit the loop
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

# Release the resources
cap.release()
cv2.destroyAllWindows()