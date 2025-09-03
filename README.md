import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
import pygame
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

video_path = r"D:\V\Movies\Oppenheimer (2023).mkv"
cap_video = cv2.VideoCapture(video_path)
if not cap_video.isOpened():
    print("Cannot open video")
    exit()

cap_cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hand_right = cv2.imread(r"C:\Users\admin\Downloads\Volume-PNG-Transparent-HD-Photo-removebg-preview.png", cv2.IMREAD_UNCHANGED)
hand_left = cv2.imread(r"C:\Users\admin\Downloads\3166090.png", cv2.IMREAD_UNCHANGED)
if hand_right is None or hand_left is None:
    print("Error: Hand emoji images not found")
    exit()


play_video = True
hand_alpha = 0.5
tube_height = 200
tube_width = 30
volume_level = 50
brightness_level = 1.0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max, _ = volume_interface.GetVolumeRange()

def overlay_image_alpha(img, img_overlay, pos, alpha=1.0):
    x, y = pos
    h, w = img_overlay.shape[:2]

    if img_overlay.shape[2] == 4:
        overlay_img = img_overlay[..., :3]
        overlay_mask = img_overlay[..., 3:] / 255.0 * alpha
    else:
        overlay_img = img_overlay
        overlay_mask = np.full((h, w, 1), alpha)

    roi = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = (1.0 - overlay_mask) * roi + overlay_mask * overlay_img

cv2.namedWindow("Gesture Video Player", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Gesture Video Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

pygame.mixer.init()
pygame.mixer.music.load(r"D:\V\Movies\0902 (1)\0902 (1).MP3")
pygame.mixer.music.play(-1)

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    max_num_hands=2) as hands:

    last_play_action = 0
    last_volume_action = 0
    last_brightness_action = 0

    while True:

        if play_video:
            ret_v, frame_v = cap_video.read()
            if not ret_v:
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        frame_v = np.clip(frame_v * brightness_level, 0, 255).astype(np.uint8)
        h_v, w_v, _ = frame_v.shape

        ret_c, frame_c = cap_cam.read()
        if not ret_c:
            break
        frame_c = cv2.flip(frame_c, 1)
        rgb = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h_c, w_c, _ = frame_c.shape

        right_vol = None
        left_bright = None
        play_pause = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                label = handedness.classification[0].label
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                x_thumb, y_thumb = int(thumb_tip.x * w_c), int(thumb_tip.y * h_c)
                x_index, y_index = int(index_tip.x * w_c), int(index_tip.y * h_c)

                dist = np.hypot(x_index - x_thumb, y_index - y_thumb)


                fingers_open = [
                    hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,    # Index
                    hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y, # Middle
                    hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y, # Ring
                    hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y  # Pinky
                ]

               
                if label == "Right":
                    thumb_up = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
                else:  # Left hand
                    thumb_up = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x

                if thumb_up and not any(fingers_open):
                    current_time = time.time()
                    if current_time - last_play_action > 2:  # 2 sec delay
                        play_pause = True
                        last_play_action = current_time

           
                if label == "Right":
                    current_time = time.time()
                    if current_time - last_volume_action > 2:  # 2 sec delay
                        volume_level = int(np.clip(np.interp(dist, [20, 200], [0, 100]), 0, 100))
                        right_vol = volume_level

                        vol_value = np.interp(volume_level, [0, 100], [vol_min, vol_max])
                        volume_interface.SetMasterVolumeLevel(vol_value, None)

                        last_volume_action = current_time

              
                if label == "Left":
                    current_time = time.time()
                    if current_time - last_brightness_action > 2: 
                        brightness_level = np.clip(np.interp(dist, [20, 200], [0.5, 2.0]), 0.5, 2.0)
                        left_bright = brightness_level

                        try:
                            sbc.set_brightness(int(brightness_level * 50))
                        except:
                            pass

                        last_brightness_action = current_time

                mp_drawing.draw_landmarks(frame_c, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        overlay = frame_v.copy()


        if right_vol is not None:
            try:
                emoji_size = 100
                hand_resized = cv2.resize(hand_right, (emoji_size, emoji_size))

                top = h_v // 2 - tube_height // 2
                bottom = h_v // 2 + tube_height // 2
                fill = int(bottom - (tube_height * (volume_level / 100)))
                fill = max(top, min(fill, bottom))

                if volume_level <= 33:
                    color = (255, 255, 255)
                elif volume_level <= 66:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)

                cv2.rectangle(overlay, (w_v - 120, top), (w_v - 100, bottom), (200, 200, 200), 2)
                cv2.rectangle(overlay, (w_v - 120, fill), (w_v - 100, bottom), color, -1)

         
                emoji_x = w_v - 120 - (emoji_size // 2) + 10
                emoji_y = bottom + 20
                overlay_image_alpha(overlay, hand_resized, (emoji_x, emoji_y), alpha=hand_alpha)

            except Exception as e:
                print("Error drawing volume bar:", e)

       
        if left_bright is not None:
            try:
                emoji_size = 100
                hand_resized = cv2.resize(hand_left, (emoji_size, emoji_size))

                top = h_v // 2 - tube_height // 2
                bottom = h_v // 2 + tube_height // 2
                fill = int(bottom - (tube_height * ((brightness_level - 0.5) / 1.5)))
                fill = max(top, min(fill, bottom))

                cv2.rectangle(overlay, (100, top), (120, bottom), (200, 200, 200), 2)
                cv2.rectangle(overlay, (100, fill), (120, bottom), (255, 255, 255), -1)

          
                emoji_x = 100 - (emoji_size // 2) + 10
                emoji_y = bottom + 20
                overlay_image_alpha(overlay, hand_resized, (emoji_x, emoji_y), alpha=hand_alpha)

            except Exception as e:
                print("Error drawing brightness bar:", e)

        frame_v = cv2.addWeighted(overlay, 1.0, frame_v, 0, 0)

      
        cam_h, cam_w = h_v // 8, w_v // 8
        cam_small = cv2.resize(frame_c, (cam_w, cam_h))
        frame_v[0:cam_h, w_v - cam_w:w_v] = cam_small

        if play_pause:
            play_video = not play_video

        cv2.imshow("Gesture Video Player", frame_v)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap_video.release()
cap_cam.release()
cv2.destroyAllWindows()
