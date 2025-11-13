import cv2
import mediapipe as mp

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def classify_gesture(hand_landmarks):
    # Get landmark coordinates
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Simple heuristics for 3 gestures:
    # Open Palm: All fingertips are higher than their PIP joints
    # Fist: All fingertips are below their MCP joints
    # Thumbs Up: Thumb tip far from palm, others bent
    
    tips = [index_tip, middle_tip, ring_tip, pinky_tip]
    tip_ys = [tip.y for tip in tips]
    
    palm_open = all(tip.y < hand_landmarks.landmark[idx-2].y for tip, idx in zip(tips, 
                  [mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.PINKY_DIP]))
    
    fist = all(tip.y > hand_landmarks.landmark[idx-4].y for tip, idx in zip(tips, 
                 [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]))
    
    thumb_up = (thumb_tip.y < index_tip.y and 
                index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and 
                middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    
    thumb_down = (thumb_tip.y > index_tip.y and 
                  index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and 
                  middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    
    if palm_open:
        return "Open Palm ✋: Play/Pause"
    elif fist:
        return "Fist ✊: Increase Volume"
    elif thumb_up:
        return "Thumbs Up 👍: Success!"
    elif thumb_down:
        return "Thumbs Down 👎: Not Approved"
    else:
        return "Gesture: Unrecognized"

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Flip horizontally for natural selfie view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = "No hand detected"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_text = classify_gesture(hand_landmarks)
    
    # Display gesture and mapped action
    cv2.putText(frame, gesture_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
