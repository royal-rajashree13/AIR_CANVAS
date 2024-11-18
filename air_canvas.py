import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Constants for text display and window dimensions
TEXT_POSITION = (50, 50)

# Function to check if all five fingers are extended (full hand)
def check_full_hand(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # All fingers are extended if the thumb tip is above all other finger tips
    return (thumb_tip.y < index_tip.y and
            index_tip.y < middle_tip.y and
            middle_tip.y < ring_tip.y and
            ring_tip.y < pinky_tip.y)

# Function to check if only the index finger is extended
def check_index_finger(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Index finger extended and others curled
    return (index_tip.y < thumb_tip.y and
            index_tip.y < middle_tip.y and
            index_tip.y < ring_tip.y and
            index_tip.y < pinky_tip.y)

# Function to check if index and middle fingers are extended together
def check_index_middle_finger(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Index and middle fingers extended, others curled
    return (index_tip.y < thumb_tip.y and
            index_tip.y < middle_tip.y and
            middle_tip.y < ring_tip.y and
            middle_tip.y < pinky_tip.y)

# Function to check if all fingers are curled (fist)
def check_fist(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # All fingers curled
    return (thumb_tip.y > landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y and
            index_tip.y > landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
            middle_tip.y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
            ring_tip.y > landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
            pinky_tip.y > landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y)

# Function to recognize gestures
def recognize_gesture(hand_landmarks):
    if check_full_hand(hand_landmarks):
        return "Hello"
    elif check_index_middle_finger(hand_landmarks):
        return "What is your name?"
    elif check_fist(hand_landmarks):
        return "Goodbye"
    elif check_index_finger(hand_landmarks):
        return "Hi"
    else:
        return "Unknown Gesture"

# Initialize video capture and window dimensions
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the frame width
cap.set(4, 480)  # Set the frame height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Default message
    recognized_text = "Show gestures to recognize."

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize gestures
            recognized_text = recognize_gesture(hand_landmarks)

    # Create a black image for text output
    text_window = np.zeros_like(frame)
    cv2.putText(text_window, recognized_text, TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame with hand detection and the text output
    cv2.imshow("Hand Gesture Recognition", frame)
    cv2.imshow("Text Output", text_window)

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
