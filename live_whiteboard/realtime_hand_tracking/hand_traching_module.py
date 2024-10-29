import cv2 as cv
import mediapipe as mp
import numpy as np

class HandDetector:
    """
    A class to detect and track hands using MediaPipe.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initialize the hand detector with the given parameters.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.lmList = []

    def find_hands(self, img, draw=True):
        """
        Detect hands in an image and draw landmarks if draw is True.
        """
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Draw hand landmarks
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img, handNo=0, draw=True):
        """
        Find the positions of hand landmarks in the image.
        """
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # Convert normalized coordinates to pixel values
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

                # Draw circles on landmarks
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        return self.lmList

    def fingers_up(self):
        """
        Check which fingers are up and return a list of binary values.
        """
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]

        # Check if any landmarks are detected
        if not self.lmList:
            return []

        # Thumb
        if self.lmList[tip_ids[0]][1] < self.lmList[tip_ids[0] - 1][1]:
            fingers.append(1)  # Thumb is open
        else:
            fingers.append(0)  # Thumb is closed

        # Other four fingers
        for id in range(1, 5):
            if self.lmList[tip_ids[id]][2] < self.lmList[tip_ids[id] - 2][2]:
                fingers.append(1)  # Finger is open
            else:
                fingers.append(0)  # Finger is closed

        return fingers

    def find_distance(self, p1, p2, img=None, draw=True, r=15, t=3):
        """
        Find the distance between two landmarks.
        """
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        if draw and img is not None:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]
