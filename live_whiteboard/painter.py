import os
import cv2 as cv
import numpy as np
from collections import deque
from realtime_hand_tracking.hand_traching_module import HandDetector
import datetime

# ------------------------------
# Configuration and Constants
# ------------------------------

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
HEADER_PATH = os.path.join(CURRENT_PATH, "header")
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 100
CANVAS_SIZE = (720, 1280, 3)
HEADER_HEIGHT = 125

# ------------------------------
# Utility Functions
# ------------------------------

def load_header_images(folder_path):
    """Load header images from the specified folder."""
    overlay_list = []
    for img_path in sorted(os.listdir(folder_path)):
        image = cv.imread(os.path.join(folder_path, img_path))
        overlay_list.append(image)
    return overlay_list

def initialize_canvas():
    """Create a white canvas for drawing."""
    return np.ones(CANVAS_SIZE, np.uint8) * 255

def merge_images(img1, img2):
    """Merge two images with proper masking."""
    img_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    _, img_inv = cv.threshold(img_gray, 254, 255, cv.THRESH_BINARY)
    img_inv = cv.cvtColor(img_inv, cv.COLOR_GRAY2BGR)
    img1 = cv.bitwise_and(img1, img_inv)
    drawing = cv.bitwise_xor(img2, img_inv)
    result = cv.bitwise_or(img1, drawing)
    return result

# ------------------------------
# Main Application Class
# ------------------------------

class VirtualPainter:
    def __init__(self):
        # Load header images
        self.overlay_list = load_header_images(HEADER_PATH)
        self.header = self.overlay_list[0]

        # Initialize drawing variables
        self.color = (0, 0, 0)  # Default color: black
        self.brush_thickness = BRUSH_THICKNESS
        self.eraser_thickness = ERASER_THICKNESS
        self.eraser_mode = False
        self.canvas = initialize_canvas()
        self.canvas_history = deque([self.canvas.copy()], maxlen=20)  # For undo functionality
        self.xp, self.yp = 0, 0  # Previous drawing positions

        # Initialize hand detector
        self.detector = HandDetector()

        # Set up the webcam
        self.cap = cv.VideoCapture(0)
        self.cap.set(3, 1280)  # Width
        self.cap.set(4, 720)   # Height

    def run(self):
        """Main loop to run the virtual painter application."""
        while True:
            success, img = self.cap.read()
            if not success:
                break
            img = cv.flip(img, 1)  # Flip the image horizontally

            # Process the frame
            img = self.process_frame(img)

            # Display the image
            cv.imshow("Virtual Painter", img)

            # Exit condition
            if cv.waitKey(1) & 0xFF == ord('d'):
                break

        # Release resources
        self.cap.release()
        cv.destroyAllWindows()

    def process_frame(self, img):
        """Process each frame for hand detection and drawing."""
        img = self.detector.find_hands(img)
        lm_list = self.detector.find_position(img, draw=False)

        if lm_list:
            # Get fingertip positions
            x1, y1 = lm_list[8][1:]   # Index finger tip
            x2, y2 = lm_list[12][1:]  # Middle finger tip
            fingers = self.detector.fingers_up()

            if fingers == [0, 1, 1, 0, 0]:
                self.selection_mode(img, x1, y1, x2, y2)
            elif fingers == [0, 1, 0, 0, 0]:
                self.drawing_mode(img, x1, y1)
            elif fingers == [1, 0, 0, 0, 1]:
                self.undo_action(img)
            elif fingers == [0, 0, 0, 0, 1]:
                self.save_canvas(img)
            elif all(fingers) and self.eraser_mode:
                self.clear_canvas()
            elif fingers == [0, 1, 1, 1, 0]:
                self.adjust_brush_size(img)

        # Merge canvas and frame
        img = merge_images(img, self.canvas)
        img[0:HEADER_HEIGHT, 0:1280] = self.header  # Add header
        return img

    def selection_mode(self, img, x1, y1, x2, y2):
        """Handle color and tool selection."""
        self.xp, self.yp = 0, 0  # Reset previous positions
        cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (255, 0, 255), cv.FILLED)

        # Check for selection in the header area
        if y1 < HEADER_HEIGHT:
            self.select_tool(x1)

    def select_tool(self, x1):
        """Select drawing tool based on x-coordinate."""
        if 0 < x1 < 200:
            self.header = self.overlay_list[0]
            self.color = (0, 0, 0)
            self.eraser_mode = False
        elif 200 < x1 < 400:
            self.header = self.overlay_list[1]
            self.color = (0, 0, 0)
            self.eraser_mode = False
        elif 400 < x1 < 600:
            self.header = self.overlay_list[2]
            self.color = (0, 0, 255)  # Red
            self.eraser_mode = False
        elif 600 < x1 < 800:
            self.header = self.overlay_list[3]
            self.color = (0, 255, 0)  # Green
            self.eraser_mode = False
        elif 800 < x1 < 1000:
            self.header = self.overlay_list[4]
            self.color = (255, 0, 0)  # Blue
            self.eraser_mode = False
        elif 1000 < x1 < 1200:
            self.header = self.overlay_list[5]
            self.color = (255, 255, 255)  # White (eraser)
            self.eraser_mode = True

    def drawing_mode(self, img, x1, y1):
        """Handle drawing on the canvas."""
        cv.circle(img, (x1, y1), self.brush_thickness, self.color, cv.FILLED)

        if self.xp == 0 and self.yp == 0:
            self.xp, self.yp = x1, y1

        if self.eraser_mode:
            cv.line(self.canvas, (self.xp, self.yp), (x1, y1), (255, 255, 255), self.eraser_thickness)
        else:
            cv.line(self.canvas, (self.xp, self.yp), (x1, y1), self.color, self.brush_thickness)

        self.xp, self.yp = x1, y1  # Update previous positions
        self.canvas_history.append(self.canvas.copy())

    def undo_action(self, img):
        """Undo the last action on the canvas."""
        if len(self.canvas_history) > 1:
            self.canvas_history.pop()
            self.canvas = self.canvas_history[-1].copy()
            cv.putText(img, 'Undo', (600, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        else:
            cv.putText(img, 'Nothing to Undo', (500, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    
    def adjust_brush_size(self, img):
        """Adjust the brush size based on the distance between index and middle fingers."""
        length, _, _ = self.detector.find_distance(8, 12, img, draw=True)
        self.brush_thickness = int(np.interp(length, [30, 200], [5, 50]))
        cv.putText(img, f'Brush Size: {self.brush_thickness}', (50, 150),
                   cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    def save_canvas(self, img):
        """Save the current canvas to an image file."""
        cv.putText(img, 'Saving...', (550, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        # create a folder named drawing in the current directory
        if not os.path.exists(os.path.join(CURRENT_PATH, 'drawing')):
            os.makedirs(os.path.join(CURRENT_PATH, 'drawing'))
        drawing_path = os.path.join(CURRENT_PATH, 'drawing')
        cv.imwrite(os.path.join(drawing_path, f'drawing_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'), self.canvas)
        cv.putText(img, 'Saved!', (600, 200), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    def clear_canvas(self):
        """Clear the entire canvas."""
        self.canvas = initialize_canvas()
        self.canvas_history = deque([self.canvas.copy()], maxlen=20)

# ------------------------------
# Run the Application
# ------------------------------

if __name__ == "__main__":
    painter = VirtualPainter()
    painter.run()
