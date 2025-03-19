import os
import cv2 as cv
import numpy as np
from collections import deque
from enum import Enum
import logging
import datetime
from realtime_hand_tracking.hand_traching_module import HandDetector

# Configuration and Constants
# ------------------------------

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 100
MAX_UNDO = 20
WIDTH, HEIGHT = 640, 480
CANVAS_SIZE = (HEIGHT, WIDTH, 3)
HEADER_PATH = os.path.join(CURRENT_PATH, f"header\{WIDTH}")
# Get the height of the header image from the first image
HEADER_HEIGHT = cv.imread(os.path.join(HEADER_PATH, "1.jpg")).shape[0]



# Set up logging
logging.basicConfig(level=logging.INFO)

# Gesture Enum for clarity
class Gesture(Enum):
    SELECTION = 1
    DRAWING = 2
    UNDO = 3
    SAVE = 4
    CLEAR = 5
    ADJUST_BRUSH_SIZE = 6
    NONE = 7

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
        self.drawing_flag = False
        self.canvas = initialize_canvas()
        self.canvas_history = deque([self.canvas.copy()], maxlen=MAX_UNDO)  # For undo functionality
        self.xp, self.yp = 0, 0  # Previous drawing positions

        # Initialize hand detector
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)

        # Set up the webcam
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    def run(self):
        """Main loop to run the virtual painter application."""
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    logging.error("Failed to read from webcam.")
                    break
                img = cv.flip(img, 1)  # Flip the image horizontally

                # Process the frame
                img = self.process_frame(img)

                # Display the image
                cv.imshow("Virtual Painter", img)

                # Exit condition
                if cv.waitKey(1) & 0xFF == ord('d'):
                    break

        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            # Release resources
            self.cap.release()
            cv.destroyAllWindows()

    def process_frame(self, img):
        """Process each frame for hand detection and drawing."""
        img = self.detector.find_hands(img)
        lm_list = self.detector.find_position(img)

        if lm_list:
            # Get fingertip positions
            x1, y1 = lm_list[8][1:]   # Index finger tip
            x2, y2 = lm_list[12][1:]  # Middle finger tip
            fingers = self.detector.fingers_up()

            gesture = self.detect_gestures(fingers, img)

            if gesture == Gesture.SELECTION:
                self.selection_mode(img, x1, y1, x2, y2)
            elif gesture == Gesture.DRAWING:
                self.drawing_mode(img, x1, y1)
            elif gesture == Gesture.UNDO:
                self.undo_action(img)
            elif gesture == Gesture.SAVE:
                self.save_canvas(img)
            elif gesture == Gesture.CLEAR:
                self.clear_canvas(img)
            elif gesture == Gesture.ADJUST_BRUSH_SIZE:
                self.adjust_brush_size(img)
            else:
                self.reset_pen_position()
        else:
            self.reset_pen_position()

        # Merge canvas and frame
        img = merge_images(img, self.canvas)
        img[0:HEADER_HEIGHT, 0:WIDTH] = self.header
        # Display current color and brush size
        cv.putText(img, f'Color: {self.color}', (10, 700), cv.FONT_HERSHEY_PLAIN, 2, self.color, 2)
        cv.putText(img, f'Brush Size: {self.brush_thickness}', (10, 680), cv.FONT_HERSHEY_PLAIN, 2, self.color, 2)
        return img

    def detect_gestures(self, fingers, img):
        """Detect the current gesture based on finger positions."""
        if fingers == [0, 1, 1, 0, 0]:
            return Gesture.SELECTION
        elif self.detector.is_thumb_index_touching(img=img) and self.drawing_flag:
            return Gesture.DRAWING
        elif fingers == [1, 0, 0, 0, 1]:
            return Gesture.UNDO
        elif fingers == [0, 0, 0, 0, 1]:
            return Gesture.SAVE
        elif all(fingers) and self.eraser_mode:
            return Gesture.CLEAR
        elif fingers == [0, 1, 1, 1, 0]:
            return Gesture.ADJUST_BRUSH_SIZE
        else:
            return Gesture.NONE

    def selection_mode(self, img, x1, y1, x2, y2):
        """Handle color and tool selection."""
        self.reset_pen_position()
        cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (255, 0, 255), cv.FILLED)

        # Check for selection in the header area
        if y1 < HEADER_HEIGHT:
            self.select_tool(x1)

    def select_tool(self, x1):
        """Select drawing tool based on x-coordinate."""
        selection_slices = [WIDTH // 6 * i for i in range(1, 7)]
        if 0 < x1 < selection_slices[0]:
            self.header = self.overlay_list[0]
            self.color = (0, 0, 0)
            self.eraser_mode = False
            self.drawing_flag = False
        elif selection_slices[0] < x1 < selection_slices[1]:
            self.header = self.overlay_list[1]
            self.color = (0, 0, 0) # Black
            self.eraser_mode = False
            self.drawing_flag = True
        elif selection_slices[1] < x1 < selection_slices[2]:
            self.header = self.overlay_list[2]
            self.color = (0, 0, 255)  # Red
            self.eraser_mode = False
            self.drawing_flag = True
        elif selection_slices[2] < x1 < selection_slices[3]:
            self.header = self.overlay_list[3]
            self.color = (0, 255, 0)  # Green
            self.eraser_mode = False
            self.drawing_flag = True
        elif selection_slices[3] < x1 < selection_slices[4]:
            self.header = self.overlay_list[4]
            self.color = (255, 0, 0)  # Blue
            self.eraser_mode = False
            self.drawing_flag = True
        elif selection_slices[4] < x1 < WIDTH:
            self.header = self.overlay_list[5]
            self.color = (255, 255, 255)  # White (eraser)
            self.eraser_mode = True
            self.drawing_flag = True

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
        self.reset_pen_position()
        if len(self.canvas_history) > 1:
            self.canvas_history.pop()
            self.canvas = self.canvas_history[-1].copy()
            cv.putText(img, 'Undo', (600, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            logging.info("Undo action performed.")
        else:
            cv.putText(img, 'Nothing to Undo', (500, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            logging.info("No actions to undo.")

    def adjust_brush_size(self, img):
        """Adjust the brush size based on the distance between index and middle fingers."""
        self.reset_pen_position()
        length, _, _ = self.detector.find_distance(8, 12, img)
        self.brush_thickness = int(np.interp(length, [30, 200], [5, 50]))
        cv.putText(img, f'Brush Size: {self.brush_thickness}', (50, 150),
                   cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        logging.info(f"Brush size adjusted to {self.brush_thickness}.")

    def save_canvas(self, img):
        """Save the current canvas to an image file."""
        self.reset_pen_position()
        cv.putText(img, 'Saving...', (550, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        # Create a folder named 'drawing' in the current directory
        drawing_path = os.path.join(CURRENT_PATH, 'drawing')
        if not os.path.exists(drawing_path):
            os.makedirs(drawing_path)
        # Save the canvas without the header or video feed
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        cv.imwrite(os.path.join(drawing_path, f'drawing_{timestamp}.png'), self.canvas)
        cv.putText(img, 'Saved!', (600, 200), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        logging.info(f"Canvas saved as drawing_{timestamp}.png.")

    def clear_canvas(self, img):
        """Clear the entire canvas."""
        self.reset_pen_position()
        self.canvas = initialize_canvas()
        self.canvas_history = deque([self.canvas.copy()], maxlen=MAX_UNDO)
        cv.putText(img, 'Canvas Cleared', (500, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        logging.info("Canvas cleared.")

    def reset_pen_position(self):
        """Reset the pen position."""
        self.xp, self.yp = 0, 0

# Run the Application
# ------------------------------

if __name__ == "__main__":
    painter = VirtualPainter()
    painter.run()
