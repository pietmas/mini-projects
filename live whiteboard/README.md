# Virtual Painter - A Gesture-Controlled Whiteboard

This project is a virtual whiteboard application that allows you to draw, write, and interact with a digital canvas using hand gestures captured via a webcam. It leverages computer vision and machine learning technologies like OpenCV and MediaPipe for real-time hand tracking and gesture recognition.

This application was built over the project example from [Murtaza's Workshop - Robotics and AI](https://www.youtube.com/watch?v=ZiwZaAVbXQo), extending its functionality with additional features and optimizations.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Gesture Controls](#gesture-controls)


## Features

- **Real-Time Drawing**: Draw on a virtual canvas using hand gestures captured by your webcam.
- **Multiple Colors**: Choose from a palette of colors (black, red, green, blue) to draw.
- **Eraser Tool**: Erase parts of your drawing with a virtual eraser.
- **Brush Size Adjustment**: Dynamically adjust the brush size by changing the distance between your index and middle fingers.
- **Undo Functionality**: Undo your last action with thumb and pinkie finger up.
- **Save Drawing**: Save your artwork as an image file by raising your pinkie finger.
- **Clear Canvas**: Clear the entire canvas to start fresh with an open hand in eraser mode.

## Usage 

Run the main application script:
```bash
python main.py
```

## Gesture Control

- **Selection Mode**: Put your index and middle fingers up to enter in selection mode. Navigate the cursor over the desired tool or color in the header.
- **Draw**: Touch your thumb and index finger together to start drawing.
- **Adjust Brush Size**: Raise your index, middle, and ring fingers. Adjust the distance between the index and middle fingers to change the brush size. A visual indicator will display the current size.
- **Undo Action**: Raise your thumb and pinkie finger to undo the last action.
- **Save Drawing**: Raise your pinkie finger to save the current canvas as an image file.
- **Clear Canvas**: Open your hand fully (all fingers up) while in eraser mode to clear the entire canvas.

