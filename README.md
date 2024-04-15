# AR-Sudoku-Solver/Realtime Sudoku Solver

## Introduction
This project is a Realtime Sudoku Solver that utilizes computer vision techniques and Convolutional Neural Networks (CNN) to recognize and solve Sudoku puzzles in real time. The application captures a live video feed of a Sudoku puzzle, processes the image using computer vision algorithms to detect the grid and extract the digits, and then employs a CNN model to recognize and solve the puzzle.

## Features

👉 Realtime Sudoku Recognition: Capture live video feed of a Sudoku puzzle and recognize the digits in real time.

👉 Computer Vision Grid Detection: Automatically detect the grid structure of the Sudoku puzzle using computer vision techniques.

👉 CNN Digit Recognition: Utilize a trained CNN model to recognize and classify the digits extracted from the Sudoku grid.

👉 Realtime Solution Display: Display the solution to the Sudoku puzzle in real time, overlaid on the video feed.

## Project Demo 
• [Click Here](https://drive.google.com/file/d/1bfQtx1im1Fl0pPNkC0wBA0HW6iDRtQHW/view?usp=sharing)

## CNN Model
• The CNN model is trained to recognize the digits 0-9 from the Sudoku puzzle. It comprises multiple convolutional and pooling layers followed by fully connected layers for classification. The model is trained using the labeled images from the dataset to learn the features of handwritten digits.

• After training the CNN model on the subset of the Chars74K dataset, it achieves an impressive accuracy of 99% on the test data. This high accuracy demonstrates the effectiveness of the CNN approach for digit recognition in Sudoku puzzles.

• Link of the dataset : [Chars74k Dataset](https://www.kaggle.com/code/anuraggupta29/optical-character-recognition-chars74k-dataset)

## Algorithm
• The Sudoku solver algorithm is implemented in Python using the backtracking technique. Sudoku is a popular number puzzle game where the objective is to fill a 9x9 grid with digits so that each column, each row, and each of the nine 3x3 subgrids contain all of the digits from 1 to 9. The backtracking algorithm efficiently solves Sudoku puzzles by recursively exploring possible solutions and backtracking when a dead-end is reached.

• Example Sudoku puzzle input (0 represents empty cells)
sudoku_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

## Requirements
• Python 3.x

• OpenCV

• NumPy

• TensorFlow

• Keras

## Installation

1. Clone the repository:
      ```
      https://github.com/MrShivam-Pal/AR-Sudoku-Solver.git
      ```

3. Install the required Python packages:
   ```python
   pip install -r requirements.
   ```

## Usage
1. Run the app.py script:
   ```
   Python app.py
   ```

2. Point your camera towards a Sudoku puzzle and watch the solver in action!

   
   
