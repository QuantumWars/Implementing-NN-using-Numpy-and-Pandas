# About the Dataset

The dataset used in this project is derived from the MNIST dataset of handwritten digits. It contains grayscale images of digits ranging from 0 to 9. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels.

## Dataset Details

### train.csv

- **File:** `train.csv`
- **Source:** [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv)
- **Description:** This file contains 42,000 rows and 785 columns. Each row represents one image, and each column represents one pixel's intensity (0-255) in the image, except for the first column, which contains the label.

#### Structure

- **Rows:** Each row represents one image.
- **Columns:** 
  - The first column, named `label`, contains the digit drawn by the user (0-9).
  - The remaining 784 columns are named `pixel0`, `pixel1`, ..., `pixel783`, and contain the pixel-values of the image.

#### Pixel Naming Convention

Each pixel column is named `pixelx`, where `x` is an integer between 0 and 783. The position of this pixel in the 28x28 image can be determined by decomposing `x` as `x = i * 28 + j`, where `i` and `j` are integers between 0 and 27.

For example, `pixel31` indicates the pixel located in the fourth column from the left and the second row from the top of the 28x28 matrix:

