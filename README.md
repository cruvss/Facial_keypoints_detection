
## Overview

This project implements a Facial Keypoint Detection system that identifies and marks key facial features in images [eyes, nose, and mouth] . Utilizing deep learning techniques, this application allows users to upload images and receive visual feedback with keypoints highlighted on the face. 

## Features

- Upload an image to detect facial keypoints.
- Visualize keypoints on the original image.
- Built with Streamlit for an interactive user interface.
- Utilizes a pre-trained VGG16 model as a backbone for facial keypoint detection.

## Technologies Used

- Python
- Streamlit
- PyTorch
- NumPy
- Matplotlib
- Pillow (PIL)

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:cruvss/Facial_keypoints_detection.git
   cd Facial-keypoint-detection

2. Create a virtual environment (optional but recommended):
    
   ```bash
   conda create -n facial_keypoints python=3.11 --yes
   conda activate facial_keypoints 

3. Install the required packages:

   ```bash
   pip install -r requirements.txt

## Usage

Start the Streamlit app:

   ```bash
   cd demo
   streamlit run demo.py


Upload an image and observe the detected facial keypoints visualized on the image.
