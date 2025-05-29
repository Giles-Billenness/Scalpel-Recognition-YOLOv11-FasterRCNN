# Scalpel Detection in Surgical Video YOLOv11 FasterRCNN

This project implements and compares two deep learning models, Faster R-CNN and YOLOv11x, for the task of detecting scalpels in surgical video footage. The notebook covers the entire pipeline: video acquisition, frame extraction, manual annotation, dataset preparation, model training, inference, and performance evaluation.

## Features

* **Video Processing**: Downloads a surgical video from YouTube and extracts individual frames.
* **Custom Annotation**: Includes an interactive Matplotlib-based tool for drawing bounding box annotations for scalpels on extracted frames.
* **Dataset Preparation**:
  * Converts custom annotations into a format suitable for PyTorch's Faster R-CNN.
  * Converts custom annotations into the YOLO format.
* **Model Training**:
  * Fine-tunes a Faster R-CNN model (with a MobileNetV3 Large FPN backbone) on the custom scalpel dataset.
  * Fine-tunes a YOLOv11x model using the Ultralytics framework on the custom scalpel dataset.
* **Inference & Evaluation**:
  * Performs inference on the original surgical video using both trained models to produce annotated videos.
  * Measures and logs inference speed (ms/frame and FPS) for both models.
  * Evaluates model accuracy using Mean Average Precision (mAP) metrics.
* **Results Comparison**: Provides a summary comparing the accuracy and runtime performance of the two approaches.

## Project Structure

* `task3.ipynb`: The main Jupyter Notebook containing all the code for data processing, model training, and evaluation.
* `surgical_video/`: Directory to store the downloaded video, extracted frames, annotated data, and inference outputs.
  * `Making_an_Incision.mp4`: Example input video.
  * `extracted_frames/`: Stores frames extracted from the video.
  * `annotated/`: Stores images selected for annotation and the `annotations.json` file.
  * `yolo_converted/`: Stores the dataset converted to YOLO format, including `dataset.yaml`.
  * `inference_output/`: Stores annotated videos from model inference.
* `runs/`: Directory automatically created by Ultralytics to store YOLO training results (logs, weights, plots).

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2. **Create a Python virtual environment or easily use uv:**

    ```bash
    uv sync
    ```

3. **Install dependencies if required:**
    The core dependencies are:

    ```
    ipykernel
    ipywidgets
    matplotlib
    torch
    torchvision
    numpy
    pytubefix
    opencv-python
    torchmetrics[detection]
    scikit-learn
    ultralytics
    ```

    More details are available in the `pyproject.toml` file, which can be used to create a different virtual environment.

    An extracted packages list `requirements.txt` is included, generated using `uv export --no-emit-workspace --no-dev --no-header --no-hashes --output-file requirements.txt`

4. **CUDA (Optional but Recommended for GPU acceleration):**
    For significantly faster training and inference, ensure you have a CUDA-compatible NVIDIA GPU and have installed the appropriate versions of CUDA Toolkit and cuDNN. PyTorch and Ultralytics will automatically detect and use the GPU if available.

## Usage

**Note on Paths:** The notebook uses absolute paths for data storage (e.g., `C:\Users\giles\...`). You may need to adjust these paths to match your local environment if you run the notebook on a different machine or directory structure.

## Models Compared

* **Faster R-CNN**: Utilizes a MobileNetV3 Large FPN backbone, pre-trained on ImageNet, and fine-tuned for scalpel detection.
* **YOLOv11x**: A state-of-the-art YOLO model, fine-tuned for scalpel detection using the Ultralytics framework.

## Summary of Results

The project concludes with a comparison of the two models based on:

* **Accuracy**: mAP@0.50, mAP@0.50-0.95, mAP@0.75 (for Faster R-CNN); Precision, Recall, mAP50, mAP50-95 (for YOLO).
* **Runtime Performance**: Average inference time per frame (ms) and Frames Per Second (FPS) during video processing.

Detailed results and discussion can be found in the "Summary of Results" section at the end of the `task3.ipynb` notebook.

## Potential Future Improvements

* Expand the custom dataset with more annotated frames from diverse surgical scenarios.
* Experiment with different model backbones for Faster R-CNN or smaller/larger YOLO variants.
* Implement more sophisticated data augmentation techniques.
* Conduct more extensive hyperparameter tuning for both models.
* Investigate methods to improve temporal consistency of detections in the video.
