# Scalpel Detection in Surgical Video YOLOv11 FasterRCNN

This project implements and compares two deep learning models, Faster R-CNN and YOLOv11x, for the task of detecting scalpels in surgical video footage. The notebook covers the entire pipeline: video acquisition, frame extraction, manual annotation, dataset preparation, model training, inference, and performance evaluation.

## Key Highlights

| Feature                   | Faster R-CNN (MobileNetV3 Large FPN)                                                       | YOLO (YOLOv11x)                                                                                                                 |
|---------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Approach                  | Fine-tuned on custom scalpel dataset. Pre-trained on ImageNet. Standard data augmentation. | Fine-tuned yolo11x.pt on the same custom dataset (YOLO format). Trained with Ultralytics library (100 epochs, AdamW, lr0=1e-5). |
| mAP@0.50                  | 1.0000                                                                                     | 0.995 (mAP50)                                                                                                                   |
| mAP@0.50-0.95             | 0.8257                                                                                     | 0.88 (mAP50-95)                                                                                                                 |
| mAP@0.75                  | 1.0000                                                                                     | N/A                                                                                                                             |
| Precision                 | N/A                                                                                        | 0.99                                                                                                                            |
| Recall                    | N/A                                                                                        | 1.0                                                                                                                             |
| Total Time (5430 frames)  | 108.9159 seconds                                                                           | 178.9 seconds                                                                                                                   |
| Avg. Inference Time/Frame | 20.06 ms                                                                                   | 25.8 ms                                                                                                                         |
| Inference-only FPS        | 49.85 FPS                                                                                  | 38.76 FPS                                                                                                                       |

Detailed results and discussion can be found in the "Summary of Results" section at the end of the `task3.ipynb` notebook.

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

This project uses `uv` for Python environment and package management.

2. **Install uv:**
    Follow the official installation instructions for `uv` from [docs.astral.sh/uv/getting-started/](https://docs.astral.sh/uv/getting-started/installation/).

3. **Create a Python virtual environment or easily use uv:**

    ```bash
    uv sync
    ```

4. **Install dependencies if required:**
    The core dependencies are:

    ```bash
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

5. **CUDA (Optional but Recommended for GPU acceleration):**
    For significantly faster training and inference, ensure you have a CUDA-compatible NVIDIA GPU and have installed the appropriate versions of CUDA Toolkit and cuDNN. PyTorch and Ultralytics will automatically detect and use the GPU if available.

## Usage

**Note on Paths:** The notebook uses absolute paths for data storage (e.g., `C:\Users\giles\...`). You may need to adjust these paths to match your local environment if you run the notebook on a different machine or directory structure.

## Models Compared

* **Faster R-CNN**: Utilizes a MobileNetV3 Large FPN backbone, pre-trained on ImageNet, and fine-tuned for scalpel detection.
* **YOLOv11x**: A state-of-the-art YOLO model, fine-tuned for scalpel detection using the Ultralytics framework.

## Conclusion

Both models were successfully trained to detect scalpels.
The Faster R-CNN model demonstrated notably faster inference speeds compared to the YOLO model,
making it more suitable for real-time applications if its accuracy is acceptable.
Alternative backbones or further optimizations could potentially improve Faster R-CNN's performance, especially in terms of inference speed.

However the YOLOv11 model is a more recent architecture and generally offers a good balance between speed and accuracy, especially in real-time applications.
As the YOLOv11 xl model varient was used, experimentation with smaller variants (like YOLOv11l (~half the size) or YOLOv11m) could yield faster inference times while maintaining good accuracy especially with larger datasets.

The performance metrics on the validation set indicate that the models are capable of detecting scalpels with high precision and recall.
Even with the extreamly small dataset, the models achieved high mAP scores, indicating that they can generalize well to the task of scalpel detection in surgical videos.

The choice between them would depend on the specific requirements for accuracy versus speed for the target application,
as well as the level of control and customization needed for the model and its training.

Further hyperparameter tuning, dataset augmentation, or using different model backbones/sizes could potentially improve the performance of both approaches.

Qualitative analysis of the model outputs on the video frames shows that both models effectively detect scalpels, with bounding boxes accurately placed around the instruments.
However with the Faster R-CNN model, the bounding boxes are more erratic and temporally inconsistent, while the YOLO model provides more stable and consistent detections across frames
as well as being less prone to false positives that could be very important for surgical applications.

## Potential Future Improvements

* Expand the custom dataset with more annotated frames from diverse surgical scenarios.
* Experiment with different model backbones for Faster R-CNN or smaller/larger YOLO variants.
* Implement more sophisticated data augmentation techniques.
* Conduct more extensive hyperparameter tuning for both models.
* Investigate methods to improve temporal consistency of detections in the video.
