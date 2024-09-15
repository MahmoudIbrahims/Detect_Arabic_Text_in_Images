# EAST Text Detection Project

This project implements text detection using the Efficient and Accurate Scene Text (EAST) model in TensorFlow. The code processes images, detects text, and applies Non-Maximum Suppression (NMS) to filter bounding boxes.

## Requirements

Ensure you have the following dependencies installed:

- TensorFlow 
- TensorFlow Slim (`tf-slim`)
- OpenCV (`opencv-python`)
  

You can install the required libraries using `pip`:

```bash
pip install tensorflow opencv-python numpy tf-slim
```

## structue project
project/
│
├── README.md
├── data_processor.py
├── east_model.py
├── east_utils.py
├── test.py
└── content/
    └── drive/
        └── MyDrive/
            └── DS_store/
                └── DS_train/
                    └── Det_train/
                        ├── pretrained/
                        │   └── Checkpoint-500
                        ├── test/
                        │   └── images/
                        └── Detection/
                            └── detect_text/

