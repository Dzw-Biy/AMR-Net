📘 README
Notice

This repository is currently under review for IOTJ submission.
The complete source code and model implementations will be made publicly available upon paper acceptance.

At this stage, only the dataset configuration and preprocessing scripts are provided to demonstrate the data organization and preparation pipeline used in our study.

🧩 Dataset Overview

The dataset used in this work is derived from DeepSense 6G — a public multi-modal sensing platform available at https://www.deepsense6g.net/.
Specifically, we utilize Scenario 8 and Scenario 9, which provide synchronized data across four sensing modalities: RGB Image, Gray Image, LiDAR, and GPS. The datasets can be directly downloaded from the official DeepSense 6G website.

Image – RGB camera data

Gray – Grayscale image representations

LiDAR – Polar-coordinate LiDAR point clouds

GPS – Positioning data in latitude/longitude

Each modality is processed independently and standardized through a consistent preprocessing pipeline to enable multi-modal fusion and robustness evaluation.

🧠 Code Structure
The dataset preprocessing framework is organized into the following modules:

Core Processing
Includes standard preprocessing pipelines for each sensing modality:
Image_processing.py, Gray_processing.py, LiDAR_processing.py, GPS_processing.py, and All_data_processing.py.

Weather Simulation (/weather)
Contains scripts for synthetic environmental augmentation (rain, fog, snow) applied to both RGB and LiDAR modalities.

Noise & Missing Simulation (/noise)
Implements controlled noise injection and random sample/frame dropout to emulate sensor degradation.

Each module can be executed independently or integrated into the unified preprocessing workflow.

⚙️ Usage

Each preprocessing script can be executed independently.
For example:

python Image_processing.py
python LiDAR_processing.py
python All_data_processing.py


