# 🦺 Personal Protective Equipment (PPE) Detection

A computer vision application for detecting **Personal Protective
Equipment (PPE)** in images and videos using a custom-trained **YOLO
object detection model** and **Streamlit**.

## 🚀 Live Demo

Try the deployed application here:

**https://ppe-detector.streamlit.app/**

You can upload an image or video and run PPE detection directly from
your browser.

------------------------------------------------------------------------

## 📌 Features

-   🖼️ **Image Detection**
    -   Upload JPG, JPEG, or PNG images.
    -   Detect PPE and other trained classes.
    -   Display the original and detected images.
    -   Show detected object names and confidence percentages.
    -   Download the processed image.
-   🎥 **Video Detection**
    -   Upload MP4, MOV, or AVI videos.
    -   Run frame-by-frame object detection.
    -   Preview the processed video.
    -   Download the detected video.
-   ⚡ **YOLO-based Detection**
    -   Uses a custom-trained YOLO model stored as `best.pt`.
    -   Confidence threshold is set to `0.25`.
-   🌐 **Streamlit Web Application**
    -   Simple browser-based interface.
    -   No local setup is required to try the deployed version.

------------------------------------------------------------------------

## 🧠 Detection Classes

The model is trained to detect the following 8 classes:

1.  Boots
2.  Ear-protection
3.  Glass
4.  Glove
5.  Helmet
6.  Mask
7.  Person
8.  Vest

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   YOLO / Ultralytics
-   OpenCV
-   Streamlit
-   Pillow
-   NumPy
-   Roboflow
-   PyTorch

------------------------------------------------------------------------

## 📂 Project Structure

``` text
PPE-Detector/
│
├── app.py
├── best.pt
├── requirements.txt
├── packages.txt
├── data.yaml
├── PPE_Notebook--Part-1.ipynb
├── safety-1/
├── output/
└── README.md
```

### Important Files

  -----------------------------------------------------------------------
  File / Folder                       Description
  ----------------------------------- -----------------------------------
  `app.py`                            Streamlit application and
                                      image/video detection logic

  `best.pt`                           Trained YOLO model weights

  `requirements.txt`                  Python dependencies

  `packages.txt`                      System-level dependencies required
                                      by the deployment environment

  `data.yaml`                         Dataset configuration and class
                                      information

  `PPE_Notebook--Part-1.ipynb`        Notebook containing the model
                                      development, training, evaluation
                                      and prediction workflow

  `safety-1/`                         PPE dataset

  `output/`                           Stores processed detection outputs
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the repository URL with your GitHub repository URL.

### 2. Create a virtual environment

``` bash
python -m venv venv
```

Activate it:

**Windows**

``` bash
venv\Scripts\activate
```

**Linux / macOS**

``` bash
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

Make sure `best.pt` is present in the project root.

------------------------------------------------------------------------

## ▶️ Run the Application Locally

Run:

``` bash
streamlit run app.py
```

The application will open in your browser.

You can then select:

``` text
Image
```

or

``` text
Video
```

from the sidebar.

------------------------------------------------------------------------

## 🔍 How It Works

### Image Detection

1.  User uploads an image.
2.  The image is passed to the YOLO model.
3.  The model detects objects with a confidence threshold of `0.25`.
4.  Bounding boxes are drawn around detected objects.
5.  The detected class names and confidence percentages are displayed.
6.  The processed image can be downloaded.

### Video Detection

1.  User uploads a video.
2.  The video is temporarily stored on the server.
3.  OpenCV reads the video frame by frame.
4.  Each frame is passed through the YOLO model.
5.  Detected bounding boxes are drawn on each frame.
6.  The processed frames are written into an output video.
7.  The processed video can be viewed and downloaded.

------------------------------------------------------------------------

## 📊 Detection Output

For images, the application displays a table similar to:

  Detected Object     Confidence
  ----------------- ------------
  Person                  97.42%
  Helmet                  94.81%
  Vest                    91.36%
  Boots                   88.72%

The actual results depend on the uploaded image and model predictions.

------------------------------------------------------------------------

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

**https://ppe-detector.streamlit.app/**

The application can be accessed directly without installing Python or
any dependencies locally.

------------------------------------------------------------------------

## 📓 Model Development

The project includes a Jupyter notebook:

``` text
PPE_Notebook--Part-1.ipynb
```

The notebook contains the workflow used for dataset handling, model
training, evaluation, and prediction.

The trained model weights are saved as:

``` text
best.pt
```

The Streamlit application loads these weights directly:

``` python
model = YOLO("best.pt")
```

------------------------------------------------------------------------

## 📦 Dependencies

The project uses packages including:

``` text
ultralytics
roboflow
ipython
streamlit
glob2
opencv-python-headless
pillow
numpy
torch
torchvision
```

System-level OpenCV dependencies are specified in:

``` text
packages.txt
```

------------------------------------------------------------------------

## 🔮 Possible Future Improvements

-   Real-time webcam PPE detection
-   Detection statistics and analytics
-   PPE compliance monitoring
-   Alert system for missing PPE
-   Detection history
-   Multi-camera monitoring
-   Improved video processing performance
-   Deployment with GPU acceleration

------------------------------------------------------------------------

## 👨‍💻 Author

**Pradyumna Parida**

Computer Science & Technology

------------------------------------------------------------------------

## ⭐ Try the Application

### 👉 https://ppe-detector.streamlit.app/

If you find the project useful, consider giving the repository a ⭐ on
GitHub.
