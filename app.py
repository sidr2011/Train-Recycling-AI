# -*- coding: utf-8 -*-
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from PIL import Image

# Create a dictionary to map class names to custom text
recycling_classes = {
    0.0: 'Aluminium foil', 
    1.0: 'Bottle cap',  # If metal, usually recyclable
    2.0: 'Bottle',  # Plastic or glass bottles are recyclable
    3.0: 'Broken glass',  # Recyclable in glass bins (check local policies)
    4.0: 'Can',  # Aluminium and tin cans are recyclable
    5.0: 'Carton',  # Milk and juice cartons are recyclable
    8.0: 'Lid',  # If metal or recyclable plastic
    11.0: 'Paper',  # Recyclable unless contaminated
    13.0: 'Plastic container'  # Most plastic containers are recyclable
}

trash_classes = {
    6.0: 'Cigarette',  # Not recyclable
    7.0: 'Cup',  # Most disposable cups (paper/plastic) are not recyclable due to coatings
    9.0: 'Other litter',  # General waste
    10.0: 'Other plastic',  # Miscellaneous plastics, some are non-recyclable
    12.0: 'Plastic bag - wrapper',  # Usually not recyclable curbside
    14.0: 'Pop tab',  # Small aluminium parts are often discarded
    15.0: 'Straw',  # Plastic straws are non-recyclable
    16.0: 'Styrofoam piece',  # Styrofoam is usually not recyclable
    17.0: 'Unlabeled litter'  # Unclassified waste
}

CLASS_LIST = [
    'Aluminium foil', 
    'Bottle cap', 
     'Bottle', 
     'Broken glass', 
     'Can', 
     'Carton', 
     'Cigarette', 
     'Cup', 
     'Lid',
     'Other litter', 
     'Other plastic', 
     'Paper', 
     'Plastic bag - wrapper', 
     'Plastic container',
     'Pop tab', 
     'Straw', 
     'Styrofoam piece', 
     'Unlabeled litter' 
]

#Get the absolute path of the current file
FILE = Path(__file__).resolve()

#Get the parent directory of the current file
ROOT = FILE.parent

#Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

#Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

#Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'


SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

#Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'image1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'detectedimage1.jpg'

#Videos Config
VIDEO_DIR = ROOT/'videos'
VIDEOS_DICT = {
    'video 1': VIDEO_DIR/'video1.mp4',
    'video 2': VIDEO_DIR/'video2.mp4'
}

#Model Configurations
DETECTION_MODEL = 'best.pt'


#Page Layout
st.set_page_config(
    page_title = "Recycle AI Detector",
    page_icon = "🗑️",
)

#Header
st.header("Recycle AI Detector")

#Select Confidence Value
confidence_value = 0.5 

#Selecting Detection, Segmentation, Pose Estimation Model
model_path = Path(DETECTION_MODEL)

#Load the YOLO Model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the sepcified path: {model_path}")
    st.error(e)

#Image / Video Configuration
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", SOURCES_LIST
)

st.sidebar.header("Trash / Recyclig Classes")
class_value = st.sidebar.radio(
    "Select Class", 
)



source_image = None
if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader(
        "Choose an Image....", type = ("jpg", "png", "jpeg", "bmp", "webp")
    )
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, caption = "Default Image", use_column_width=True)
            else:
                uploaded_image  =Image.open(source_image)
                st.image(source_image, caption = "Uploaded Image", use_column_width = True)
        except Exception as e:
            st.error("Error Occured While Opening the Image")
            st.error(e)
    with col2:
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption = "Detected Image", use_column_width = True)
            else:
                if st.sidebar.button("Detect Objects"):
                    result = model.predict(uploaded_image, conf = confidence_value)
                    boxes = result[0].boxes
                    result_plotted = result[0].plot()[:,:,::-1]
                    st.image(result_plotted, caption = "Detected Image", use_column_width = True)

                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as e:
                        st.error(e)
        except Exception as e:
            st.error("Error Occured While Opening the Image")
            st.error(e)

elif source_radio == VIDEO:
    source_video = st.sidebar.selectbox(
        "Choose a Video...", VIDEOS_DICT.keys()
    )
    with open(VIDEOS_DICT.get(source_video), 'rb') as video_file:
        video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
        if st.sidebar.button("Detect Video Objects"):
            try:
                video_cap = cv2.VideoCapture(
                    str(VIDEOS_DICT.get(source_video))
                )
                st_frame = st.empty()
                while (video_cap.isOpened()):
                    success, image = video_cap.read()
                    if success:
                        image = cv2.resize(image, (720, int(720 * (9/16))))
                        #Predict the objects in the image using YOLO11
                        results = model.predict(image, conf = confidence_value)
                        #Plot the detected objects on the video frame
                        boxes = results[0].boxes.xywh.cpu()
                        clss = results[0].boxes.cls.cpu().tolist()
                        annotator = Annotator(image, line_width=2, example=str('trash'))
                        for box, cls in zip(boxes, clss):
                            x, y, w, h = box
                            if float(cls) in recycling_classes:
                                label = recycling_classes[cls] + " recycling"
                            elif float(cls) in trash_classes:
                                label = trash_classes[cls] + " trash"
                            else:
                                label = str(cls)
                        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                        result_plotted = annotator.box_label([x1, y1, x2, y2], label=label, color=(0, 0, 255))
                       
                        annotated_image = annotator.result()

                    # Plot the detected objects on the video frame
                        st_frame.image(annotated_image, caption="Detected Webcam", channels="BGR", use_column_width=True)
                    else:
                        video_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error Loading Video"+str(e))
elif source_radio == WEBCAM:
    if st.sidebar.button("Detect Webcam Objects"):
        try:
            video_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            while (video_cap.isOpened()):
                success, image = video_cap.read()
                if success:
                    #Predict the objects in the image using YOLO11
                    results = model.predict(image, conf = confidence_value)

                    boxes = results[0].boxes.xywh.cpu()
                    clss = results[0].boxes.cls.cpu().tolist()
                    annotator = Annotator(image, line_width=2, example=str('trash'))
                    for box, cls in zip(boxes, clss):
                        x, y, w, h = box
                        if float(cls) in recycling_classes:
                            label = recycling_classes[cls] + " recycling"
                        elif float(cls) in trash_classes:
                            label = trash_classes[cls] + " trash"
                        else:
                            label = str(cls)
                        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                        result_plotted = annotator.box_label([x1, y1, x2, y2], label=label, color=(0, 0, 255))
                       
                    annotated_image = annotator.result()

                    # Plot the detected objects on the video frame
                    st_frame.image(annotated_image, caption="Detected Webcam", channels="BGR", use_column_width=True)
                else:
                    video_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error Loading Webcam"+str(e))
    else:
        st.write("Click on Detect Webcam Objects to start the webcam")
    video_cap = cv2.VideoCapture(0)
    
    if not video_cap.isOpened():
        st.sidebar.error("Error: Could not open webcam.")
    else:
        stframe = st.empty()
        while True:
            ret, frame = video_cap.read()
            if not ret:
                st.sidebar.error("Error: Failed to capture image.")
                break
            stframe.image(frame, channels="BGR")
    
    video_cap.release()