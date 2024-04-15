import time
import streamlit as st
from streamlit_option_menu import option_menu
import ultralytics
import pickle
from PIL import Image
import cv2
import inference
from streamlit_webrtc import webrtc_streamer

st.set_page_config(
        page_title="RoadSafety.AI",
        page_icon="🩺"
   
)
api ="B23RU3MxCJVf0RabBD5c"
project = "testing-pcqgi"
version ="/3" 
names= {0: 'accident'}
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
    selected=option_menu("RoadSafety.AI",
    ['Accident Detection',
    'Distracted Driver Detection'],
    icons=['activity','person','person','bandaid-fill'],
    default_index=0,
 
    )




if selected == "Accident Detection":
    st.title("Accident Detection")
    form = st.form(key="my_form")
    camera_name = form.text_input("Camera Name (Optional)", "", key="camera_name",placeholder="NH-47 Highway")
    rtsp_address = form.text_input("RTSP Address of Live Camera", "", key="rtsp_address",placeholder="rtsp://server.example.org:8080/test.sdp")
    submit_button = form.form_submit_button("Submit")
    if submit_button:
     if not rtsp_address:
        st.warning("Please fill in the RTSP address field.")
    if submit_button and rtsp_address:
        st.title(f"Live Camera Feed ({camera_name})")
        
        checklist=False
        model = inference.get_model(project+version,api_key=api)
        cap = cv2.VideoCapture("main.mp4")
        if not cap.isOpened():
            st.error("Error: Failed to open RTSP stream")
        else:
            frame_container = st.empty()
            while True:
               
                if checklist==False:
                    checklist=set()
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Failed to read frame from RTSP stream")
                    break
                prediction=model.infer(frame,confidence=0.85)
                # print(prediction)
                for prediction in prediction[0].predictions:  # Accessing the list of predictions within the first ObjectDetectionInferenceResponse
                    x = int(prediction.x)
                    y = int(prediction.y)
                    width = int(prediction.width)
                    height = int(prediction.height)
                    predicted_class=prediction.class_name
                    if checklist is not True:
                        checklist.add(predicted_class)
                    x0 = x - width / 2
                    x1 = x + width / 2
                    y0 = y - height / 2
                    y1 = y + height / 2

                    start_point = (int(x0), int(y0))
                    end_point = (int(x1), int(y1))
            
                    cv2.rectangle(frame, start_point, end_point, color=(0,0,0), thickness=4)
                    if str(type(checklist))=="<class 'set'>":
                        for item in checklist:
                       
                            if item in names.values():
                                
                                    checklist=True
                                    st.toast("Survillence officer was aelrted via phone call")

                                    break
                
                frame_container.image(frame, channels="BGR", use_column_width=True)

                
                
                
                # time.sleep(0.05)  # 20 frames per second (1 / 0.05)


# Distracted Driver Detection
if (selected == 'Distracted Driver Detection'):
    st.title("Distracted Driver Detection")
    webrtc_streamer(key="sample") 


