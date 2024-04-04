# hfer
A Python/FastAPI/Streamlit web app for Human Facial Emotion Recognition (HFER)

## How to run locally:
1. From the `hfer` directory run:
   ``` bash
   streamlit run hfer/server/streamlit_fe.py
   ```
2. From the `hfer/hfer` directory run:
   ``` bash
   uvicorn hfer.server.main_server:app --reload
   ```
