import os
import sys
import subprocess

def print_banner():
    print("""
====================================================================
    AI-BASED HUMAN EMOTION RECOGNITION USING FACIAL EXPRESSIONS
                     Deep Learning Minor Project
====================================================================
  Tech Stack: TensorFlow/Keras | OpenCV | Flask | Streamlit | Python
  Categories: Artificial Intelligence | Computer Vision | Deep Learning
====================================================================
""")

def print_menu():
    print("""
Select an option to execute:
--------------------------------------------------------------------
  [1] Setup Dataset (Verify FER-2013 / Generate Synthetic Data)
  [2] Train Model (CNN on FER-2013)
  [3] Evaluate Model (Accuracy, Precision, Recall, Confusion Matrix)
  [4] Test Single Image Prediction
  [5] Launch Real-Time Webcam Stream (OpenCV)
  [6] Launch Flask Web Application (http://127.0.0.1:5000)
  [7] Launch Streamlit Application (http://localhost:8501)
  [8] Exit
--------------------------------------------------------------------
""")

def run_script(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit status {e.returncode}")
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user.")

def main():
    print_banner()
    
    while True:
        print_menu()
        choice = input("Enter option [1-8]: ").strip()
        
        if choice == '1':
            print("\n[ACTION] Running Dataset Setup...")
            run_script(f"{sys.executable} utils/dataset_downloader.py")
        elif choice == '2':
            epochs = input("Enter number of epochs for training [default 15]: ").strip()
            if not epochs:
                epochs = "15"
            print(f"\n[ACTION] Starting Training for {epochs} epochs...")
            run_script(f"{sys.executable} train.py --epochs {epochs}")
        elif choice == '3':
            print("\n[ACTION] Running Model Evaluation...")
            run_script(f"{sys.executable} evaluate.py")
        elif choice == '4':
            img_path = input("Enter image path [leave empty for sample image]: ").strip()
            cmd = f"{sys.executable} predict_image.py"
            if img_path:
                cmd += f" --image \"{img_path}\""
            print("\n[ACTION] Running Single Image Prediction...")
            run_script(cmd)
        elif choice == '5':
            print("\n[ACTION] Launching Real-time Webcam Feed...")
            run_script(f"{sys.executable} webcam/webcam_predict.py")
        elif choice == '6':
            print("\n[ACTION] Launching Flask Web Server (http://127.0.0.1:5000)...")
            run_script(f"{sys.executable} flask_app/app.py")
        elif choice == '7':
            print("\n[ACTION] Launching Streamlit Web App...")
            run_script(f"streamlit run streamlit_app/app.py")
        elif choice == '8':
            print("\nExiting AI Human Emotion Recognition System. Goodbye!")
            sys.exit(0)
        else:
            print("\n[INVALID] Please select a valid option from 1 to 8.")

if __name__ == "__main__":
    main()
