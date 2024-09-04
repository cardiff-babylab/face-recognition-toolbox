import os
import csv
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sys
from datetime import datetime
import threading

# Lazy imports
def import_pandas():
    global pd
    import pandas as pd

def import_yolo():
    global YOLO
    from ultralytics import YOLO

def import_cv2():
    global cv2
    import cv2

def import_retinaface():
    global RetinaFace
    from retinaface import RetinaFace

def import_torch():
    global torch
    import torch

def import_requests():
    global requests
    from requests import get as requests_get

def import_tqdm():
    global tqdm
    from tqdm import tqdm


# Face recognition application class
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        # Initialize variables
        self.image_dir = ""
        self.confidence = 0.7
        self.file_paths = []
        self.model_path = "yolov8n-face.pt"  # Default model path
        self.model_type = "YOLOv8"  # Default model type
        self.save_dir = ""
        self.model = None
        
        # Initialize widgets that we'll reference later
        self.conf_value = None
        self.results_text = None
        self.progress = None
        self.progress_label = None

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Create main frames for left and right columns
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=0, column=0, sticky="nsew")

        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Left column widgets
        # File and folder selection
        ttk.Label(left_frame, text="Select File or Folder:").grid(row=0, column=0, sticky="w", pady=5)
        self.folder_path = ttk.Entry(left_frame, width=40)
        self.folder_path.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(left_frame, text="Browse File", command=self.browse_file).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Button(left_frame, text="Browse Folder", command=self.browse_folder).grid(row=3, column=0, sticky="w", pady=2)

        # Model selection
        ttk.Label(left_frame, text="Select Model:").grid(row=4, column=0, sticky="w", pady=(10, 5))
        self.model_combobox = ttk.Combobox(left_frame, values=["yolov8n-face.pt", "yolov8m-face.pt", "yolov8l-face.pt", "RetinaFace"])
        self.model_combobox.grid(row=5, column=0, sticky="ew", pady=2)
        self.model_combobox.current(2)  # Set default value
        self.model_combobox.bind("<<ComboboxSelected>>", self.update_model)

        # Confidence selection
        ttk.Label(left_frame, text="Select Confidence:").grid(row=6, column=0, sticky="w", pady=(10, 5))
        self.conf_slider = ttk.Scale(left_frame, from_=0.0, to=1.0, orient="horizontal", command=self.update_conf_value)
        self.conf_slider.grid(row=7, column=0, sticky="ew", pady=2)
        self.conf_slider.set(0.7)
        self.conf_value = ttk.Label(left_frame, text="0.7")
        self.conf_value.grid(row=8, column=0, sticky="e", pady=2)

        # Start button
        self.start_button = ttk.Button(left_frame, text="Start Recognition", command=self.start_recognition)
        self.start_button.grid(row=9, column=0, sticky="ew", pady=(10, 5))

        # Progress bar
        self.progress = ttk.Progressbar(left_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=10, column=0, sticky="ew", pady=(10, 2))

        # Progress percentage label
        self.progress_label = ttk.Label(left_frame, text="0%")
        self.progress_label.grid(row=11, column=0, sticky="e", pady=2)

        # Right column widgets
        # Results text box
        self.results_text = tk.Text(right_frame, wrap=tk.WORD, width=50, height=30)
        self.results_text.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbar for results text box
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=scrollbar.set)

        # Configure right frame grid
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)

        # Initialize logging
        self.log_message("Application initialized. Ready to start face recognition.")

    # Browse file
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("Image and Video files", "*.jpg;*.jpeg;*.png;*.bmp;*.mp4;*.avi;*.mov")]
        )
        if file_path:
            self.file_paths = [file_path]
            self.folder_path.delete(0, tk.END)
            self.folder_path.insert(0, file_path)

    # Browse folder
    def browse_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            self.file_paths = [folder_path]
            self.folder_path.delete(0, tk.END)
            self.folder_path.insert(0, folder_path)

    # Update confidence value
    def update_conf_value(self, val):
        self.confidence = float(val)
        if self.conf_value:  # Check if the widget exists
            self.conf_value.config(text=f"{self.confidence:.2f}")

    # Update model
    def update_model(self, event):
        selected_model = self.model_combobox.get()
        if selected_model == "RetinaFace":
            self.model_type = "RetinaFace"
        else:
            self.model_type = "YOLOv8"
            self.model_path = selected_model

    # Reset stdout and stderr redirection
    def reset_stdout_stderr(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # Start recognition
    def start_recognition(self):
        if not self.file_paths:
            messagebox.showerror("Error", "Please select a file or folder")
            return

        self.save_dir = filedialog.askdirectory(title="Select Folder to Save Results")
        if not self.save_dir:
            messagebox.showerror("Error", "Please select a folder to save results")
            return

        self.start_button.config(state=tk.DISABLED)
        self.progress['value'] = 0
        
        self.log_message("Starting recognition process...")
        
        # Start loading the model in a separate thread
        threading.Thread(target=self.recognition_thread, daemon=True).start()

        # Start the recognition process in a separate thread
        # threading.Thread(target=self.recognition_thread).start()

    def recognition_thread(self):
        try:
            # Download and initialize the model
            self.download_and_init_model()
            # Perform face recognition
            self.recognize_faces()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.start_button.config(state=tk.NORMAL)

    def download_and_init_model(self):
        if "yolov8" in self.model_type.lower():
            import_yolo()
            import_torch()
            self.download_yolo_model(self.model_path)
            self.model = YOLO(self.model_path)
            if torch.cuda.is_available():
                self.model.to('cuda:0')
        else:
            import_retinaface()
            # Initialize RetinaFace model if needed
            pass

    @staticmethod
    def download_yolo_model(model_name):
        if not os.path.exists(model_name):
            print(f"Downloading {model_name}...")
            import_requests()
            import_tqdm()
            base_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/"
            file_url = base_url + model_name
            
            try:
                response = requests_get(file_url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(model_name, 'wb') as file, tqdm(
                    desc=model_name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = file.write(data)
                        progress_bar.update(size)
                
                print(f"{model_name} downloaded successfully.")
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
                raise
        else:
            print(f"{model_name} already exists.")
        return model_name

    # Recognize faces
    def recognize_faces(self):
        import_pandas()
        import_cv2()

        summary_headers = ['filename', 'total_frames', 'duration', 'frames_with_faces', 'face_percentage']
        max_faces = 0
        results = []
        summary_results = []

        total_files = len(self.file_paths)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_folder = os.path.join(self.save_dir, current_time)
        os.makedirs(result_folder)
        os.makedirs(os.path.join(result_folder, "results"))

        for idx, file_path in enumerate(self.file_paths):
            # self.log_message(f"Processing file {idx + 1}/{total_files}: {file_path}")
            if os.path.isdir(file_path):
                faces, summary = self.recognize_faces_in_folder(file_path, result_folder)
                                
                # Create DataFrame
                df = pd.DataFrame(faces)
                # If you want to reorder the columns
                df = df[['image_file', 'x', 'y', 'width', 'height', 'confidence']]
                # create a new column called "prediction" which is 1 if the "confidence" is greater 0 otherwise 0
                df['prediction'] = df['confidence'].apply(lambda x: 1 if x > 0 else 0)
                # make prediction column the the second column
                df.insert(1, 'prediction', df.pop('prediction'))
                # export the dataframe to a csv file
                df.to_csv(os.path.join(result_folder, "results.csv"), index=False)
                                
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                faces, summary = self.recognize_faces_in_video(file_path, result_folder)
            else:
                
                print("else...")
                
                if "yolov8" in self.model_type.lower():
                    self.log_message(f"Processing image {idx + 1}/{total_files}: {file_path}")
                    faces = self.recognize_with_yolo(file_path, result_folder)
                else:
                    self.log_message(f"Processing image {idx + 1}/{total_files}: {file_path}")
                    faces = self.recognize_with_retinaface(file_path, result_folder)
                summary = [os.path.basename(file_path), 'N/A', 'N/A', len(faces), 'N/A']
                
                prediction = 1 if len(faces) > 0 else 0
                face_count = len(faces)
                result = [os.path.basename(file_path), prediction, face_count]

                for face in faces:
                    result.extend([face['x'], face['y'], face['width'], face['height'], face['confidence']])
                results.append(result)
                
                headers = ['filename', 'prediction', 'face_count']
                for i in range(max_faces):
                    headers.extend([f'face_{i+1}_x', f'face_{i+1}_y', f'face_{i+1}_width', f'face_{i+1}_height', f'face_{i+1}_confidence'])

                with open(os.path.join(result_folder, "results.csv"), 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                    for result in results:
                        while len(result) < len(headers):
                            result.append('')
                        writer.writerow(result)

            self.log_message(f"Faces detected: {len(faces)}")

            summary_results.append(summary)

            if len(faces) > max_faces:
                max_faces = len(faces)

            # Update progress bar and percentage label
            progress_percent = ((idx + 1) / total_files) * 100
            self.progress['value'] = progress_percent
            self.progress_label.config(text=f"{int(progress_percent)}%")
            self.root.update_idletasks()

        with open(os.path.join(result_folder, "summary.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(summary_headers)
            for summary in summary_results:
                writer.writerow(summary)

        self.start_button.config(state=tk.NORMAL)
        
        # Open results folder
        self.open_results_folder(result_folder)
        
        messagebox.showinfo("Completed", "Face recognition completed successfully!")

        # Restore stdout and stderr redirection
        # sys.stdout = TextRedirector(self.result_text)
        # sys.stderr = TextRedirector(self.result_text)

    # Implement other methods (recognize_with_yolo, recognize_with_retinaface, etc.) here...
    def recognize_with_yolo(self, file_path, result_folder, save_results=True):
        # Use the pre-initialized model
        results = self.model.predict(source=file_path, conf=self.confidence, save=save_results, save_txt=save_results, save_conf=save_results)
        
        import shutil
        
        # Copy results to the new result folder
        if results and results[0].save_dir and save_results:
            yolo_result_dir = results[0].save_dir
            for item in os.listdir(yolo_result_dir):
                s = os.path.join(yolo_result_dir, item)
                d = os.path.join(result_folder, "results", item)
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.move(s, d)
                else:
                    if os.path.exists(d):
                        os.remove(d)
                    shutil.move(s, d)

            # Delete original YOLOv8 results directory
            shutil.rmtree(yolo_result_dir)

        faces = []
        label_file = os.path.join(result_folder, "results", "labels", os.path.splitext(os.path.basename(file_path))[0] + ".txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as lf:
                lines = lf.readlines()
                for line in lines:
                    class_id, x_center, y_center, width, height, confidence = map(float, line.strip().split())
                    faces.append({'x': x_center, 'y': y_center, 'width': width, 'height': height, 'confidence': confidence})

        return faces

    # Recognize using RetinaFace
    def recognize_with_retinaface(self, file_path, result_folder, save_results=True):
        img = cv2.imread(file_path)
        detections = RetinaFace.detect_faces(img, threshold=self.confidence)
        result_img = img.copy()

        faces = []
        for key, detection in detections.items():
            facial_area = detection["facial_area"]
            confidence = detection["score"]
            x, y, w, h = facial_area[0], facial_area[1], facial_area[2] - facial_area[0], facial_area[3] - facial_area[1]
            faces.append({'confidence': confidence, 'x': x, 'y': y, 'width': w, 'height': h})
            if save_results:
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_img, f"{confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save result image
        if save_results:
            result_img_path = os.path.join(result_folder, "results", os.path.basename(file_path))
            cv2.imwrite(result_img_path, result_img)

        return faces
    
    # Recognize faces in folder
    def recognize_faces_in_folder(self, folder_path, result_folder):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        total_frames = len(image_files)
        frames_with_faces = 0
        faces = []

        # Reset progress bar
        self.progress['value'] = 0
        self.progress['maximum'] = total_frames

        for idx, image_file in enumerate(image_files):
            file_path = os.path.join(folder_path, image_file)
            
            self.log_message(f"Processing image {idx + 1}/{total_frames}: {image_file}")
            
            if "yolov8" in self.model_type.lower():
                detected_faces = self.recognize_with_yolo(file_path, result_folder, save_results=True)
            else:
                detected_faces = self.recognize_with_retinaface(file_path, result_folder, save_results=True)
            
            if detected_faces:
                frames_with_faces += 1
                detected_faces[0]['image_file'] = image_file
                detected_faces[0]['model'] = self.model_type
                detected_faces[0]['confidence'] = self.conf_value
                faces.append(detected_faces[0])
            else:
                detected_faces = [{'image_file': image_file}]
                detected_faces[0]['model'] = self.model_type
                detected_faces[0]['confidence'] = self.conf_value
                faces.append(detected_faces[0])

            # Update progress bar
            self.progress['value'] = idx + 1
            progress_percent = ((idx + 1) / total_frames) * 100
            self.progress_label.config(text=f"{progress_percent:.1f}%")
            self.root.update_idletasks()

        face_percentage = (frames_with_faces / total_frames) * 100 if total_frames > 0 else 0

        # Ensure progress bar is at 100% when done
        self.progress['value'] = total_frames
        self.progress_label.config(text="100%")
        self.root.update_idletasks()

        return faces, [os.path.basename(folder_path), total_frames, 'N/A', frames_with_faces, face_percentage]
    

    # Open results folder
    def open_results_folder(self, path):
        if os.name == 'nt':  # For Windows
            os.startfile(path)
        elif os.name == 'posix':  # For Linux and macOS
            import subprocess
            subprocess.Popen(['xdg-open', path])
    
    def log_message(self, message):
        print(message)  # Print to console
        if hasattr(self, 'results_text'):
            self.results_text.insert(tk.END, message + "\n")
            self.results_text.see(tk.END)
            self.root.update_idletasks()
