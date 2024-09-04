import tkinter as tk
from tkinter import ttk
import threading
import time
import traceback

class SplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # Remove window decorations
        self.root.attributes('-topmost', True)
        
        # Center the window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = 300
        height = 100
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        self.frame = ttk.Frame(self.root)
        self.frame.pack(expand=True, fill='both')
        
        self.label = ttk.Label(self.frame, text="Application is starting...\nPlease wait.", font=("Helvetica", 12))
        self.label.pack(pady=20)
        
        self.progress = ttk.Progressbar(self.frame, mode="indeterminate")
        self.progress.pack(pady=10)
        self.progress.start()

    def destroy(self):
        if self.root.winfo_exists():
            self.progress.stop()  # Stop the progress bar
            self.root.destroy()

def load_main_app(splash):
    try:
        # Simulate long-running initialization tasks
        time.sleep(3)  # Adjust this value as needed
        
        # Import your main application here
        from face_recognition_app import FaceRecognitionApp
        
        # Schedule the creation of the main app on the main thread
        if splash.root.winfo_exists():
            splash.root.after(0, create_main_app, splash, FaceRecognitionApp)
    except Exception as e:
        print(f"Error loading main app: {e}")
        traceback.print_exc()
        if splash.root.winfo_exists():
            splash.root.after(0, show_error_and_quit, splash)

def create_main_app(splash, FaceRecognitionApp):
    try:
        # Destroy splash screen
        splash.destroy()
        
        # Create and show the main application window
        root = tk.Tk()
        app = FaceRecognitionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error creating main app: {e}")
        traceback.print_exc()
        show_error_and_quit(splash)

def show_error_and_quit(splash):
    try:
        if hasattr(splash, 'root') and splash.root.winfo_exists():
            splash.destroy()
    except:
        pass  # If there's an error destroying the splash screen, just ignore it
    
    error_window = tk.Tk()
    error_window.title("Error")
    ttk.Label(error_window, text="An error occurred while loading the application.", font=("Helvetica", 12)).pack(pady=20)
    ttk.Button(error_window, text="Quit", command=error_window.quit).pack(pady=10)
    error_window.mainloop()

def main():
    splash = SplashScreen()
    
    # Start loading the main app in a separate thread
    threading.Thread(target=load_main_app, args=(splash,), daemon=True).start()
    
    # Show the splash screen
    splash.root.mainloop()

if __name__ == "__main__":
    main()