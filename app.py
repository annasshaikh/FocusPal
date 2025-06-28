import tkinter as tk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import sqlite3
import time
import datetime
import sys
import os # To check if DB file exists

class PresenceTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Person Presence Tracker")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- State Variables ---
        self.running = False # Overall application state (Started/Stopped)
        self.paused = False  # Application state (Paused/Not Paused)

        self.camera = None   # OpenCV VideoCapture object
        self.model = None    # YOLOv8 model object
        self.conn = None     # SQLite database connection
        self.cursor = None   # SQLite database cursor

        # --- Session Tracking Variables ---
        self.session_active = False  # True if a person session is currently ongoing
        self.session_start_time = None # time.time() timestamp when the current session started
        self.current_session_duration = 0 # Duration of the current session in seconds

        self.last_detection_run_time = time.time() # time.time() timestamp of the last YOLO inference run
        # detection_interval controls how often YOLO inference is performed (in seconds)
        # 1.0s when person detected or recently absent, 10.0s when absent for a while
        self.detection_interval = 1.0 # Start checking frequently

        self.absence_start_time = None # time.time() timestamp when absence was first detected
        self.absence_threshold = 3.0   # Seconds of consecutive absence to end a session

        # --- Daily Total Time ---
        self.total_time_today = 0 # Load from DB on startup and update when sessions are saved

        # --- GUI Elements ---
        self.camera_label = None
        self.status_label = None
        self.today_time_label = None
        self.session_time_label = None
        self.start_button = None
        self.pause_resume_button = None
        self.stop_button = None

        # --- Initialize ---
        self.connect_db() # Connect to DB first to load initial total time
        self.total_time_today = self.get_total_time_for_today()
        self.create_widgets() # Setup GUI elements
        self.update_gui_state() # Set initial button states etc.


    # --- Database Methods ---
    def connect_db(self):
        db_exists = os.path.exists('presence.db')
        try:
            self.conn = sqlite3.connect('presence.db')
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS presence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT,
                    end_time TEXT,
                    duration REAL
                )
            ''')
            self.conn.commit()
            if not db_exists:
                print("Database 'presence.db' created and connected successfully.")
            else:
                print("Database 'presence.db' connected successfully.")
        except sqlite3.Error as e:
            print(f"Database error during connection or table creation: {e}")
            self.conn = None
            self.cursor = None
            # Indicate DB connection failed in GUI? Or disable DB related features?
            # For now, just print error.

    def close_db(self):
        if self.conn:
            try:
                self.conn.close()
                print("Database connection closed.")
            except sqlite3.Error as e:
                print(f"Database error during closing: {e}")
            self.conn = None
            self.cursor = None

    def save_session(self, start_time, end_time, duration):
        if self.cursor:
            try:
                # Convert timestamps to human-readable strings
                start_str = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
                end_str = datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")

                self.cursor.execute("INSERT INTO presence (start_time, end_time, duration) VALUES (?, ?, ?)",
                                    (start_str, end_str, duration))
                self.conn.commit()
                print(f"Session saved: Start={start_str}, End={end_str}, Duration={duration:.2f}s")

                # Update the total time for today after saving a session
                self.total_time_today = self.get_total_time_for_today()
                if self.today_time_label: # Ensure label exists before updating
                     self.today_time_label.config(text=f"Total Time Today: {self.format_duration(self.total_time_today)}")

            except sqlite3.Error as e:
                print(f"Failed to save session to database: {e}")
        else:
            print("Database not connected. Session not saved.")

    def get_total_time_for_today(self):
        total_duration = 0
        if self.cursor:
            try:
                today = datetime.date.today().strftime("%Y-%m-%d")
                # Select sum of duration for sessions whose start date is today
                self.cursor.execute("SELECT SUM(duration) FROM presence WHERE DATE(start_time) = ?", (today,))
                result = self.cursor.fetchone()
                if result and result[0] is not None:
                    total_duration = result[0]
            except sqlite3.Error as e:
                print(f"Failed to retrieve total time from database: {e}")
        return total_duration

    # --- GUI Methods ---
    def create_widgets(self):
        # Video feed label - will display the webcam feed
        self.camera_label = tk.Label(self.root)
        self.camera_label.pack(pady=10)

        # Status Label - shows detection status
        self.status_label = tk.Label(self.root, text="Status: Stopped", font=('Arial', 12))
        self.status_label.pack()

        # Time Labels - show total time today and current session time
        self.today_time_label = tk.Label(self.root, text=f"Total Time Today: {self.format_duration(self.total_time_today)}", font=('Arial', 12))
        self.today_time_label.pack()

        self.session_time_label = tk.Label(self.root, text="Current Session: 0s", font=('Arial', 12))
        self.session_time_label.pack()

        # Control Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start", command=self.start_tracker)
        self.start_button.grid(row=0, column=0, padx=5)

        self.pause_resume_button = tk.Button(button_frame, text="Pause", command=self.pause_resume_tracker, state=tk.DISABLED)
        self.pause_resume_button.grid(row=0, column=1, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_tracker, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, padx=5)

    def update_gui_state(self):
        # Updates button states and status label text/color based on application state
        if not self.running and not self.paused: # State: Stopped
            self.start_button.config(state=tk.NORMAL)
            self.pause_resume_button.config(state=tk.DISABLED, text="Pause")
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped", fg="black")
        elif self.running and not self.paused: # State: Running
            self.start_button.config(state=tk.DISABLED)
            self.pause_resume_button.config(state=tk.NORMAL, text="Pause")
            self.stop_button.config(state=tk.NORMAL)
            # Status label text/color updated in update_frame based on detection results
        elif self.paused: # State: Paused
            self.start_button.config(state=tk.DISABLED)
            self.pause_resume_button.config(state=tk.NORMAL, text="Resume")
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Paused", fg="orange")

    def format_duration(self, seconds):
        # Helper function to format seconds into Hh M:S format
        if seconds is None:
            seconds = 0
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}h {m:02d}m {s:02d}s"

    # --- Core Logic Methods ---
    def start_tracker(self):
        if not self.running and not self.paused:
            print("Starting tracker...")
            self.running = True
            self.paused = False
            self._pause_start_time = None # Ensure pause tracking is reset

            # Ensure database is connected
            if not self.conn or not self.cursor:
                 print("Database connection failed. Attempting to reconnect.")
                 self.connect_db()
                 if not self.conn or not self.cursor:
                      print("Failed to connect to database. Cannot start.")
                      self.running = False
                      self.update_gui_state()
                      return

            # Load YOLO model
            try:
                self.model = YOLO('yolov8n.pt')
                print("YOLOv8n model loaded successfully.")
            except Exception as e:
                 print(f"Failed to load YOLO model: {e}")
                 self.running = False
                 self.close_db()
                 self.update_gui_state()
                 return

            # Open Webcam
            self.camera = cv2.VideoCapture(0) # 0 is default camera index
            if not self.camera.isOpened():
                print("Error: Could not open webcam.")
                self.running = False
                self.model = None # Release model reference
                self.close_db()
                self.update_gui_state()
                # Display an error message in the GUI?
                self.status_label.config(text="Status: Camera Error", fg="red")
                return

            # Reset session and detection state for a fresh start
            self.session_active = False
            self.session_start_time = None
            self.current_session_duration = 0
            self.absence_start_time = None
            self.last_detection_run_time = time.time() # Start detection timer now
            self.detection_interval = 1.0 # Start checking frequently

            # Refresh total time from DB
            self.total_time_today = self.get_total_time_for_today()
            self.today_time_label.config(text=f"Total Time Today: {self.format_duration(self.total_time_today)}")
            self.session_time_label.config(text="Current Session: 0s")


            self.update_gui_state() # Update buttons
            self.update_frame() # Start the main frame processing loop

    def pause_resume_tracker(self):
        if not self.running:
            # Cannot pause/resume if not running
            return

        if not self.paused: # Currently Running, needs Pause
            print("Pausing tracker...")
            self.paused = True
            self._pause_start_time = time.time() # Record the time when pause was initiated
            self.update_gui_state() # Update buttons to show "Resume"
            self.status_label.config(text="Status: Paused", fg="orange") # Update status label

        else: # Currently Paused, needs Resume
            print("Resuming tracker...")
            # Calculate how long the tracker was paused
            pause_duration = time.time() - self._pause_start_time if self._pause_start_time is not None else 0

            # Adjust timers to account for the pause duration
            if self.session_active:
                # Shift the session start time forward by the pause duration
                # This way, when we calculate duration using current_time, it's correct.
                self.session_start_time += pause_duration
                # self.current_session_duration will be recalculated in update_frame
                # The session timer display might jump initially until the next frame update

            if self.absence_start_time is not None:
                 # Shift the absence start time forward as well
                 self.absence_start_time += pause_duration

            # Shift the last detection run time forward to resume detection checks correctly
            self.last_detection_run_time += pause_duration


            self.paused = False
            self._pause_start_time = None # Reset pause start time

            self.update_gui_state() # Update buttons back to "Pause"
            # Status label will be updated by the detection logic in update_frame
            self.update_frame() # Resume the frame processing loop

    def stop_tracker(self):
        if self.running or self.paused:
            print("Stopping tracker...")
            self.running = False
            self.paused = False
            self._pause_start_time = None # Ensure pause tracking is reset

            # End current session if it's active and save it
            if self.session_active:
                 # A session ends when Stop is pressed, irrespective of the 3s rule.
                 session_end_time = time.time()
                 duration = session_end_time - self.session_start_time
                 if duration > 0.1: # Only save sessions with meaningful duration
                     self.save_session(self.session_start_time, session_end_time, duration)

                 # Reset session state
                 self.session_active = False
                 self.session_start_time = None
                 self.current_session_duration = 0
                 # Update GUI immediately
                 self.session_time_label.config(text="Current Session: 0s")
                 print("Current session ended and saved due to stop.")

            # Release the webcam
            if self.camera:
                self.camera.release()
                self.camera = None
                # Clear the video feed display in the GUI
                if self.camera_label:
                     self.camera_label.config(image=None)
                     self.camera_label.image = None # Crucial to prevent garbage collection

            # Close the database connection
            self.close_db()

            # Release the YOLO model
            self.model = None

            # Final GUI state update
            self.update_gui_state()
            self.status_label.config(text="Status: Stopped", fg="black")
            # Update total time one last time from DB in case session was saved
            self.total_time_today = self.get_total_time_for_today()
            self.today_time_label.config(text=f"Total Time Today: {self.format_duration(self.total_time_today)}")

    def update_frame(self):
        # This function is called repeatedly to process frames and update the GUI
        # The frequency of calls is controlled by root.after()

        if not self.running:
             # If tracker is stopped, prevent further calls
             return

        if self.paused:
             # If paused, just reschedule the next call to keep the loop alive
             # No frame processing or detection happens when paused
             self.root.after(30, self.update_frame) # Schedule next call (~30-40 FPS attempt)
             return # Exit the rest of the function


        # --- Process Frame (only if running and not paused) ---
        ret, frame = self.camera.read()
        if ret:
            # Convert OpenCV frame (BGR) to RGB for Pillow
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to Pillow Image
            img = Image.fromarray(cv2image)

            # Optional: Resize image to fit GUI or improve performance
            # img = img.resize((640, 480))

            # Convert Pillow Image to Tkinter PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            # Update the label to display the new frame
            self.camera_label.imgtk = imgtk # Keep a reference!
            self.camera_label.config(image=imgtk)

            current_time = time.time()

            # --- Detection Logic (runs based on detection_interval, not every frame) ---
            # Check if it's time to run YOLO inference
            if current_time - self.last_detection_run_time >= self.detection_interval:
                 # print(f"Running detection (interval: {self.detection_interval:.1f}s)")
                 self.last_detection_run_time = current_time # Record time of this detection run

                 person_detected_in_this_run = False
                 try:
                     # Perform detection using YOLO model
                     # classes=0 filters for 'person' class
                     # conf=0.7 sets confidence threshold
                     # persist=True helps with tracking between frames (optional but can improve session logic robustness)
                     # verbose=False suppresses model output
                     results = self.model.track(frame, classes=0, conf=0.7, persist=True, verbose=False)
                     boxes = results[0].boxes # Get the bounding boxes

                     # Check if any boxes correspond to detected persons
                     person_detected_in_this_run = len(boxes) > 0

                 except Exception as e:
                      # Handle potential errors during detection (e.g., CUDA error, model issue)
                      print(f"Error during YOLO detection: {e}")
                      # Update status to indicate error, maybe keep detection interval short to retry
                      self.status_label.config(text="Status: Detection Error", fg="orange")
                      # Decide if error should stop the tracker? For now, just log and continue.
                      # Set person_detected_in_this_run to False in case of error
                      person_detected_in_this_run = False
                      # The detection interval might stay short, leading to more retries.

                 # --- Session and Absence Logic based on detection result ---
                 if person_detected_in_this_run:
                      # Person detected
                      self.status_label.config(text="Status: Detected ✅", fg="green")
                      self.absence_start_time = None # Reset the absence timer as presence is confirmed
                      self.detection_interval = 1.0 # When a person is detected, check frequently (every 1 second)

                      if not self.session_active:
                          # If a person is detected and no session is active, start a new one
                          self.session_start_time = current_time
                          self.session_active = True
                          print("Session started.")

                 else: # Person not detected in this detection run
                      self.status_label.config(text="Status: Not Detected ❌", fg="red")

                      if self.session_active:
                          # If a session is active, check for consecutive absence
                          if self.absence_start_time is None:
                              # This is the first detection run where person was not detected during an active session
                              self.absence_start_time = current_time # Start the absence timer

                          # Calculate the duration of the current consecutive absence
                          absence_duration = current_time - self.absence_start_time

                          if absence_duration >= self.absence_threshold:
                              # Absence threshold reached (e.g., 3 seconds) - End the session
                              # The session end time is when the absence started that triggered the end
                              session_end_time = self.absence_start_time
                              duration = session_end_time - self.session_start_time

                              if duration > 0.1: # Only save sessions with meaningful duration
                                  self.save_session(self.session_start_time, session_end_time, duration)

                              # Reset session state
                              self.session_active = False
                              self.session_start_time = None
                              self.current_session_duration = 0 # Reset for display
                              # Update GUI immediately
                              self.session_time_label.config(text="Current Session: 0s")

                              self.absence_start_time = None # Reset absence timer after session ends
                              self.detection_interval = 10.0 # When a session ends, check less frequently (every 10 seconds)
                              print(f"Session ended due to {self.absence_threshold}s absence.")

                          # If absence_duration < self.absence_threshold, the session remains active
                          # The absence_start_time is kept to continue tracking absence duration

                      else: # Session is not active and no person is detected
                          # Keep checking less frequently when no session is active and no person is seen
                          self.detection_interval = 10.0


            # --- Update Current Session Timer Display ---
            # This updates every frame (approx. 30-40 FPS) for smoother display
            if self.session_active:
                # Calculate current duration based on the session start time
                self.current_session_duration = current_time - self.session_start_time
                # Update the session time label in the GUI
                self.session_time_label.config(text=f"Current Session: {self.format_duration(self.current_session_duration)}")
            # If session is not active, the label is reset when the session ends
            
            # Update Total Time Today Display
            # This label should show saved time + current session time if active
            display_total = self.total_time_today # Start with the total from *saved* sessions for today
            if self.session_active:
                # Add the duration of the current *active* session to the saved total
                display_total += self.current_session_duration

            # Update the total time label in the GUI
            self.today_time_label.config(text=f"Total Time Today: {self.format_duration(display_total)}")
            # --- Schedule the next frame update ---
            # Call update_frame again after a short delay to process the next frame
            self.root.after(30, self.update_frame) # Adjust delay for desired frame rate (e.g., 30ms for ~33 FPS)

        else:
            # If ret is False, the camera failed to read a frame (e.g., disconnected)
            print("Error: Failed to read frame from camera.")
            self.stop_tracker() # Stop the tracker gracefully on camera error
            self.status_label.config(text="Status: Camera Read Error", fg="red")


    def on_closing(self):
        # This method is called when the user clicks the window's close button
        print("Closing application...")
        self.stop_tracker() # Perform cleanup (stop camera, save session, close DB)
        self.root.destroy() # Close the Tkinter window
        sys.exit() # Ensure the application process exits cleanly


# --- Main Execution Block ---
if __name__ == "__main__":
    # Create the main Tkinter window
    root = tk.Tk()
    # Create an instance of our PresenceTrackerApp
    app = PresenceTrackerApp(root)
    # Start the Tkinter event loop (this keeps the GUI running)
    root.mainloop()