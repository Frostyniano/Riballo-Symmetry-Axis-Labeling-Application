import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import os
from data_manager import DataManager 
from config import QUESTIONS_PER_SESSION, TUTORIAL_MODE, RESULTS_DIR, SCORE_DIR, PRE_SCALING_GREATER, PRE_SCALING_SMALLER, MIN_DIM_TARGET

class AxisLabelingApp:
    def __init__(self, master):
        self.master = master
        master.title("Riballo Symmetry Axis Labeling Application")
        master.geometry("1000x700") 

        # self.data_load_format and self.load_mode will be defined in start_new_session_or_score_prompt if a session starts
        self.data_manager = None # Initialize to None, it will be created after the user chooses options

        self.session_axes = [] # Axes selected for the current session
        self.current_axis_index = -1 # Index of the current axis in the session
        self.current_image_tk = None # Reference to keep the image in Tkinter (preventing garbage collection)

        self.setup_ui()
        self.bind_keys() # Bind keys for shortcuts (currently empty, but placeholder)
        self.update_navigation_buttons_state() # Update the state of navigation buttons

        # Ensure the window has focus for keyboard input (if bind_keys were active)
        self.master.focus_set() 

        # Show tutorial or main prompt based on configuration
        if TUTORIAL_MODE:
            self.show_tutorial()
        else:
            self.start_new_session_or_score_prompt() 

    def ask_data_load_options(self): 
        """
        Displays dialogs to ask the user for data loading format and mode.
        Returns (data_format, load_mode) or (None, None) if canceled.
        """
        format_response = messagebox.askyesno(
            "Load Data", 
            "Do you want to load axis data from .mat files?\n\n"
            "• If you select 'Yes', data will be loaded from .mat files.\n"
            "• If you select 'No', the application will attempt to load from CSV files."
        )

        if format_response is None: 
            return None, None

        data_format = "mat" if format_response else "csv"
        load_mode = "multiple_files" # Default mode for .mat, and for CSV if not specified further

        if data_format == "csv":
            mode_response = messagebox.askyesno(
                "Load CSV Data",
                "Is all CSV axis data in a SINGLE CSV file?\n\n"
                "• If you select 'Yes', a single CSV file will be searched in the configured folder.\n"
                "• If you select 'No', multiple CSV files (one per image, e.g., refs_001.csv) will be searched in the configured folder."
            )
            if mode_response is None: 
                return None, None
            
            load_mode = "single_file" if mode_response else "multiple_files"
        
        return data_format, load_mode

    def setup_ui(self):
        """Sets up the main user interface elements."""
        # Main frame containing all UI elements
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for displaying the image and the axis
        self.canvas = tk.Canvas(self.main_frame, bg="white", bd=2, relief="groove", highlightthickness=0) 
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame for controls (questions, navigation buttons, etc.)
        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Question 1 and its Radiobuttons
        self.q1_label = tk.Label(self.controls_frame, text="Q1: Is the displayed axis an acceptable symmetry axis for the image?", wraplength=250)
        self.q1_label.pack(pady=10)
        self.q1_var = tk.StringVar(value="None")
        self.q1_yes_radio = tk.Radiobutton(self.controls_frame, text="Yes", variable=self.q1_var, value="Yes", command=self.handle_q1_yes_selected)
        self.q1_yes_radio.pack(anchor=tk.W)
        self.q1_no_radio = tk.Radiobutton(self.controls_frame, text="No", variable=self.q1_var, value="No", command=self.handle_q1_no_selected)
        self.q1_no_radio.pack(anchor=tk.W)

        # Question 2 and its Radiobuttons (initially disabled)
        self.q2_label = tk.Label(self.controls_frame, text="Q2: Is the displayed axis the main symmetry axis in the image?", wraplength=250)
        self.q2_label.pack(pady=10)
        self.q2_var = tk.StringVar(value="None")
        # Both Q2 radiobuttons will call _process_and_move_next to automatically advance
        self.q2_yes_radio = tk.Radiobutton(self.controls_frame, text="Yes", variable=self.q2_var, value="Yes", command=self._process_and_move_next) 
        self.q2_yes_radio.pack(anchor=tk.W)
        self.q2_no_radio = tk.Radiobutton(self.controls_frame, text="No", variable=self.q2_var, value="No", command=self._process_and_move_next) 
        self.q2_no_radio.pack(anchor=tk.W)
        self.disable_q2() # Ensure Q2 is disabled at startup

        # Next Button (also serves as manual advance or finish button)
        self.next_button = tk.Button(self.controls_frame, text="Next Axis", command=self._process_and_move_next) 
        self.next_button.pack(pady=20)

        # Previous Button
        self.prev_button = tk.Button(self.controls_frame, text="Previous Axis", command=self.prev_axis) 
        self.prev_button.pack(pady=5)

        # Progress Indicator Label
        self.progress_label = tk.Label(self.controls_frame, text="Progress: 0/0")
        self.progress_label.pack(pady=10)

        # Finish Session Button (initially disabled)
        self.finish_button = tk.Button(self.controls_frame, text="Finish Session", command=self.confirm_finish_session, state=tk.DISABLED)
        self.finish_button.pack(pady=20)

        # Calculate Score Button (initially hidden)
        self.calculate_score_button = tk.Button(self.controls_frame, text="Calculate Score", command=self.calculate_score_action)
        self.calculate_score_button.pack(pady=10)
        self.calculate_score_button.pack_forget() # Hide by default at startup


    def enable_q2(self):
        """Enables the radio buttons for Question 2."""
        self.q2_yes_radio.config(state=tk.NORMAL)
        self.q2_no_radio.config(state=tk.NORMAL)
        self.master.focus_set() # Ensure main window has focus

    def disable_q2(self):
        """Disables the radio buttons for Question 2 and resets its value."""
        self.q2_var.set("None") 
        self.q2_yes_radio.config(state=tk.DISABLED)
        self.q2_no_radio.config(state=tk.DISABLED)
        self.master.focus_set() 

    def handle_q1_yes_selected(self):
        """Handles 'Yes' selection for Question 1: enables Question 2."""
        self.enable_q2()

    def handle_q1_no_selected(self):
        """Handles 'No' selection for Question 1: sets Q2 to 'No' and advances."""
        self.disable_q2() 
        self.q2_var.set("No") 
        self._process_and_move_next() 

    def bind_keys(self):
        """
        This function is a placeholder. Keyboard shortcuts are not implemented as per requirement.
        If keyboard shortcuts were desired, the 'master.bind_all()' calls would go here.
        """
        pass # No keyboard binding implemented


    def update_navigation_buttons_state(self): 
        """Updates the state (enabled/disabled) of navigation buttons."""
        # 'Previous' Button state
        if self.current_axis_index <= 0:
            self.prev_button.config(state=tk.DISABLED)
        else:
            self.prev_button.config(state=tk.NORMAL)
        
        # 'Next' Button state (and changes text/command for finishing)
        if self.session_axes: # Check if session_axes is initialized before accessing its length
            if self.current_axis_index >= len(self.session_axes) - 1:
                self.next_button.config(text="Finish Session", command=self.finish_session)
            else:
                self.next_button.config(text="Next Axis", command=self._process_and_move_next) 
        else: # If no axes are loaded (e.g., at startup or after finishing)
            self.next_button.config(state=tk.DISABLED, text="Next Axis") 
            self.prev_button.config(state=tk.DISABLED) 

        # 'Finish Session' Button state (enabled if there's an active session)
        if self.session_axes and self.current_axis_index < len(self.session_axes):
            self.finish_button.config(state=tk.NORMAL)
        else:
            self.finish_button.config(state=tk.DISABLED)


    def show_tutorial(self):
        """Displays the tutorial screen."""
        self.main_frame.pack_forget() # Hide the main labeling interface

        self.tutorial_frame = tk.Frame(self.master, bg="lightgray")
        self.tutorial_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tutorial_text = """
        Welcome to the Riballo Symmetry Axis Labeling tutorial!

        In this application, you will be shown images with a proposed symmetry axis. Your task is to answer two questions about each axis:

        Q1: Is the displayed axis an acceptable symmetry axis for the image?
             - Answer 'Yes' if you believe the axis is a valid symmetry.
             - Answer 'No' if it is not.

        Q2: Is the displayed axis the main symmetry axis in the image?
             - This question will only appear if you answer 'Yes' to Q1.
             - If you answer 'No' to Q1, Q2 will automatically be marked 'No' and you will advance, as it is impossible for an axis to be main if it is not a valid symmetry axis.

        Automated response flow:
        - If you answer 'Yes' to Q1, Q2 is enabled. Answering Q2 automatically advances you.
        - If you answer 'No' to Q1, Q2 is marked 'No' and you automatically advance.

        Click 'Got It' to start.
        """
        tk.Label(self.tutorial_frame, text=tutorial_text, justify=tk.LEFT, bg="lightgray", wraplength=900).pack(pady=20)
        tk.Button(self.tutorial_frame, text="Got It", command=self.start_session_from_tutorial).pack(pady=20)

    def start_session_from_tutorial(self):
        """Initiates the labeling session after the tutorial."""
        self.tutorial_frame.destroy() # Destroy the tutorial frame
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) # Show the main interface
        self.start_new_session_or_score_prompt() # Go to the initial prompt to choose between session or score calculation

    def start_new_session_or_score_prompt(self):
        """
        Prompts the user whether to start a new labeling session
        or calculate the score from existing results.
        This is the initial screen after tutorial or app startup.
        """
        # Hide the labeling interface if visible (e.g., if returning from score calculation)
        self.main_frame.pack_forget()

        # Create a new frame for the startup options buttons
        self.start_options_frame = tk.Frame(self.master)
        self.start_options_frame.pack(expand=True)

        start_prompt_text = "Welcome!\n\nWhat would you like to do?"
        self.start_prompt_label = tk.Label(self.start_options_frame, text=start_prompt_text, font=("Arial", 14, "bold"), pady=20)
        self.start_prompt_label.pack(side=tk.TOP, pady=50)

        self.start_session_button_main = tk.Button(self.start_options_frame, text="Start New Labeling Session", command=self.start_new_session, font=("Arial", 12))
        self.start_session_button_main.pack(pady=10)

        self.calculate_score_button_main = tk.Button(self.start_options_frame, text="Calculate Score from Existing Results", command=self.calculate_score_action, font=("Arial", 12))
        self.calculate_score_button_main.pack(pady=10)

        self.master.focus_set() # Ensure main window has focus

    def start_new_session(self):
        """Initializes a new labeling session."""
        # Hide startup options and show the main labeling interface
        self.start_options_frame.pack_forget() 
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) 

        # Ask user for data loading options (MAT/CSV, Single/Multiple)
        data_format, load_mode = self.ask_data_load_options()
        if data_format is None:
            # If user cancels data loading, return to the main startup prompt
            self.main_frame.pack_forget()
            self.start_new_session_or_score_prompt() 
            return

        # Create DataManager instance with chosen loading options for this session
        self.data_manager = DataManager(data_format=data_format, load_mode=load_mode) 

        # Check if any image data was loaded successfully
        if not self.data_manager.all_image_data:
            messagebox.showerror("Loading Error", "Could not load image data. Please check folders and file format.")
            self.main_frame.pack_forget()
            self.start_new_session_or_score_prompt() 
            return

        # Select axes for the session based on configuration
        self.session_axes = self.data_manager.select_axes_for_session(QUESTIONS_PER_SESSION)
        if not self.session_axes:
            messagebox.showinfo("Session End", "No axes available for the session or all axes have been completed.")
            self.finish_session() 
            return
        
        self.current_axis_index = -1 # Initialize to -1 so _process_and_move_next() advances to index 0
        self._process_and_move_next(is_initial_load=True) # Load the first axis
        self.update_navigation_buttons_state() # Update button states at session start

    def _process_and_move_next(self, is_initial_load=False): 
        """
        Processes the current axis's response (if applicable) and moves to the next axis.
        This function centralizes the logic for advancing and saving results.
        """
        # Only record result if not initial load AND current axis is valid
        if not is_initial_load and self.current_axis_index >= 0: 
            q1_answer = self.q1_var.get()
            q2_answer = self.q2_var.get()

            # Ensure all relevant questions are answered before proceeding
            if q1_answer == "None" or (q1_answer == "Yes" and q2_answer == "None"):
                messagebox.showwarning("Attention", "Please answer both questions before continuing.")
                return # Do not advance if answers are incomplete

            current_axis_info = self.session_axes[self.current_axis_index]
            
            # Get full axis data (coords, score) for recording purposes
            full_axis_data_for_record = self.data_manager.get_axis_info(
                current_axis_info['img_base_name'], 
                current_axis_info['axis_row_index']
            )
            score_to_record = full_axis_data_for_record['score'] if full_axis_data_for_record else 'N/A'

            # Record the result for the current axis
            self.data_manager.record_result(
                current_axis_info['img_base_name'], 
                current_axis_info['axis_row_index'], 
                q1_answer,
                q2_answer,
                current_axis_info['expected'], # Use the pre-classified expected_type
                score_to_record 
            )
        
        self.current_axis_index += 1 # Advance to the next axis index

        # Load the next axis or end the session
        if self.current_axis_index < len(self.session_axes):
            self.load_axis_data() # Load and display the current axis
            self.master.focus_set() # Ensure main window has focus after loading new image
        else:
            messagebox.showinfo("Session End", "You have completed all axes for this session. Thank you!")
            self.finish_session() # Automatically finalize the session
        
        self.update_navigation_buttons_state() # Update button states after moving


    def prev_axis(self): 
        """
        Moves back to the previous axis. Does not record answers when moving back.
        """
        if self.current_axis_index <= 0: # If already at the first axis
            messagebox.showinfo("Navigation", "You are at the first axis of the session.")
            return # Do nothing if at the beginning

        self.current_axis_index -= 1 # Move back to the previous axis index
        self.load_axis_data() # Load and display the current axis
        self.master.focus_set() # Ensure main window has focus
        self.update_navigation_buttons_state() # Update button states


    def load_axis_data(self):
        """
        Loads the current axis's image and coordinates, performs scaling, and displays it on the canvas.
        Handles image loading errors by skipping to the next axis.
        """
        # Safety check: ensure session_axes is valid and index is within bounds
        if not self.session_axes or not (0 <= self.current_axis_index < len(self.session_axes)):
            messagebox.showerror("Navigation Error", "Axis index out of range. This should not happen.")
            self.finish_session() # Attempt to finalize to prevent indefinite state
            return False

        current_axis_info = self.session_axes[self.current_axis_index]
        img_base_name = current_axis_info['img_base_name']
        axis_row_index = current_axis_info['axis_row_index']

        axis_data = self.data_manager.get_axis_info(img_base_name, axis_row_index)
        
        if axis_data:
            image_path = axis_data['image_full_path']
            coords = axis_data['coords'] 
            score = axis_data['score'] 

            try:
                original_image = Image.open(image_path)
            except FileNotFoundError:
                messagebox.showerror("Image Error", f"Image not found: {image_path}. Skipping to next axis.")
                # If image fails, attempt to move to the next axis to prevent blocking the app
                self._process_and_move_next(is_initial_load=True) # Simulate initial load to avoid recording invalid response
                return False
            except Exception as e:
                messagebox.showerror("Error loading image", f"Error loading {image_path}: {e}. Skipping to next axis.")
                self._process_and_move_next(is_initial_load=True)
                return False

            img_width_orig, img_height_orig = original_image.size 

            # 1: Pre-scaling to ensure the smallest dimension is MIN_DIM_TARGET if boths are bigger than MIN_DIM_TARGET or one is smaller---
            
            current_width = img_width_orig
            current_height = img_height_orig
            
            pre_scale_factor = 1.0 
            
            min_current_dim = min(current_width, current_height)
            if min_current_dim > MIN_DIM_TARGET and PRE_SCALING_GREATER:
                 pre_scale_factor = MIN_DIM_TARGET / min_current_dim
            elif min_current_dim < MIN_DIM_TARGET and PRE_SCALING_SMALLER:
                 pre_scale_factor = MIN_DIM_TARGET / min_current_dim

            if pre_scale_factor != 1.0: 
                img_pre_scaled = original_image.resize(
                    (int(img_width_orig * pre_scale_factor), int(img_height_orig * pre_scale_factor)), 
                    Image.LANCZOS
                )
            else:
                img_pre_scaled = original_image 
            
            current_img_width, current_img_height = img_pre_scaled.size


            # 2: Final scaling to fit into the canvas ---
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width == 1 or canvas_height == 1:
                master_width = self.master.winfo_width()
                master_height = self.master.winfo_height()
                if master_width > 1 and master_height > 1:
                    canvas_width = master_width * 0.8
                    canvas_height = master_height * 0.6
                else: 
                    canvas_width = 800
                    canvas_height = 500

            final_scale_ratio = min(canvas_width / current_img_width, canvas_height / current_img_height)
            
            new_width_img_on_canvas = int(current_img_width * final_scale_ratio)
            new_height_img_on_canvas = int(current_img_height * final_scale_ratio)
            
            img_display = img_pre_scaled.resize((new_width_img_on_canvas, new_height_img_on_canvas), Image.LANCZOS)
            
            draw = ImageDraw.Draw(img_display)
            line_color = "red" 
            line_width = 3

            x1_orig, y1_orig, x2_orig, y2_orig = coords

            # 3. Apply final_scale_ratio
            x1_final_scaled = x1_orig * final_scale_ratio
            y1_final_scaled = y1_orig * final_scale_ratio
            x2_final_scaled = x2_orig * final_scale_ratio
            y2_final_scaled = y2_orig * final_scale_ratio
            
            # --- DEBUGGING LINES (commented by default) ---
            # print(f"DEBUG: Original Image: {img_width_orig}x{img_height_orig}")
            # print(f"DEBUG: Pre-scaling Factor: {pre_scale_factor:.4f}")
            # print(f"DEBUG: Image after pre-scaling: {current_img_width}x{current_img_height}")
            # print(f"DEBUG: Canvas Dimensions: {canvas_width}x{canvas_height}")
            # print(f"DEBUG: Final Scale Ratio: {final_scale_ratio:.4f}")
            # print(f"DEBUG: Image on Canvas (final resized): {new_width_img_on_canvas}x{new_height_img_on_canvas}")
            # print(f"DEBUG: Original Axis Coords: ({x1_orig:.2f}, {y1_orig:.2f}) -> ({x2_orig:.2f}, {y2_orig:.2f})")
            # print(f"DEBUG: Inverted & Final Scaled Axis Coords: ({x1_final_scaled:.2f}, {y1_final_scaled:.2f}) -> ({x2_final_scaled:.2f}, {y2_final_scaled:.2f})")
            # print(f"DEBUG: Axis Score: {score:.1f}") 
    
            # DRAW THE MAIN AXIS LINE
            draw.line((x1_final_scaled, y1_final_scaled, x2_final_scaled, y2_final_scaled), fill=line_color, width=line_width)
            
            self.current_image_tk = ImageTk.PhotoImage(img_display) 

            center_x_canvas = canvas_width / 2
            center_y_canvas = canvas_height / 2
            
            self.canvas.create_image(center_x_canvas, center_y_canvas, anchor=tk.CENTER, image=self.current_image_tk)

            self.update_progress_label() 
            self.q1_var.set("None") 
            self.q2_var.set("None")
            self.disable_q2()
            
            return True
        return False

    def update_progress_label(self):
        """Updates the text of the progress indicator."""
        self.progress_label.config(text=f"Progress: {self.current_axis_index + 1}/{len(self.session_axes)}")


    def confirm_finish_session(self):
        """Confirms with the user before ending the session."""
        if messagebox.askyesno("End Session", "Are you sure you want to end the session and save the results?"):
            self.finish_session()

    def finish_session(self):
        """Saves the final results and closes the application."""
        # Attempt to save the answer for the last axis if not already done by advancing
        if self.current_axis_index >= 0 and self.current_axis_index < len(self.session_axes):
            q1_answer = self.q1_var.get()
            q2_answer = self.q2_var.get()

            # Only save if there's a complete answer for the last axis
            if q1_answer != "None" and (q1_answer == "No" or q2_answer != "None"): 
                current_axis_info = self.session_axes[self.current_axis_index]
                img_base_name = current_axis_info['img_base_name']
                axis_row_index = current_axis_info['axis_row_index']
                expected_type = self.data_manager._classify_axis_by_score(
                    axis_row_index, 
                    self.data_manager.get_axis_info(img_base_name, axis_row_index)['score']
                )
                self.data_manager.record_result(
                    img_base_name,
                    axis_row_index,
                    q1_answer,
                    q2_answer,
                    expected_type,
                    self.data_manager.get_axis_info(img_base_name, axis_row_index)['score']
                )

        self.data_manager.save_session_results()
        
        if messagebox.askyesno("Session Finished", "Results saved.\n\nDo you want to calculate the PA/NA score now?"):
            self.master.after(100, self.calculate_score_action) 
        else:
            messagebox.showinfo("Application", "The application will close.")
            self.master.destroy()

    def calculate_score_action(self):
        """
        Manages the process of calculating PA/NA scores from result files.
        Called from the startup button or after completing a session.
        """
        # Hide the labeling interface if visible
        self.main_frame.pack_forget()

        # Show startup buttons (to allow starting a new session or calculating again)
        self.start_options_frame.pack(expand=True) 
        
        # Get all CSV result files
        result_files = [f for f in os.listdir(RESULTS_DIR) if os.path.isfile(os.path.join(RESULTS_DIR, f)) and f.endswith('.csv')]
        result_file_paths = [os.path.join(RESULTS_DIR, f) for f in result_files]

        if len(result_file_paths) < 2:
            messagebox.showinfo("Calculate Score", "At least 2 CSV result files are needed in the 'results/' folder to calculate PA/NA.")
            return

        use_all = messagebox.askyesno("Calculate Score", 
                                     f"Found {len(result_file_paths)} result files.\n"
                                     "Do you want to use ALL these files to calculate the PA/NA score?\n"
                                     "(If there are different configurations, the calculation will fail.)")
        if not use_all:
            messagebox.showinfo("Calculate Score", "Calculation canceled.")
            return

        # It is crucial that self.data_manager exists to call calculate_pa_na
        # If the app starts with the calculate button, data_manager would be None
        if self.data_manager is None:
            # Create a basic DataManager instance. The loading parameters here are placeholders
            # as it will not load image data in this context, only use its PA/NA calculation methods.
            self.data_manager = DataManager(data_format="mat", load_mode="multiple_files") 

        pa_score, na_score, num_results_used = self.data_manager.calculate_pa_na(result_file_paths)

        if pa_score is not None and na_score is not None:
            self.data_manager.save_score_results(pa_score, na_score, num_results_used) # <--- Call to save_score_results
            messagebox.showinfo(
                "PA/NA Score",
                f"Score Calculation Complete:\n"
                f"Results used: {num_results_used}\n"
                f"PA: {pa_score:.4f}\n"
                f"NA: {na_score:.4f}\n\n"
                f"Results have been saved to '{os.path.join(SCORE_DIR, 'score.csv')}'"
            )
        else:
            messagebox.showerror("Calculation Error", "Could not calculate PA/NA score. Check console messages.")