import random
from datetime import datetime
import os
import scipy.io
import numpy as np
import csv
from itertools import combinations

# Import configuration variables from config.py
from config import (
    N_IMAGES, THRESHOLD_NEAR_ONE, THRESHOLD_FAR_FROM_ONE,
    RESULTS_DIR, IMAGES_DIR, MAT_FILES_DIR, CSV_INPUT_DIR, SCORE_DIR,
    TARGET_YY_PERCENTAGE, TARGET_YN_PERCENTAGE, QUESTIONS_PER_SESSION
)

class DataManager:
    """
    Manages data loading, axis selection logic, and saving/calculating results.
    """
    def __init__(self, data_format="mat", load_mode="multiple_files"):
        """
        Initializes the DataManager with specified data loading format and mode.
        Args:
            data_format (str): 'mat' or 'csv'
            load_mode (str): 'single_file' or 'multiple_files' (relevant for CSV)
        """
        self.data_format = data_format
        self.load_mode = load_mode
        self.all_image_data = self._load_all_image_data() # Loads all axis data into memory
        self.selected_axes_for_session = [] # List of axes selected for the current labeling session
        self.current_session_results = [] # Stores results of the current session before saving

    def _load_all_image_data(self):
        """
        Dispatches to the appropriate data loading function based on
        the chosen data_format and load_mode.
        """
        if self.data_format == "mat":
            # For .mat, only the 'multiple_files' mode is implemented as it's typical
            # for the 'Out_f6_ap25_refs_XXX.mat' naming convention (one .mat per image).
            print("Loading data from .mat files (multiple files per image).")
            return self._load_all_image_data_mat_multiple_files()
        elif self.data_format == "csv":
            if self.load_mode == "single_file":
                print("Loading data from a single CSV file.")
                return self._load_all_image_data_csv_single_file()
            elif self.load_mode == "multiple_files":
                print("Loading data from multiple CSV files (one per image).")
                return self._load_all_image_data_csv_multiple_files()
            else:
                print("Error: Unrecognized CSV load mode. Using single file by default.")
                return self._load_all_image_data_csv_single_file()
        else:
            print("Error: Unrecognized data format. Using .mat by default.")
            return self._load_all_image_data_mat_multiple_files()


    def _load_all_image_data_mat_multiple_files(self):
        """
        Loads symmetry axis data from multiple .mat files (one per image).
        Expects .mat files in MAT_FILES_DIR named 'Out_f6_ap25_refs_XXX.mat'.
        Returns a dictionary: { "refs_001": {"image_full_path": "...", "axes_data": np.array([...])}, ... }
        """
        all_data = {}
        mat_files_full_path = MAT_FILES_DIR

        if not os.path.exists(mat_files_full_path):
            print(f"Error: The .mat file folder was not found at '{mat_files_full_path}'.")
            print("Please ensure your .mat files are in 'your_application/data/mat_files/'.")
            return {}

        # Iterate through expected image base names (e.g., refs_001 to refs_096)
        for i in range(1, N_IMAGES + 1):
            img_base_name = f"refs_{i:03d}" # e.g., 'refs_001'
            img_file_name = f"{img_base_name}.jpg" # e.g., 'refs_001.jpg'
            mat_file_name_suffix = f"{img_base_name}.mat" # e.g., 'refs_001.mat' (for searching)

            image_path = os.path.join(IMAGES_DIR, img_file_name) # Full path to the image file

            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image '{image_path}' not found. Skipping {img_base_name}.")
                continue

            # Search for the .mat file with the specific suffix (e.g., handles 'Out_f6_ap25_')
            found_mat_file_path = None
            for filename in os.listdir(mat_files_full_path):
                if filename.endswith(mat_file_name_suffix): # Checks if 'Out_f6_ap25_refs_001.mat' ends with 'refs_001.mat'
                    found_mat_file_path = os.path.join(mat_files_full_path, filename)
                    break # Found it, stop searching

            if not found_mat_file_path:
                print(f"Warning: No .mat file ending with '{mat_file_name_suffix}' found in '{mat_files_full_path}'. Skipping {img_base_name}.")
                continue

            try:
                mat_contents = scipy.io.loadmat(found_mat_file_path) # Load .mat file content

                # Verify that 'img_detected_refs' variable exists and is a NumPy array
                if 'img_detected_refs' in mat_contents and isinstance(mat_contents['img_detected_refs'], np.ndarray):
                    axes_data = mat_contents['img_detected_refs'] # This is the nx5 matrix of axes data

                    # Ensure the score of the first axis (index 0) is 1.0, as per specification
                    if axes_data.shape[0] > 0: # If there's at least one axis
                        if axes_data[0, 4] != 1.0: # Check the 5th column (index 4) of the first row
                            axes_data[0, 4] = 1.0 # Force it to 1.0

                    # Store the loaded data
                    all_data[img_base_name] = {
                        "image_full_path": image_path, # Full path to the image for display
                        "axes_data": axes_data # The nx5 matrix with coordinates and scores
                    }
                else:
                    print(f"Warning: 'img_detected_refs' not found or not a valid array in {found_mat_file_path}. Skipping {img_base_name}.")
            except Exception as e:
                print(f"Error loading {found_mat_file_path}: {e}. Skipping {img_base_name}.")

        return all_data

    def _load_all_image_data_csv_single_file(self):
        """
        Loads symmetry axis data from a single CSV file.
        Expects one CSV file in CSV_INPUT_DIR named e.g., 'axes_input.csv'.
        CSV format: 'image_base_name,axis_row_index,x1,y1,x2,y2,score'
        """
        all_data = {}
        csv_input_full_path = CSV_INPUT_DIR

        if not os.path.exists(csv_input_full_path):
            print(f"Error: The input CSV file folder was not found at '{csv_input_full_path}'.")
            return {}

        csv_files = [f for f in os.listdir(csv_input_full_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"Error: No CSV files found in '{csv_input_full_path}'.")
            return {}

        input_csv_file = os.path.join(csv_input_full_path, csv_files[0]) # Takes the first CSV found
        print(f"Loading data from CSV: {input_csv_file}")

        # Temporary storage: { 'image_base_name': { axis_row_index: [x1,y1,x2,y2,score] } }
        temp_image_axes_by_index = {}

        try:
            with open(input_csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                required_cols = ['image_base_name', 'axis_row_index', 'x1', 'y1', 'x2', 'y2', 'score']
                if not all(col in reader.fieldnames for col in required_cols):
                    print(f"Error: CSV '{input_csv_file}' does not contain all required columns: {required_cols}. Found columns: {reader.fieldnames}")
                    return {}

                for row in reader:
                    try:
                        img_base_name = row['image_base_name']
                        axis_row_index = int(row['axis_row_index'])
                        x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
                        score = float(row['score'])

                        if img_base_name not in temp_image_axes_by_index:
                            temp_image_axes_by_index[img_base_name] = {}

                        temp_image_axes_by_index[img_base_name][axis_row_index] = [x1, y1, x2, y2, score]

                    except ValueError as ve:
                        print(f"Data format error in CSV row: {row}. Details: {ve}. Skipping row.")
                        continue

            # Convert the temporary dictionary to the final all_data structure (NumPy arrays)
            for img_base_name, axes_dict in temp_image_axes_by_index.items():
                if not axes_dict:
                    print(f"Warning: No axes found for image {img_base_name} in CSV. Skipping.")
                    continue

                max_idx = max(axes_dict.keys())
                axes_matrix = np.zeros((max_idx + 1, 5)) # Initialize matrix with zeros based on max index

                for idx, axis_data_list in axes_dict.items():
                    if 0 <= idx <= max_idx: # Ensure index is within the created matrix bounds
                        axes_matrix[idx, :] = axis_data_list
                    else:
                        print(f"Warning: axis_row_index {idx} out of range for image {img_base_name}. Ignoring.")

                # Ensure the first axis score is 1.0, as per specification
                if axes_matrix.shape[0] > 0:
                    if axes_matrix[0, 4] != 1.0:
                        axes_matrix[0, 4] = 1.0

                # Check if the associated image file exists
                image_full_path = os.path.join(IMAGES_DIR, f"{img_base_name}.jpg") # Assumes .jpg for images
                if not os.path.exists(image_full_path):
                    print(f"Warning: Associated image '{image_full_path}' not found. Skipping {img_base_name}.")
                    continue

                all_data[img_base_name] = {
                    "image_full_path": image_full_path,
                    "axes_data": axes_matrix
                }

            return all_data

        except FileNotFoundError:
            print(f"Error: Input CSV file not found at '{input_csv_file}'.")
            return {}
        except Exception as e:
            print(f"General error loading from CSV: {e}")
            return {}

    def _load_all_image_data_csv_multiple_files(self):
        """
        Loads symmetry axis data from multiple CSV files (one per image).
        Expects CSV files in CSV_INPUT_DIR named 'refs_XXX.csv'.
        CSV format: 'axis_row_index,x1,y1,x2,y2,score' (no 'image_base_name' column).
        """
        all_data = {}
        csv_input_full_path = CSV_INPUT_DIR

        if not os.path.exists(csv_input_full_path):
            print(f"Error: The input CSV file folder was not found at '{csv_input_full_path}'.")
            return {}

        csv_files = [f for f in os.listdir(csv_input_full_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"Error: No CSV files found in '{csv_input_full_path}'.")
            return {}

        required_cols = ['axis_row_index', 'x1', 'y1', 'x2', 'y2', 'score']

        for csv_file_name in csv_files:
            # Check for expected naming convention (e.g., refs_001.csv)
            if not csv_file_name.startswith("refs_") or not csv_file_name.endswith(".csv"):
                print(f"Warning: Unexpected CSV file name: {csv_file_name}. Ignoring.")
                continue

            img_base_name = csv_file_name.replace(".csv", "") # Extract base name (e.g., 'refs_001')
            csv_file_path = os.path.join(csv_input_full_path, csv_file_name)
            image_full_path = os.path.join(IMAGES_DIR, f"{img_base_name}.jpg") # Assumes .jpg for images

            # Check if the associated image file exists
            if not os.path.exists(image_full_path):
                print(f"Warning: Associated image '{image_full_path}' not found for {csv_file_name}. Skipping.")
                continue

            temp_axes_for_image = {} # { axis_row_index: [x1,y1,x2,y2,score] }

            try:
                with open(csv_file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)

                    if not all(col in reader.fieldnames for col in required_cols):
                        print(f"Error: CSV '{csv_file_name}' does not contain all required columns: {required_cols}. Found columns: {reader.fieldnames}. Skipping.")
                        continue

                    for row in reader:
                        try:
                            axis_row_index = int(row['axis_row_index'])
                            x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
                            score = float(row['score'])
                            temp_axes_for_image[axis_row_index] = [x1, y1, x2, y2, score]
                        except ValueError as ve:
                            print(f"Data format error in row of {csv_file_name}: {row}. Details: {ve}. Skipping row.")
                            continue

                if not temp_axes_for_image:
                    print(f"Warning: No valid axes found in {csv_file_name}. Skipping this image.")
                    continue

                max_idx = max(temp_axes_for_image.keys())
                axes_matrix = np.zeros((max_idx + 1, 5)) # Initialize matrix with zeros

                for idx, axis_data_list in temp_axes_for_image.items():
                    if 0 <= idx <= max_idx: # Ensure index is within matrix bounds
                        axes_matrix[idx, :] = axis_data_list
                    else:
                        print(f"Warning: axis_row_index {idx} out of range for {csv_file_name}. Ignoring.")

                # Ensure the first axis score is 1.0, as per specification
                if axes_matrix.shape[0] > 0:
                    if axes_matrix[0, 4] != 1.0:
                        axes_matrix[0, 4] = 1.0

                all_data[img_base_name] = {
                    "image_full_path": image_full_path,
                    "axes_data": axes_matrix
                }

            except FileNotFoundError:
                print(f"Error: CSV file '{csv_file_path}' not found. Skipping.")
            except Exception as e:
                print(f"Error loading from {csv_file_name}: {e}. Skipping.")

        return all_data

    def _classify_axis_by_score(self, axis_row_index, score, total_axes_in_image=1):
        """
        Helper function to classify an axis based on its score and index.
        Rule: Anything not classified as YY or YN, is NN.
        Args:
            axis_row_index (int): The 0-indexed row of the axis in the matrix.
            score (float): The confidence score of the axis.
            total_axes_in_image (int): Total number of axes detected for the image.
        Returns:
            str: 'YY', 'YN', or 'NN'.
        """
        if axis_row_index == 0: # The first axis (index 0) is always YY
            return 'YY'

        # YN can only be the second axis (index 1) if it meets the score condition
        if axis_row_index == 1 and THRESHOLD_NEAR_ONE <= score < 1.0:
            return 'YN'

        # Everything else (any other axis with index > 1, or the second axis if it doesn't qualify for YN) is NN
        return 'NN'

    def select_axes_for_session(self, num_axes_to_select):
        """
        Selects axes for the session, attempting to balance types (YY, YN, NN)
        using the loaded data.
        YN is exclusively for the second axis if it meets score conditions.
        All axes are classified as YY, YN, or NN.
        Args:
            num_axes_to_select (int): The desired number of axes for the session.
        Returns:
            list: A list of dictionaries, each representing a selected axis for the session.
        """
        self.selected_axes_for_session = []
        all_image_base_names = list(self.all_image_data.keys())
        random.shuffle(all_image_base_names) # Shuffle images for random selection

        # Calculate target counts for each axis type based on percentages
        target_yy = int(num_axes_to_select * TARGET_YY_PERCENTAGE)
        target_yn = int(num_axes_to_select * TARGET_YN_PERCENTAGE)
        target_nn = max(0, num_axes_to_select - target_yy - target_yn) # Ensure non-negative

        yy_count = 0 # Current count of selected YY axes
        yn_count = 0 # Current count of selected YN axes
        nn_count = 0 # Current count of selected NN axes

        for img_base_name in all_image_base_names:
            if len(self.selected_axes_for_session) >= num_axes_to_select:
                break # Stop if enough axes are selected for the session

            img_data_obj = self.all_image_data.get(img_base_name)
            # Skip if image data is missing or has no axes
            if not img_data_obj or img_data_obj.get('axes_data') is None or img_data_obj['axes_data'].shape[0] == 0:
                continue

            axes_matrix = img_data_obj['axes_data']
            scores = axes_matrix[:, 4] # Extract scores column
            total_axes_in_image = axes_matrix.shape[0]

            potential_axes = [] # List of axis candidates for the current image

            # Add YY candidate (first axis) if still needed for the global target
            if target_yy > 0:
                potential_axes.append({'img_base_name': img_base_name, 'axis_row_index': 0, 'expected': 'YY'})

            # Add YN candidate (second axis) if available and needed
            if total_axes_in_image > 1:
                classified_type_for_a2 = self._classify_axis_by_score(1, scores[1], total_axes_in_image)
                if classified_type_for_a2 == 'YN' and target_yn > 0:
                    potential_axes.append({'img_base_name': img_base_name, 'axis_row_index': 1, 'expected': 'YN'})

            # Add NN candidates (any axis not already classified as YY/YN) if needed
            if target_nn > 0 and total_axes_in_image > 1:
                # Iterate from the second axis onwards (index 1)
                for axis_idx_in_matrix in range(1, total_axes_in_image):
                    score_current_axis = scores[axis_idx_in_matrix]
                    classified_type = self._classify_axis_by_score(axis_idx_in_matrix, score_current_axis, total_axes_in_image)

                    if classified_type == 'NN':
                        # Avoid adding the same axis multiple times if it was already considered
                        if not any(a['axis_row_index'] == axis_idx_in_matrix and a['expected'] == 'NN' for a in potential_axes):
                            potential_axes.append({'img_base_name': img_base_name, 'axis_row_index': axis_idx_in_matrix, 'expected': 'NN'})

            random.shuffle(potential_axes) # Shuffle candidates for the current image

            # Fill the session's selected_axes_for_session based on target counts
            for axis_info in potential_axes:
                if len(self.selected_axes_for_session) >= num_axes_to_select:
                    break # Stop if session is full

                # Avoid adding duplicates from previous image selections
                if any(a['img_base_name'] == axis_info['img_base_name'] and a['axis_row_index'] == axis_info['axis_row_index'] for a in self.selected_axes_for_session):
                    continue

                # Add if current type count is below target
                if axis_info['expected'] == 'YY' and yy_count < target_yy:
                    self.selected_axes_for_session.append(axis_info)
                    yy_count += 1
                elif axis_info['expected'] == 'YN' and yn_count < target_yn:
                    self.selected_axes_for_session.append(axis_info)
                    yn_count += 1
                elif axis_info['expected'] == 'NN' and nn_count < target_nn:
                    self.selected_axes_for_session.append(axis_info)
                    nn_count += 1

        # Fallback: If not enough axes were selected based on targets, fill with any available.
        # All axes will be classified as YY, YN, or NN.
        if len(self.selected_axes_for_session) < num_axes_to_select:
            all_possible_axes_for_fill = []
            for img_base_name in all_image_base_names:
                img_data_obj = self.all_image_data.get(img_base_name)
                if img_data_obj and img_data_obj.get('axes_data') is not None and img_data_obj['axes_data'].shape[0] > 0:
                    axes_matrix = img_data_obj['axes_data']
                    scores = axes_matrix[:, 4]
                    total_axes_in_image_fill = axes_matrix.shape[0]

                    for i in range(total_axes_in_image_fill):
                        # Classify the axis for filling using the same logic (always YY/YN/NN)
                        true_expected_type = self._classify_axis_by_score(i, scores[i], total_axes_in_image_fill)
                        candidate = {'img_base_name': img_base_name, 'axis_row_index': i, 'expected': true_expected_type}

                        # Add only if not already in the list of selected axes
                        if not any(a['img_base_name'] == candidate['img_base_name'] and a['axis_row_index'] == candidate['axis_row_index'] for a in self.selected_axes_for_session):
                            all_possible_axes_for_fill.append(candidate)

            random.shuffle(all_possible_axes_for_fill)
            for axis_info in all_possible_axes_for_fill:
                if len(self.selected_axes_for_session) >= num_axes_to_select:
                    break # Stop if session is full after filling
                self.selected_axes_for_session.append(axis_info)

        random.shuffle(self.selected_axes_for_session) # Final shuffle of the entire selected session
        return self.selected_axes_for_session

    def get_axis_info(self, img_base_name, axis_row_index):
        """
        Retrieves the coordinates and score of a specific axis for a given image.
        Args:
            img_base_name (str): Base name of the image (e.g., 'refs_001').
            axis_row_index (int): The 0-indexed row of the axis in the image's data matrix.
        Returns:
            dict: A dictionary containing 'image_full_path', 'coords', and 'score', or None if not found.
        """
        img_data_obj = self.all_image_data.get(img_base_name)
        if img_data_obj and img_data_obj.get('axes_data') is not None:
            axes_matrix = img_data_obj['axes_data']
            if 0 <= axis_row_index < axes_matrix.shape[0]:
                axis_coords = axes_matrix[axis_row_index, :4].tolist() # Extract x1, y1, x2, y2 as list
                score = float(axes_matrix[axis_row_index, 4]) # Extract score as float
                return {
                    'image_full_path': img_data_obj['image_full_path'],
                    'coords': axis_coords,
                    'score': score
                }
        return None

    def record_result(self, img_base_name, axis_row_index, q1_answer, q2_answer, expected_type, score):
        """
        Records or updates the user's answer for a specific axis.
        If an answer for this axis already exists in the current session, it is overwritten.
        Args:
            img_base_name (str): Base name of the image.
            axis_row_index (int): Index of the axis.
            q1_answer (str): User's answer for Q1 ('Yes'/'No'/'None').
            q2_answer (str): User's answer for Q2 ('Yes'/'No'/'None').
            expected_type (str): The pre-classified expected type of the axis ('YY'/'YN'/'NN').
            score (float/str): The confidence score of the axis.
        """
        timestamp = datetime.now().isoformat() # Current timestamp for the record

        # Format score to 1 decimal place string for saving in CSV
        formatted_score = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)

        new_result_data = {
            'timestamp': timestamp,
            'image_base_name': img_base_name,
            'axis_row_index': axis_row_index,
            'score': formatted_score,
            'q1_answer': q1_answer,
            'q2_answer': q2_answer,
            'expected_type': expected_type
        }

        found_existing = False
        # Search for an existing record for the same axis in the current session
        for i, existing_result in enumerate(self.current_session_results):
            if (existing_result['image_base_name'] == img_base_name and
                existing_result['axis_row_index'] == axis_row_index):
                self.current_session_results[i] = new_result_data # Overwrite the existing record
                found_existing = True
                break

        if not found_existing:
            self.current_session_results.append(new_result_data) # Append as a new record if not found

    def save_session_results(self):
        """
        Saves the results of the current labeling session to a CSV file.
        The filename includes configuration details for later validation.
        """
        if not self.current_session_results:
            return # Do nothing if no results to save

        os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure the results directory exists

        # Construct configuration string to embed in the filename
        # This allows checking consistency when calculating PA/NA
        config_str = f"Q{QUESTIONS_PER_SESSION}_TYY{int(TARGET_YY_PERCENTAGE*100)}_TYN{int(TARGET_YN_PERCENTAGE*100)}_TN{int(THRESHOLD_NEAR_ONE*100)}_TF{int(THRESHOLD_FAR_FROM_ONE*100)}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULTS_DIR, f"session_results_{timestamp}_{config_str}.csv") # Full path for the output CSV

        # Define the header row for the CSV
        fieldnames = [
            'timestamp',
            'image_base_name',
            'axis_row_index',
            'score',
            'q1_answer',
            'q2_answer',
            'expected_type'
        ]

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader() # Write the header row
                writer.writerows(self.current_session_results) # Write all collected results

            print(f"Session results saved to: {filename}")
            self.current_session_results = [] # Clear results after saving
        except Exception as e:
            print(f"Error saving results in CSV format: {e}")

    # --- NEW FUNCTIONS FOR PA/NA CALCULATION ---

    def _load_single_result_csv(self, file_path):
        """
        Loads a single CSV result file from a user's session.
        Args:
            file_path (str): The full path to the CSV file.
        Returns:
            list: A list of dictionaries, where each dict is a row from the CSV, or None on error.
        """
        results = []
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    results.append(row)
            return results
        except Exception as e:
            print(f"Error loading result file '{file_path}': {e}")
            return None

    def _extract_config_from_filename(self, filename):
        """
        Extracts the configuration string from a result CSV filename.
        Example filename: 'session_results_YYYYMMDD_HHMMSS_Q50_TYY40_TYN30_TN80_TF30.csv'
        It extracts 'Q50_TYY40_TYN30_TN80_TF30'.
        Args:
            filename (str): The base name of the file (e.g., 'session_results_...csv').
        Returns:
            str: The extracted configuration string, or an empty string if not found.
        """
        parts = filename.split('_')
        for part in parts:
            if part.startswith("Q"): # Look for the part starting with 'Q'
                # Assuming the full config string starts with 'Q' and ends before '.csv'
                config_str_start_index = parts.index(part) # Get the index where 'Q' part begins
                full_config_str = "_".join(parts[config_str_start_index:]) # Join parts from 'Q' onwards
                if full_config_str.endswith(".csv"): # Remove .csv extension if present
                    full_config_str = full_config_str[:-4]
                return full_config_str
        return "" # Return empty string if no config part found

    def _check_config_consistency(self, result_files_paths):
        """
        Checks if all selected result files share the same configuration.
        This is crucial for meaningful PA/NA calculation across different users/sessions.
        Args:
            result_files_paths (list): List of full paths to result CSV files.
        Returns:
            str: The common configuration string if consistent, or None if inconsistent or no files.
        """
        if not result_files_paths:
            return None

        # Extract config from the first file to use as reference
        first_config_str = self._extract_config_from_filename(os.path.basename(result_files_paths[0]))
        if not first_config_str:
            print(f"Warning: File '{os.path.basename(result_files_paths[0])}' does not have config in its name. Consistency cannot be verified.")
            return None

        # Compare against all other files
        for file_path in result_files_paths:
            current_config_str = self._extract_config_from_filename(os.path.basename(file_path))
            if current_config_str != first_config_str:
                print(f"Consistency error: File '{os.path.basename(file_path)}' has config '{current_config_str}' different from '{first_config_str}'.")
                return None # Inconsistency detected

        return first_config_str # Return the common config string if all are consistent

    def calculate_pa_na(self, result_files_paths):
        """
        Calculates Positive Agreement (PA) and Negative Agreement (NA) scores
        based on the provided result files.
        PA/NA is calculated as specified in the 'D. Perceptual User Tests on a New Database' section
        of the associated scientific paper.
        Args:
            result_files_paths (list): List of full paths to result CSV files (each representing one person's results).
        Returns:
            tuple: (pa_score, na_score, num_results_used) or (None, None, 0) if calculation fails.
        """
        if not result_files_paths or len(result_files_paths) < 2:
            print("At least 2 result files are needed to calculate PA/NA.")
            return None, None, 0

        # 1. Check configuration consistency of files
        common_config = self._check_config_consistency(result_files_paths)
        if common_config is None:
            print("Cannot calculate PA/NA due to inconsistencies in result file configuration.")
            return None, None, 0

        print(f"Calculating PA/NA for {len(result_files_paths)} files with configuration: {common_config}")

        # 2. Load all results and pool them by unique axis (image_base_name, axis_row_index)
        # pooled_results: { ('img_base_name', axis_row_index): { 'person_idx': {'q1_answer': 'Yes', 'q2_answer': 'No'} } }
        pooled_results = {}

        for person_idx, file_path in enumerate(result_files_paths):
            person_results = self._load_single_result_csv(file_path)
            if person_results is None:
                print(f"Error loading results for person {person_idx} from '{file_path}'. Skipping this person's data.")
                # Decide if to skip person or abort. For robustness, we skip this person for calculation if their file fails.
                continue

            for row in person_results:
                try:
                    # Use (image_base_name, axis_row_index) as a unique key for each axis
                    img_key = (row['image_base_name'], int(row['axis_row_index']))
                    if img_key not in pooled_results:
                        pooled_results[img_key] = {}

                    # Store Q1 and Q2 answers for this person and axis
                    pooled_results[img_key][person_idx] = {
                        'q1_answer': row['q1_answer'],
                        'q2_answer': row['q2_answer']
                    }
                except KeyError as ke:
                    print(f"Error: Column '{ke}' not found or incorrect format in result file '{file_path}'. Skipping row.")
                    continue
                except ValueError as ve:
                    print(f"Value error in result file '{file_path}': {ve}. Skipping row.")
                    continue

        # 3. Calculate YY, YN, NY, NN pairs for Q1 and Q2
        # These counts are totals across all axes and all unique pairs of annotators.
        total_yy_q1 = 0
        total_yn_q1 = 0
        total_ny_q1 = 0
        total_nn_q1 = 0

        total_yy_q2 = 0
        total_yn_q2 = 0
        total_ny_q2 = 0
        total_nn_q2 = 0

        # Iterate over each unique axis and its answers from all annotators
        for img_key, answers_by_person in pooled_results.items():
            # Only consider axes for which at least two annotators have provided answers
            if len(answers_by_person) < 2:
                continue

            # Generate all unique combinations of pairs of annotators for this axis
            for p1_idx, p2_idx in combinations(answers_by_person.keys(), 2):
                ans1 = answers_by_person[p1_idx]
                ans2 = answers_by_person[p2_idx]

                # Tally for Q1
                if ans1['q1_answer'] == 'Yes' and ans2['q1_answer'] == 'Yes':
                    total_yy_q1 += 1
                elif ans1['q1_answer'] == 'Yes' and ans2['q1_answer'] == 'No':
                    total_yn_q1 += 1
                elif ans1['q1_answer'] == 'No' and ans2['q1_answer'] == 'Yes':
                    total_ny_q1 += 1
                elif ans1['q1_answer'] == 'No' and ans2['q1_answer'] == 'No':
                    total_nn_q1 += 1

                # Tally for Q2 (only if both answers for Q2 are valid 'Yes' or 'No')
                if ans1['q2_answer'] in ['Yes', 'No'] and ans2['q2_answer'] in ['Yes', 'No']:
                    if ans1['q2_answer'] == 'Yes' and ans2['q2_answer'] == 'Yes':
                        total_yy_q2 += 1
                    elif ans1['q2_answer'] == 'Yes' and ans2['q2_answer'] == 'No':
                        total_yn_q2 += 1
                    elif ans1['q2_answer'] == 'No' and ans2['q2_answer'] == 'Yes':
                        total_ny_q2 += 1
                    elif ans1['q2_answer'] == 'No' and ans2['q2_answer'] == 'No':
                        total_nn_q2 += 1

        # 4. Calculate PA and NA scores for Q1 and Q2
        pa_q1 = 0.0
        na_q1 = 0.0
        denominator_q1 = (2 * total_yy_q1 + total_yn_q1 + total_ny_q1)
        if denominator_q1 > 0:
            pa_q1 = (2 * total_yy_q1) / denominator_q1

        denominator_q1_na = (2 * total_nn_q1 + total_yn_q1 + total_ny_q1)
        if denominator_q1_na > 0:
            na_q1 = (2 * total_nn_q1) / denominator_q1_na

        pa_q2 = 0.0
        na_q2 = 0.0
        denominator_q2 = (2 * total_yy_q2 + total_yn_q2 + total_ny_q2)
        if denominator_q2 > 0:
            pa_q2 = (2 * total_yy_q2) / denominator_q2

        denominator_q2_na = (2 * total_nn_q2 + total_yn_q2 + total_ny_q2)
        if denominator_q2_na > 0:
            na_q2 = (2 * total_nn_q2) / denominator_q2_na

        # 5. Return the average PA and NA between Q1 and Q2 as "final score"
        # If no valid pairs were found for a question, its score will be 0.0
        # If no questions had any valid pairs for calculation, final_pa/na can be None.
        valid_pa_scores = []
        if denominator_q1 > 0: valid_pa_scores.append(pa_q1)
        if denominator_q2 > 0: valid_pa_scores.append(pa_q2)

        valid_na_scores = []
        if denominator_q1_na > 0: valid_na_scores.append(na_q1)
        if denominator_q2_na > 0: valid_na_scores.append(na_q2)

        final_pa = sum(valid_pa_scores) / len(valid_pa_scores) if valid_pa_scores else None
        final_na = sum(valid_na_scores) / len(valid_na_scores) if valid_na_scores else None

        return final_pa, final_na, len(result_files_paths) # Return number of unique persons used

    def save_score_results(self, pa_score, na_score, num_results_used):
        """
        Saves the calculated PA/NA scores to a 'score.csv' file in the SCORE_DIR.
        Args:
            pa_score (float): Calculated Positive Agreement score.
            na_score (float): Calculated Negative Agreement score.
            num_results_used (int): Number of result files (persons) used for calculation.
        """
        print(f"DEBUG: Attempting to save PA/NA results.")

        try:
            os.makedirs(SCORE_DIR, exist_ok=True) # Ensure the score directory exists
            print(f"DEBUG: Score folder created/verified: '{SCORE_DIR}'")
        except Exception as e:
            print(f"ERROR: Could not create score folder '{SCORE_DIR}': {e}")
            return # Abort if folder cannot be created

        score_file_path = os.path.join(SCORE_DIR, "score.csv") # Full path to score.csv
        print(f"DEBUG: Full score file path: '{score_file_path}'")

        fieldnames = ['timestamp', 'num_results_used', 'pa_score', 'na_score']

        file_exists = os.path.exists(score_file_path) # Check if file already exists (to decide whether to write header)
        print(f"DEBUG: File '{score_file_path}' already exists: {file_exists}")

        try:
            with open(score_file_path, 'a', newline='', encoding='utf-8') as csvfile: # Open in append mode ('a')
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists: # If file didn't exist, write the header row
                    writer.writeheader()
                    print(f"DEBUG: Header written to '{score_file_path}'.")

                # Write the new row with scores
                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'num_results_used': num_results_used,
                    'pa_score': f"{pa_score:.4f}" if pa_score is not None else "N/A", # Format to 4 decimal places
                    'na_score': f"{na_score:.4f}" if na_score is not None else "N/A" # Format to 4 decimal places
                })
                print(f"DEBUG: Result row written to '{score_file_path}'.")

            print(f"PA/NA score saved to: {score_file_path}")
        except Exception as e:
            print(f"ERROR: Could not save PA/NA score: {e}")