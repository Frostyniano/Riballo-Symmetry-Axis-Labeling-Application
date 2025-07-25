N_IMAGES = 96 #MAX number of images to process
AXES_PER_IMAGE = 10
QUESTIONS_PER_SESSION = 100 #Number of questions (100 questios = 5min)
TUTORIAL_MODE = True
PRE_SCALING_GREATER = True #Set this to True if the images are the original ones but the axis where mesured with a small resized version of the original image (e.g if the image is too big set the smallest dimention to MIN_DIM_TARGET)
PRE_SCALING_SMALLER = False #Set this to True if the images are the original ones but the axis where mesured with a large resized version of the original image (e.g if the image is too small set the smallest dimention to MIN_DIM_TARGET)
MIN_DIM_TARGET = 200 #The maximum value you want to the smallest dimension of the big images

THRESHOLD_NEAR_ONE = 0.9 #All the seconds axis that are over this thresholds will be considered as YN (good axis but not principal)
THRESHOLD_FAR_FROM_ONE = 0.3 #Now we are not using this, we assume that all non-YY and non-YN are NN

TARGET_YY_PERCENTAGE = 0.4 #The percentage of YY images you want
TARGET_YN_PERCENTAGE = 0.3 #The percentage of YN images you want
                           #The rest will be NN

DATA_DIR = "data"
IMAGES_DIR = f"{DATA_DIR}/images"
MAT_FILES_DIR = f"{DATA_DIR}/mat_files" 
CSV_INPUT_DIR = f"{DATA_DIR}/csv_files" 
RESULTS_DIR = "results" 
SCORE_DIR = "score"