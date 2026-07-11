# core-module-ai-ml
🍽️ PlateCalc — Food Calorie Estimator (YOLOv8-Seg)

PlateCalc estimates the total calorie count of a meal from a single photo. It detects and
segments individual food items on a plate, estimates each item's weight using portion
heuristics, and looks up calorie data per item — combining computer vision with a
nutrition database to produce a total meal calorie estimate.

Objective

The goal of this project is to answer a simple question from a single 2D image: "roughly
how many calories are on this plate?" — without requiring the user to manually log foods
or weigh their meal. This is meant as an accessible, low-friction calorie estimation tool,
not a clinical-grade nutrition measurement device.

Approach & Methodology

The pipeline has four stages:


Dataset preparation — The FoodSeg103
dataset (7,000+ images across 103 food categories) is downloaded from Kaggle. Its
pixel-mask annotations (ann_dir/*.png, where each pixel value = class ID) are converted
into YOLOv8 segmentation format — one polygon per connected region — using OpenCV contour
detection (cv2.findContours).
Model training — A YOLOv8-Seg model (yolov8n-seg.pt, the nano variant, chosen for
fast training on Colab's free-tier GPU) is fine-tuned on the converted dataset. Checkpoints
save to Google Drive after each epoch, so a Colab disconnect only costs progress since the
last checkpoint — training resumes automatically from last.pt if interrupted.
Inference & portion estimation — For a new meal photo:

YOLOv8-Seg detects and segments each food item, producing per-item pixel masks.
OpenCV (cv2.connectedComponents) analyzes each mask to count discrete instances
(e.g. how many eggs) and measure pixel area.
Weight is estimated using one of two heuristics depending on food type:

Countable items (banana, egg, apple, etc.): weight = instance_count × average_unit_weight
Area-based items (rice, curry, noodles, etc.): weight = (mask_area / image_area) × reference_full_frame_weight



Optionally, a reference object of known size (e.g. a coin) placed in the photo can be
used to compute a real-world mm_per_pixel scale factor, converting pixel area into
physical area (cm²) for a more accurate, less heuristic-dependent weight estimate.



Calorie lookup — Each detected food name is queried against the
USDA FoodData Central API to get calories per 100g
(preferring lab-measured Foundation/SR Legacy entries over Branded products).
Results are cached in-memory per session to avoid redundant API calls. Estimated weight
and calories/100g are combined per item, then summed for a total meal calorie estimate.


Setup & Execution

This project runs as a single Google Colab notebook, top to bottom.

Prerequisites (once per session):


Open the notebook in Colab and set Runtime > Change runtime type > T4 GPU (training
requires a GPU; inference alone can run on CPU but is slower).
Get a free Kaggle API token: Kaggle → Account → Create New API Token (downloads
kaggle.json).
Get a free USDA FoodData Central API key: https://fdc.nal.usda.gov/api-key-signup


Steps:


Run Step 1 to install dependencies (ultralytics, opencv-python-headless,
kaggle, requests, pandas).

Run Step 1b to mount Google Drive — this is where training checkpoints are saved
(/content/drive/MyDrive/platecalc_runs), so a disconnect never loses more than one
epoch of progress.

Run Step 2 and upload your kaggle.json when prompted to download FoodSeg103.

Run Steps 3–6 to inspect the dataset structure, load category names, convert masks
to YOLO-format labels, and generate data.yaml. Adjust the hardcoded paths in these
cells if your Kaggle download unpacks with a different folder layout.

Run Step 7 to train YOLOv8-Seg (default: 10 epochs — increase once you've confirmed
the pipeline works end-to-end and want a stronger model).

Run Step 8 to load the best checkpoint (best.pt). In future sessions, once a
trained model exists on Drive, you can skip straight to Steps 1, 1b, and this step —
no need to re-download data or retrain.

Run Steps 9–11 to define the inference, mask-analysis, and weight-estimation
functions, and paste in your USDA API key.

Run Step 12, upload a plate/meal photo when prompted, and it will output a table of
detected items with estimated weight, calories/100g, and calories per item, plus a total
meal calorie estimate.

Step 13 (optional) visualizes the model's detections/masks overlaid on the image.



Assumptions, Limitations & Notes


No real-world scale by default. A single 2D photo has no inherent depth or scale
information, so weight estimates default to heuristics (average unit weight for countable
items; frame-area proportion for area-based items) calibrated for a "typical" phone photo
taken at a normal distance. These are approximations, not measurements.

Reference-object scaling is manual, not automatic. The notebook supports a
mm_per_pixel scale factor for more accurate area-to-weight conversion, but this value
must currently be measured and entered manually (e.g. by measuring a coin's pixel width in
an image editor). Automatically detecting a reference object would require an additional
trained detector, which is not implemented here.

Food density is generalized. When using the reference-object scaling path, a single
default thickness/density assumption (~0.5 cm thickness, ~1 g/cm³, i.e. water-like
density) is applied to all area-based foods, rather than food-specific densities.

Portion-weight lookup tables are limited. COUNTABLE_AVG_WEIGHT_G and
AREA_FOOD_REFERENCE_G only cover a small set of common foods; anything outside these
tables falls back to a generic default (DEFAULT_AREA_REFERENCE_G = 300), which reduces
accuracy for less common items. These tables are meant to be extended.

Calorie lookup accuracy depends on USDA name matching. The USDA search matches on the
raw detected class name; ambiguous or generic names (e.g. "curry") may return a
less-representative match than a more specific one would.

Model performance depends on training duration. The default of 10 epochs on the nano
model prioritizes fast iteration over accuracy — segmentation quality (and therefore
downstream weight/calorie accuracy) will improve with more epochs and/or a larger YOLOv8-Seg
variant (s/m), at the cost of longer training time.





