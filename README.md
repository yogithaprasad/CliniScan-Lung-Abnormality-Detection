# CliniScan: An Investigation into AI-Powered Lung Abnormality Detection

This repository contains the code and findings for a project to detect and localize lung abnormalities in chest X-ray images. The project uses the YOLOv8 model and investigates the challenges of training on the VinDr-CXR dataset.

## Project Summary & Experimental Journey

This project was conducted as a systematic investigation, documenting a common challenge in real-world data science: the quality of the source data.

### Phase 1: The Initial Goal (Multi-Class Detection)

The initial objective was to train a model to detect and classify 14 different types of lung abnormalities, as specified by the project requirements.

*   The code for this experiment is provided in `set_multi_class.py` and `train_multi_class.py`.
*   **Finding:** After training a YOLOv8s model, the performance was very low, achieving a best mAP50 score of only **0.0481**. This indicated a fundamental problem.

### Phase 2: Data Analysis & Discovery

To understand the poor performance, a deep analysis of the `train.csv` annotation file was conducted. This investigation revealed critical limitations in the dataset:
1.  **Sparse Annotations:** Out of over 67,000 annotation rows, only a small fraction provided the bounding box coordinates necessary for a detection task. After cleaning, only **446 unique images** were found to be usable.
2.  **Severe Class Imbalance:** The number of examples for each abnormality was extremely uneven, ranging from over 3,000 images for 'Aortic enlargement' to less than 100 for 'Pneumothorax'.
3.  **High Complexity:** The analysis showed that 100% of the usable images contained two or more overlapping abnormalities, meaning there were no simple, single-abnormality examples for the model to learn from.

### Phase 3: A New Strategy (Single-Class Detection)

Based on the findings from the data analysis, the project strategy was intelligently pivoted. The problem was simplified to train the model to detect a single, generic **"Abnormality"** class. This allowed the model to leverage all 446 usable images for one focused task.

*   The code for this improved approach is in `set_singleclass.py` and `train_singleclass.py`.
*   **Finding:** This experiment was a clear success. The new single-class model **nearly doubled the performance**, achieving a final mAP50 score of **0.0928**.

### Conclusion

The project successfully concludes that while an AI model can be trained on the VinDr-CXR dataset, its performance is primarily constrained by the sparse, imbalanced, and complex nature of the available bounding box annotations. 

The successful pivot to a single-class model demonstrates a key data science principle: adapting the problem to the limitations of the data can lead to a more robust and effective solution. The final deployed Streamlit application uses this superior single-class model to demonstrate these findings.

## How to Run the Final Demonstration

The final application uses the best-performing single-class model.

1.  Clone the repository and install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2.  Ensure you have the final trained model located at `D:/CliniScan_Project/yolo_results/run4_single_class_yolov8s2/weights/best.pt`. If not, update the `model_path` in `app.py`.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```