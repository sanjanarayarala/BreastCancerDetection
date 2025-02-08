Breast Cancer Detection using Deep Learning

Overview

This project implements a deep learning model using Keras to detect breast cancer based on a dataset of tumor characteristics. The model classifies tumors as malignant or benign using a neural network trained on the Breast Cancer Wisconsin dataset.

Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains:

    -	30 numerical features extracted from digitized images of fine needle aspirate (FNA) 
    of breast masses.
    
    -	A binary diagnosis label: M (Malignant) or B (Benign).
    
    -	The dataset is preprocessed by encoding labels and normalizing features.
    
Technologies Used

    -	Python
    
    -	NumPy, Pandas (Data manipulation)
    
    -	Matplotlib, Seaborn (Data visualization)
    
    -	Scikit-learn (Data preprocessing, splitting, evaluation metrics)
    
    -	TensorFlow/Keras (Deep learning model)
    
Project Workflow

    1.	Data Preprocessing
    
        -	Load dataset
        
        -	Encode categorical labels
        
        -	Handle missing values (if any)
        
        -	Feature scaling (Standardization)
        
        -	Split into training and testing sets
        
    2.	Exploratory Data Analysis (EDA)
    
        -	Summary statistics
        
        -	Correlation heatmap
        
        -	Pairplot visualization
        
    3.	Model Building
    
        -	A deep neural network with three dense layers
        
        -	Activation functions: ReLU for hidden layers, Sigmoid for output
        
        -	Loss function: Binary Crossentropy
        
        -	Optimizer: Adam
        
    4.	Hyperparameter Tuning
    
        -	Grid search over batch size and epochs to find optimal parameters
        
    5.	Training and Evaluation
    
        -	Train the model using the best parameters
        
        -	Validate performance using accuracy and loss plots
        
        -	Compute confusion matrix and heatmap for performance visualization
        
Model Performance

    -	The trained model achieves high accuracy in detecting breast cancer.
    
    -	Accuracy score is computed based on correctly classified instances.
    
Results

    -	A confusion matrix is generated to evaluate false positives/negatives.
    
    -	Training and validation accuracy/loss plots illustrate the model's learning performance.

Installation & Usage

Prerequisites

Ensure you have Python installed along with the required libraries. Install dependencies using:

    - pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
    
Run the Project

    - Save the dataset as breastcancer.csv in the working directory and execute the script: python breast_cancer_detection.py


Future Enhancements
    -	Experiment with different neural network architectures.
    -	Implement advanced feature selection techniques.
    -	Optimize hyperparameters using Bayesian optimization.

