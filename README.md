# Cloud Based DDOs Detection and Traffic Charatersiation
This project implements a machine learning-based system to detect Distributed Denial of Service (DDoS) attacks and characterize network traffic patterns in a cloud environment. The pipeline includes data preprocessing, feature selection, model training, performance evaluation, and visualization, with a focus on resource efficiency.

Objective
To build an accurate and efficient DDoS detection system using an ensemble of machine learning models, while analyzing and visualizing the behavior of various traffic types.

Dataset
The dataset used consists of network traffic logs, including features such as:
-Flow identifiers (flow_id, timestamp, src_port, dst_port)
-Packet statistics (fwd/bwd packet count, payload size, rate, etc.)
-Protocol-specific details
-Label: Benign, Attack, or Suspicious (Suspicious class is filtered out during preprocessing

Preprocessing
-Null and invalid values removed
-Protocol and label encoding using LabelEncoder
-Feature correlation analysis
-Removal of irrelevant columns (flow_id, timestamp)
-Standardization using StandardScaler

 Feature Selection
-Feature importance calculated using a Random Forest Classifier
-Top 10 features selected based on importance for dimensionality reduction

 Models Used
The following models are evaluated individually and in a stacked ensemble:
-Random Forest
-gradient Boosting
-XGBoost
-Stacked Ensemble Bagging (using the above as base learners)

Each model is evaluated using accuracy ,Precision-Recall Curve, ROC-AUC, Confusion Matrix,CPU Utilization, Time Complexity (Training and Prediction)

Visualizations
The system generates and saves:
-Label distribution plots (before & after cleaning)
-Correlation heatmap
-Feature importance bar chart
-ROC and PR curves
-Confusion matrices
-CPU and time complexity comparisons
