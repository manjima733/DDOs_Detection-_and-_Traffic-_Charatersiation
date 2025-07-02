import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import BaggingClassifier
from sklearn.base import is_classifier
from sklearn.base import is_classifier, ClassifierMixin
import time
import psutil


from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, precision_score
)


print("Starting")

 
data = pd.read_csv('/home/g_kirubavathi/Amal/Manjima/merged_CSVs.csv', index_col=None)

rows, columns = data.shape

print(f"Total Rows_before: {rows}")
print(f"Total Columns_before: {columns}")

# Get the count of each unique value in the 'protocol' column
protocol_counts = data['protocol'].value_counts()

# Print the count of each protocol
print("Count of Each Protocol:\n", protocol_counts)

# Count the occurrences of each value in the 'label' column
label_counts = data['label'].value_counts()

# Print the counts
print("Distribution of label feature (as counts):\n", label_counts)

# Plot the distribution as a bar chart
plt.figure(figsize=(5, 4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')

# Add titles and labels
plt.title("Distribution of Label Feature", fontsize=7)
plt.xlabel("Label", fontsize=7)
plt.ylabel("Count", fontsize=7)
plt.xticks(rotation=45)
plt.show()
plt.tight_layout()
plt.savefig('Distribution_before.png', dpi=300)


# List of features to keep
features_to_keep = [
    "flow_id", "timestamp",  "src_port",  "dst_port",
    "protocol", "duration", "packets_count", "fwd_packets_count", 
    "bwd_packets_count", "total_payload_bytes", "fwd_total_payload_bytes", 
    "bwd_total_payload_bytes", "fwd_payload_bytes_max", "fwd_payload_bytes_min", 
    "fwd_payload_bytes_mean", "fwd_payload_bytes_std", "fwd_total_header_bytes", 
    "bwd_total_header_bytes", "bytes_rate", "packets_rate", "fwd_packets_rate", 
    "bwd_packets_rate", "min_payload_bytes_delta_len", "max_payload_bytes_delta_len", 
    "mean_payload_bytes_delta_len", "std_payload_bytes_delta_len", "avg_segment_size", 
    "avg_fwd_bytes_per_bulk", "avg_fwd_packets_per_bulk", "avg_fwd_bulk_rate", 
    "avg_bwd_bytes_per_bulk",  "active_mean", "active_std", 
    "active_max", "active_min", "idle_mean", "idle_std", "idle_max", "idle_min", 
    "label"
]

# Keep only the specified features
df = data[features_to_keep]

# Save the cleaned dataset
#df.to_csv('/home/g_kirubavathi/Amal/Manjima/merged_CSVs.csv', index=False)




rows, columns = df.shape

print(f"Total Rows: {rows}")
print(f"Total Columns: {columns}")



 #Identify columns with missing values
columns_with_nulls = df.columns[df.isnull().any()]

# Display columns and their null count
null_counts = df[columns_with_nulls].isnull().sum()

print("Features with null values:\n", null_counts)

# Total number of missing values in the entire dataset
total_missing = df.isnull().sum().sum()
print('Total number of missing values in the entire dataset',total_missing )





# Count the occurrences of each value in the 'label' column
label_counts = df['label'].value_counts()

# Print the counts
print("Distribution of label feature (as counts):\n", label_counts)

# Plot the distribution as a bar chart
plt.figure(figsize=(5, 4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')

# Add titles and labels
plt.title("Distribution of Label Feature", fontsize=7)
plt.xlabel("Label", fontsize=7)
plt.ylabel("Count", fontsize=
           7)
plt.xticks(rotation=45)
plt.show()
plt.tight_layout()
plt.savefig("Distribution.png")


# Remove rows where 'label' is 'Suspicious' or 'label'
df = df[~df['label'].isin(['Suspicious', 'label'])]

# Verify the changes
print(df['label'].value_counts())

# Count the occurrences of each value in the 'label' column
label_count = df['label'].value_counts()

# Print the counts
print("Distribution of label feature (as counts):\n", label_count)

# Plot the distribution as a bar chart
plt.figure(figsize=(5, 4))
sns.barplot(x=label_count.index, y=label_count.values, palette='viridis')

# Add titles and labels
plt.title("Distribution of Label Feature", fontsize=7)
plt.xlabel("Label", fontsize=7)
plt.ylabel("Count", fontsize=
           7)
plt.xticks(rotation=45)
plt.show()
plt.tight_layout()
plt.savefig("Distribution_after.png")






# Remove rows where 'protocol' column has invalid values (e.g., 'protocol')
df = df[df['protocol'] != 'protocol']

# Get the count of each unique value in the 'protocol' column again
protocol_counts = df['protocol'].value_counts()

# Print the count of each protocol
print("Cleaned Count of Each Protocol:\n", protocol_counts)
 




# Remove rows with null values
df = df.dropna()

# Verify if null values are removed
print(f"Total number of missing values in the dataset after removal: {df.isnull().sum().sum()}")

# Save the cleaned dataset
#df.to_csv('clean_data.csv', index=False)


# Convert columns to appropriate data types
# Convert numerical columns to float (if they are object type)
numerical_columns = [
    'duration', 'packets_count', 'fwd_packets_count', 'bwd_packets_count', 
    'total_payload_bytes', 'fwd_total_payload_bytes', 'bwd_total_payload_bytes', 
    'fwd_payload_bytes_max', 'fwd_payload_bytes_min', 'fwd_payload_bytes_mean', 
    'fwd_payload_bytes_std', 'fwd_total_header_bytes', 'bwd_total_header_bytes', 
    'bytes_rate', 'packets_rate', 'fwd_packets_rate', 'bwd_packets_rate', 
    'min_payload_bytes_delta_len', 'max_payload_bytes_delta_len', 
    'mean_payload_bytes_delta_len', 'std_payload_bytes_delta_len', 
    'avg_segment_size', 'avg_fwd_bytes_per_bulk', 'avg_fwd_packets_per_bulk', 
    'avg_fwd_bulk_rate', 'avg_bwd_bytes_per_bulk',  
    'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean', 
    'idle_std', 'idle_max', 'idle_min'
]

# Convert to float
df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Check for missing values (NaN)
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Check for infinite values (inf or -inf)
infinite_values = df.isin([float('inf'), float('-inf')]).sum()
print("\nInfinite values in each column:")
print(infinite_values)




 #Display data info
print(df.info())
print(df.describe())



df.drop(columns=['flow_id', 'timestamp'], inplace=True)



# Encode categorical variables
label_encoder_protocol = LabelEncoder()
label_encoder_label = LabelEncoder()

df['protocol'] = label_encoder_protocol.fit_transform(df['protocol'])
df['label'] = label_encoder_label.fit_transform(df['label'])

# Print the mappings
protocol_mapping = dict(zip(label_encoder_protocol.classes_, range(len(label_encoder_protocol.classes_))))
label_mapping = dict(zip(label_encoder_label.classes_, range(len(label_encoder_label.classes_))))

print("Protocol Mapping:")
for category, encoding in protocol_mapping.items():
    print(f"{category} -> {encoding}")

print("\nLabel Mapping:")
for category, encoding in label_mapping.items():
    print(f"{category} -> {encoding}")



# Split data
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
   

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# Compute and visualize correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_columns.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
plt.tight_layout()
plt.savefig('FeatureCore.png')

# Label distribution plot
sns.countplot(x='label', data=df)
plt.title('Label Distribution')
plt.show()
plt.tight_layout()
plt.savefig('Label_Dis.png')

#FEATURE SELECTION

 
# Initialize RandomForest for feature importance
rf = RandomForestClassifier(random_state=42)

# Fit the model to calculate feature importance
rf.fit(X_train_scaled, y_train)

# Get feature importance scores
feature_importances = rf.feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display top features
print(feature_importance_df)

# Visualize feature importance

plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show highest importance on top
plt.title("Feature Importance Using Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
plt.savefig('RF_Feature_importance.png')

# Select top k features based on importance
k = 10  # Specify the number of top features
top_features = feature_importance_df.head(k)['Feature'].values
X_train_selected = X_train_scaled[:, np.isin(X.columns, top_features)]
X_test_selected = X_test_scaled[:, np.isin(X.columns, top_features)]

print(f"Top {k} Selected Features: {top_features}")




# Split data into features and target
X = df.drop(columns=['label'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Function to measure CPU utilization
def measure_cpu_utilization(interval=1):
    cpu_utilization = []
    start_time = time.time()
    while time.time() - start_time < interval:
        cpu_utilization.append(psutil.cpu_percent(interval=0.1))
    return np.mean(cpu_utilization)


 #Function to measure time complexity
def measure_time_complexity(model, X_train, y_train, X_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    model.predict(X_test)
    prediction_time = time.time() - start_time
    return training_time, prediction_time
# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_scores, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'{model_name}_precision_recall_curve.png', dpi=300)
    plt.show() 



 # Function to plot accuracy-precision curve
def plot_accuracy_precision_curve(y_true, y_scores, model_name):
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []


    for thresh in thresholds:
        y_pred_thresh = (y_scores >= thresh).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred_thresh))
        precisions.append(precision_score(y_true, y_pred_thresh))

    plt.figure()
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.title(f'{model_name} Accuracy-Precision Curve')
    plt.legend()
    plt.savefig(f'{model_name}_accuracy_precision_curve.png', dpi=300)
    plt.show()
         


 # Function to plot CPU utilization
def plot_cpu_utilization(cpu_utilization, model_name):
    plt.figure()
    plt.bar(model_name, cpu_utilization, color='blue')
    plt.xlabel('Model')
    plt.ylabel('CPU Utilization (%)')
    plt.title(f'{model_name} CPU Utilization')
    plt.savefig(f'{model_name}_cpu_utilization.png', dpi=300)
    plt.show()

# Function to plot time complexity
def plot_time_complexity(training_time, prediction_time, model_name):
    plt.figure()
    plt.bar(['Training Time', 'Prediction Time'], [training_time, prediction_time], color=['green', 'orange'])
    plt.ylabel('Time (seconds)')
    plt.title(f'{model_name} Time Complexity')
    plt.savefig(f'{model_name}_time_complexity.png', dpi=300)
    plt.show()


# Stacking Classifier for improved performance
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Measure time complexity
    training_time, prediction_time = measure_time_complexity(model, X_train_scaled, y_train, X_test_scaled)
    print(f"{name} Training Time: {training_time:.4f}s")
    print(f"{name} Prediction Time: {prediction_time:.4f}s")
    
    # Measure CPU utilization
    cpu_utilization = measure_cpu_utilization()
    print(f"{name} CPU Utilization: {cpu_utilization:.4f}%")

    # Plot CPU utilization
    plot_cpu_utilization(cpu_utilization, name)
    
    # Plot time complexity
    plot_time_complexity(training_time, prediction_time, name)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]


    # Evaluation metrics
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred) * 100:.4f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'{name} Confusion Matrix')
    plt.savefig(f'{name}_confusion_matrix.png', dpi=300)
    plt.show()


    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'{name} ROC Curve')
    plt.legend()
    plt.savefig(f'{name}_roc_curve.png', dpi=300)
    plt.show()
    

     # Precision-Recall Curve
    plot_precision_recall_curve(y_test, y_pred_proba, name)
    
    # Accuracy-Precision Curve
    plot_accuracy_precision_curve(y_test, y_pred_proba, name) 

    
# Stacked Ensemble Bagging
base_learners = [
    ('rf', BaggingClassifier(base_estimator=RandomForestClassifier(random_state=42), n_estimators=10, random_state=42)),
    ('xgb', BaggingClassifier(base_estimator=XGBClassifier(eval_metric='logloss', random_state=42), n_estimators=10, random_state=42)),
    ('gb', BaggingClassifier(base_estimator=GradientBoostingClassifier(random_state=42), n_estimators=10, random_state=42))
]
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=GradientBoostingClassifier(random_state=42))

print("\nTraining Stacked Ensemble Bagging...")


 #Measure time complexity
start_time = time.time()
stacked_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time
start_time = time.time()
stacked_model.predict(X_test_scaled)
prediction_time = time.time() - start_time

# Measure CPU utilization
cpu_utilization = measure_cpu_utilization()

# Evaluate Stacked Model
y_pred_stacked = stacked_model.predict(X_test_scaled)
y_pred_proba_stacked = stacked_model.predict_proba(X_test_scaled)[:, 1]

print("Stacked Model Classification Report:\n", classification_report(y_test, y_pred_stacked))
print("Stacked Model Accuracy:", accuracy_score(y_test, y_pred_stacked) * 100)

# Confusion Matrix for Stacked Model
cm_stacked = confusion_matrix(y_test, y_pred_stacked)
disp_stacked = ConfusionMatrixDisplay(confusion_matrix=cm_stacked)
disp_stacked.plot()
plt.title("Stacked Model Confusion Matrix")
plt.savefig('stacked_model_confusion_matrix.png', dpi=300)
plt.show()

# ROC Curve for Stacked Model
fpr_stacked, tpr_stacked, _ = roc_curve(y_test, y_pred_proba_stacked)
roc_auc_stacked = auc(fpr_stacked, tpr_stacked)
plt.figure()
plt.plot(fpr_stacked, tpr_stacked, label=f'Stacked Model (AUC = {roc_auc_stacked:.4f})')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Stacked Model ROC Curve")
plt.legend()
plt.savefig('stacked_model_roc_curve.png', dpi=300)
plt.show()

# Plot CPU utilization for Stacked Model
plot_cpu_utilization(cpu_utilization, "Stacked Ensemble Bagging")

# Plot time complexity for Stacked Model
plot_time_complexity(training_time, prediction_time, "Stacked Ensemble Bagging")

# Plot precision-recall curve for Stacked Model
plot_precision_recall_curve(y_test, y_pred_proba_stacked, "Stacked Ensemble Bagging")

# Plot accuracy-precision curve for Stacked Model
plot_accuracy_precision_curve(y_test, y_pred_proba_stacked, "Stacked Ensemble Bagging")