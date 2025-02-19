import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scipy.io as sio
import os
import pandas as pd
import numpy as np

# Ensure output folder exists
outFolder = './out/sublevel_out/version2/'
os.makedirs(outFolder, exist_ok=True)

# location of data.pkl
dataFolder = ''

data = pickle.load(open(os.path.join(dataFolder, 'data.pkl'), 'rb'))

# Extract data
tabStrf = data['tabStrf']
tabStrf = tabStrf / np.amax(tabStrf)  # Normalize
tabSession = data['tabSession']
tabSubjectNb = data['tabSubjectNb']
del data

# Convert to numpy arrays
tabStrf = np.asarray(tabStrf)
tabSession = np.asarray(tabSession)
tabSubjectNb = np.asarray(tabSubjectNb)
subjectNbTab = np.unique(tabSubjectNb)  # Unique subjects

# Class labels
class_ = tabSession

# Define PCA and SVM pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('pca', PCA()),                # PCA for dimensionality reduction
    ('svm', SVC(random_state=42))  # SVM classifier
])

# Hyperparameter grid for GridSearchCV
param_grid = {
    # Reduced maximum
    'pca__n_components': [10, 20, 30, 40, 50],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

# Initialize lists to store results
tabMeanAccuracy = []
tabStdAccuracy = []
tabBAcc = []
nDim_optimal_pca_tab = []
nbSample_subjects = []
nbSample_train_subjects = []

# Cross-validation setup
cv = StratifiedKFold(5, shuffle=True, random_state=42)

# Dictionary to store confusion matrices for each subject
all_cms = {}

# DataFrame to store all subjects' results
all_subjects_results = pd.DataFrame()

# Loop over subjects
for i, iSubject in enumerate(subjectNbTab):
    # if iSubject in [10, 17]:
    #     continue

    print(f"\n=== Training for Subject {iSubject} ===")

    # Extract data for the current subject
    X = tabStrf[tabSubjectNb == iSubject]
    Y = class_[tabSubjectNb == iSubject]

    # Split into train/test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.25,
        random_state=42,  # Fix randomness for reproducibility
        stratify=Y        # Ensure class balance in splits
    )

    # --- SAVE X_train and Y_train HERE ---  (Important!)
    X_train_path = os.path.join(outFolder, f"subject_{iSubject}_X_train.pkl")
    Y_train_path = os.path.join(outFolder, f"subject_{iSubject}_Y_train.pkl")

    with open(X_train_path, 'wb') as f:
        pickle.dump(X_train, f)
    with open(Y_train_path, 'wb') as f:
        pickle.dump(Y_train, f)
    # --- END OF SAVING ---

    X_test_path = os.path.join(outFolder, f"subject_{iSubject}_X_test.pkl")
    Y_test_path = os.path.join(outFolder, f"subject_{iSubject}_Y_test.pkl")

    with open(X_test_path, 'wb') as f:
        pickle.dump(X_test, f)
    with open(Y_test_path, 'wb') as f:
        pickle.dump(Y_test, f)
    # --- END OF SAVING X_test and Y_test ---

    # Store sample statistics
    nbSample_subjects.append(X.shape[0])
    nbSample_train_subjects.append(X_train.shape[0])

    # Grid search with cross-validation
    clf = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='balanced_accuracy',  # Use balanced accuracy for imbalanced classes
        n_jobs=-1  # Parallelize if multiple CPU cores are available
    )

    # Train the model
    clf.fit(X_train, Y_train)

    # Get best parameters
    best_pca_n = clf.best_params_['pca__n_components']
    best_svm_C = clf.best_params_['svm__C']
    best_svm_kernel = clf.best_params_['svm__kernel']
    print(f"Best PCA components: {best_pca_n}, Best SVM: C={
          best_svm_C}, kernel={best_svm_kernel}")

    # Evaluate on test set
    Y_pred = clf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    balanced_acc = balanced_accuracy_score(Y_test, Y_pred)
    print(f"Test Accuracy: {accuracy:.3f}, Balanced Accuracy: {
          balanced_acc:.3f}")

    # Store results
    tabBAcc.append(balanced_acc)
    nDim_optimal_pca_tab.append(best_pca_n)

    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)

    # Store the confusion matrix in the dictionary
    all_cms[iSubject] = cm

    # Plot and save confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Subject {iSubject})")
    plt.colorbar()

    unique_labels = np.unique(Y_test)
    tick_positions = np.arange(len(unique_labels))

    plt.xticks(ticks=tick_positions, labels=unique_labels)
    plt.yticks(ticks=tick_positions, labels=unique_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    thresh = cm.max() / 2.
    rows, cols = cm.shape

    for i in range(rows):
        for j in range(cols):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(outFolder, f"subject_{iSubject}_cm.png"))
    plt.close()

    # Classification report
    report = classification_report(Y_test, Y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))

    df = pd.DataFrame(report).transpose()  # Transpose for correct orientation
    df['subject'] = iSubject  # Add subject number as a column

    # Append to the DataFrame storing all subjects' results
    all_subjects_results = pd.concat([all_subjects_results, df])

    # Save model (optional)
    with open(os.path.join(outFolder, f"subject_{iSubject}_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

# Calculate the average confusion matrix
average_cm = np.mean(list(all_cms.values()), axis=0).astype(int)

# Plot and save the average confusion matrix
plt.figure()
plt.imshow(average_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Average Confusion Matrix (All Subjects)")
plt.colorbar()

unique_labels = np.unique(Y_test)
tick_positions = np.arange(len(unique_labels))

plt.xticks(ticks=tick_positions, labels=unique_labels)
plt.yticks(ticks=tick_positions, labels=unique_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

thresh = average_cm.max() / 2.
rows, cols = average_cm.shape

for i in range(rows):
    for j in range(cols):
        plt.text(j, i, format(average_cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if average_cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(os.path.join(outFolder, "average_confusion_matrix.png"))
plt.close()

# Save all subjects' results to a single CSV file
all_subjects_results.to_csv(os.path.join(
    outFolder, "all_subjects_results.csv"))

# Summary statistics
print("\n=== Summary ===")
print(f"Nb samples whole: M={np.mean(nbSample_subjects):.1f}, min={
      np.min(nbSample_subjects)}, max={np.max(nbSample_subjects)}")
print(f"Nb samples train: M={np.mean(nbSample_train_subjects):.1f}, min={
      np.min(nbSample_train_subjects)}, max={np.max(nbSample_train_subjects)}")
print(f"Mean Balanced Accuracy: {
      np.mean(tabBAcc):.3f} ± {np.std(tabBAcc):.3f}")
