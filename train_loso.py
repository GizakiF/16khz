import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np

# Ensure output folder exists
outFolder = './out/sublevel_out/LOSO_version/'
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
    'pca__n_components': [10, 20, 30, 40, 50],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

# Leave-One-Subject-Out Cross-Validation
loso = LeaveOneGroupOut()

# Storage for results
tabBAcc = []
all_cms = {}
all_subjects_results = pd.DataFrame()

# LOSO loop: Train on all but one subject, test on the left-out subject
for i, iSubject in enumerate(subjectNbTab):
    print(f"\n=== LOSO Training: Leaving Out Subject {iSubject} ===")

    # Train on all subjects except the current one
    train_indices = tabSubjectNb != iSubject
    test_indices = tabSubjectNb == iSubject

    X_train, Y_train = tabStrf[train_indices], class_[train_indices]
    X_test, Y_test = tabStrf[test_indices], class_[test_indices]

    # Save train-test data
    X_train_path = os.path.join(outFolder, f"subject_{iSubject}_X_train.pkl")
    Y_train_path = os.path.join(outFolder, f"subject_{iSubject}_Y_train.pkl")
    X_test_path = os.path.join(outFolder, f"subject_{iSubject}_X_test.pkl")
    Y_test_path = os.path.join(outFolder, f"subject_{iSubject}_Y_test.pkl")

    with open(X_train_path, 'wb') as f:
        pickle.dump(X_train, f)
    with open(Y_train_path, 'wb') as f:
        pickle.dump(Y_train, f)
    with open(X_test_path, 'wb') as f:
        pickle.dump(X_test, f)
    with open(Y_test_path, 'wb') as f:
        pickle.dump(Y_test, f)

    # Grid search with LOSO cross-validation
    clf = GridSearchCV(
        pipeline,
        param_grid,
        cv=loso.split(X_train, Y_train,
                      groups=tabSubjectNb[train_indices]),  # True LOSO
        scoring='balanced_accuracy',
        n_jobs=-1
    )

    # Train model
    clf.fit(X_train, Y_train)

    # Get best parameters
    best_pca_n = clf.best_params_['pca__n_components']
    best_svm_C = clf.best_params_['svm__C']
    best_svm_kernel = clf.best_params_['svm__kernel']
    print(f"Best PCA components: {best_pca_n}, Best SVM: C={
          best_svm_C}, kernel={best_svm_kernel}")

    # Test model on the left-out subject
    Y_pred = clf.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    balanced_acc = balanced_accuracy_score(Y_test, Y_pred)
    print(f"Test Accuracy: {accuracy:.3f}, Balanced Accuracy: {
          balanced_acc:.3f}")

    # Store results
    tabBAcc.append(balanced_acc)

    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    all_cms[iSubject] = cm

    # Save classification report
    report = classification_report(Y_test, Y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['subject'] = iSubject
    all_subjects_results = pd.concat([all_subjects_results, df])

    # Save model
    with open(os.path.join(outFolder, f"subject_{iSubject}_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

# Compute average confusion matrix
average_cm = np.mean(list(all_cms.values()), axis=0).astype(int)

# Save average confusion matrix
plt.figure()
plt.imshow(average_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Average Confusion Matrix (LOSO)")
plt.colorbar()

unique_labels = np.unique(class_)
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

# Save results to CSV
all_subjects_results.rename(columns={'subject': 'subject #'}, inplace=True)
all_subjects_results.to_csv(os.path.join(
    outFolder, "loso_results.csv"), index=False)

# Summary
print("\n=== LOSO Summary ===")
print(f"Mean Balanced Accuracy: {
      np.mean(tabBAcc):.3f} ± {np.std(tabBAcc):.3f}")
