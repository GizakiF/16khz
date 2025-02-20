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
from joblib import Parallel, delayed  # Import for parallel processing

# --- Parameters for modification ---
n_iterations = 50
n_splits_cv = 5  # Keeping 5-splits for CV within GridSearchCV
test_size_percent = 0.25
# Number of jobs for parallel processing, -1 means using all available cores
n_jobs_parallel = -1
# --- End of parameters ---

# Ensure output folder exists
# Modified output folder for iterations
outFolder = './out/sublevel_out/50splits/50splits_version2/'
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
    ('pca', PCA()),  # PCA for dimensionality reduction
    ('svm', SVC(random_state=42))  # SVM classifier
])

# Hyperparameter grid for GridSearchCV
param_grid = {
    # Reduced maximum
    'pca__n_components': [10, 20, 30, 40, 50],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}


def process_subject(iSubject, iteration_num, tabStrf, tabSubjectNb, class_, cv, param_grid, pipeline, iteration_outFolder, test_size_percent):
    """
    Processes a single subject for a given iteration.
    This function is designed to be run in parallel.
    """
    print(f"\n=== Training for Subject {
          iSubject} - Iteration {iteration_num + 1} ===")

    # Extract data for the current subject
    X = tabStrf[tabSubjectNb == iSubject]
    Y = class_[tabSubjectNb == iSubject]

    # Split into train/test sets - random_state is updated for each iteration
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size_percent,
        # Fix randomness for reproducibility, updated per iteration
        random_state=42 + iteration_num,
        stratify=Y  # Ensure class balance in splits
    )

    # --- SAVE X_train and Y_train HERE ---  (Important!)
    X_train_path = os.path.join(iteration_outFolder, f"subject_{
                                iSubject}_X_train.pkl")
    Y_train_path = os.path.join(iteration_outFolder, f"subject_{
                                iSubject}_Y_train.pkl")

    with open(X_train_path, 'wb') as f:
        pickle.dump(X_train, f)
    with open(Y_train_path, 'wb') as f:
        pickle.dump(Y_train, f)
    # --- END OF SAVING ---

    X_test_path = os.path.join(iteration_outFolder, f"subject_{
                               iSubject}_X_test.pkl")
    Y_test_path = os.path.join(iteration_outFolder, f"subject_{
                               iSubject}_Y_test.pkl")

    with open(X_test_path, 'wb') as f:
        pickle.dump(X_test, f)
    with open(Y_test_path, 'wb') as f:
        pickle.dump(Y_test, f)
    # --- END OF SAVING X_test and Y_test ---

    # Grid search with cross-validation
    clf = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='balanced_accuracy',  # Use balanced accuracy for imbalanced classes
        n_jobs=-1  # Parallelize within GridSearchCV
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

    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Subject {
              iSubject}) - Iteration {iteration_num + 1}")
    plt.colorbar()

    unique_labels = np.unique(Y_test)
    tick_positions = np.arange(len(unique_labels))
    plt.xticks(ticks=tick_positions, labels=unique_labels)
    plt.yticks(ticks=tick_positions, labels=unique_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    thresh = cm.max() / 2.
    rows, cols = cm.shape
    for i_cm_row in range(rows):
        for j_cm_col in range(cols):
            plt.text(j_cm_col, i_cm_row, format(cm[i_cm_row, j_cm_col], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i_cm_row, j_cm_col] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(iteration_outFolder,
                f"subject_{iSubject}_cm.png"))
    plt.close()

    # Classification report
    report = classification_report(Y_test, Y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))

    df = pd.DataFrame(report).transpose()
    df['subject'] = iSubject

    return {  # Return results as a dictionary
        'iSubject': iSubject,
        'balanced_acc': balanced_acc,
        'best_pca_n': best_pca_n,
        'cm': cm,
        'report_df': df,
        'nb_samples_subject': X.shape[0],
        'nb_samples_train_subject': X_train.shape[0]
    }


# Initialize lists to store results for all iterations
all_iterations_results = []

# --- Outer loop for iterations ---
for iteration_num in range(n_iterations):
    print(f"\n==================== Iteration {
          iteration_num + 1}/{n_iterations} =====================")

    iteration_outFolder = os.path.join(
        outFolder, f'iteration_{iteration_num + 1}')
    os.makedirs(iteration_outFolder, exist_ok=True)

    tabBAcc_iter = []
    nDim_optimal_pca_tab_iter = []
    nbSample_subjects_iter = []
    nbSample_train_subjects_iter = []
    all_cms_iter = {}
    all_reports_iter = []

    # Cross-validation setup - random_state is updated for each iteration
    cv = StratifiedKFold(n_splits_cv, shuffle=True,
                         random_state=42 + iteration_num)

    # Parallel processing of subjects using joblib
    parallel_results = Parallel(n_jobs=n_jobs_parallel)(  # Use n_jobs_parallel parameter
        delayed(process_subject)(iSubject, iteration_num, tabStrf, tabSubjectNb,
                                 class_, cv, param_grid, pipeline, iteration_outFolder, test_size_percent)
        for iSubject in subjectNbTab
    )

    # Aggregate results from parallel processing
    for result in parallel_results:
        tabBAcc_iter.append(result['balanced_acc'])
        nDim_optimal_pca_tab_iter.append(result['best_pca_n'])
        all_cms_iter[result['iSubject']] = result['cm']
        all_reports_iter.append(result['report_df'])
        nbSample_subjects_iter.append(result['nb_samples_subject'])
        nbSample_train_subjects_iter.append(result['nb_samples_train_subject'])

    # Calculate the average confusion matrix for this iteration
    average_cm_iter = np.mean(list(all_cms_iter.values()), axis=0).astype(int)

    plt.figure()
    plt.imshow(average_cm_iter, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(
        f"Average Confusion Matrix (All Subjects) - Iteration {iteration_num + 1}")
    plt.colorbar()
    # Use class_ labels directly as they are consistent
    unique_labels = np.unique(class_)
    tick_positions = np.arange(len(unique_labels))
    plt.xticks(ticks=tick_positions, labels=unique_labels)
    plt.yticks(ticks=tick_positions, labels=unique_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    thresh = average_cm_iter.max() / 2.
    rows, cols = average_cm_iter.shape
    for i_avg_cm_row in range(rows):
        for j_avg_cm_col in range(cols):
            plt.text(j_avg_cm_col, i_avg_cm_row, format(average_cm_iter[i_avg_cm_row, j_avg_cm_col], 'd'),
                     horizontalalignment="center",
                     color="white" if average_cm_iter[i_avg_cm_row, j_avg_cm_col] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(iteration_outFolder,
                "average_confusion_matrix.png"))
    plt.close()

    all_reports_df_iter = pd.concat(all_reports_iter, keys=subjectNbTab)
    all_reports_df_iter.to_csv(
        f"{iteration_outFolder}/classification_report_all_subjects.csv")

    # Store results for this iteration
    iteration_results = {
        'iteration': iteration_num + 1,
        'tabBAcc': tabBAcc_iter,
        'nDim_optimal_pca_tab': nDim_optimal_pca_tab_iter,
        'nbSample_subjects': nbSample_subjects_iter,
        'nbSample_train_subjects': nbSample_train_subjects_iter,
        'average_cm': average_cm_iter,
        'all_reports_df': all_reports_df_iter
    }
    all_iterations_results.append(iteration_results)

# --- After all iterations, summarize results ---
print("\n\n==================== Summary Across All Iterations =====================")

mean_balanced_accuracies_across_iterations = []
std_balanced_accuracies_across_iterations = []

for iteration_result in all_iterations_results:
    mean_balanced_accuracies_across_iterations.append(
        np.mean(iteration_result['tabBAcc']))
    std_balanced_accuracies_across_iterations.append(
        np.std(iteration_result['tabBAcc']))

avg_of_mean_bacc = np.mean(mean_balanced_accuracies_across_iterations)
std_of_mean_bacc = np.std(mean_balanced_accuracies_across_iterations)
avg_of_std_bacc = np.mean(std_balanced_accuracies_across_iterations)

print(f"\n=== Summary Across {n_iterations} Iterations ===")
print(f"Average of Mean Balanced Accuracy: {
      avg_of_mean_bacc:.3f} ± {std_of_mean_bacc:.3f}")
print(f"Average of Std Balanced Accuracy: {avg_of_std_bacc:.3f}")


# Deeper summary if needed, e.g., average confusion matrix across all iterations, etc.

print(f"\nOutput saved to: {outFolder}")
