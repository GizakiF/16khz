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

# --- Parallelization Imports ---
from joblib import Parallel, delayed
# --- End Parallelization Imports ---

# Ensure output folder exists
# Modified folder name for parallel run
outFolder = './out/sublevel_out/50splits/50splits_parallel/'
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

# Cross-validation setup for inner loop (hyperparameter tuning)
# Inner CV remains 5-fold
cv_inner = StratifiedKFold(5, shuffle=True, random_state=42)


# --- Define function to process a single split ---
def process_split(iSplit, iSubject, X_subject, Y_subject, outFolder, cv_inner, param_grid):
    print(
        f"\n--- Split {iSplit+1}/50 for Subject {iSubject} (Process ID: {os.getpid()})---")

    # Split into train/test sets for this split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_subject, Y_subject,
        test_size=0.25,  # 25% test size
        random_state=42 + iSplit,  # Vary random state for each split
        stratify=Y_subject        # Ensure class balance in splits
    )

    # --- SAVE X_train and Y_train HERE for EACH SPLIT ---
    X_train_path = os.path.join(outFolder, f"subject_{iSubject}_split_{
                                iSplit+1}_X_train.pkl")
    Y_train_path = os.path.join(outFolder, f"subject_{iSubject}_split_{
                                iSplit+1}_Y_train.pkl")

    with open(X_train_path, 'wb') as f:
        pickle.dump(X_train, f)
    with open(Y_train_path, 'wb') as f:
        pickle.dump(Y_train, f)
    # --- END OF SAVING ---

    X_test_path = os.path.join(outFolder, f"subject_{
                               iSubject}_split_{iSplit+1}_X_test.pkl")
    Y_test_path = os.path.join(outFolder, f"subject_{
                               iSubject}_split_{iSplit+1}_Y_test.pkl")

    with open(X_test_path, 'wb') as f:
        pickle.dump(X_test, f)
    with open(Y_test_path, 'wb') as f:
        pickle.dump(Y_test, f)
    # --- END OF SAVING X_test and Y_test ---

    # Store train sample number here before clf
    nbSample_train = X_train.shape[0]

    # Grid search with cross-validation for hyperparameter tuning
    clf = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_inner,  # Inner CV
        scoring='balanced_accuracy',
        n_jobs=-1  # Keep n_jobs=-1 for inner CV parallelization if needed
    )

    # Train the model
    clf.fit(X_train, Y_train)

    # Get best parameters
    best_pca_n = clf.best_params_['pca__n_components']
    best_svm_C = clf.best_params_['svm__C']
    best_svm_kernel = clf.best_params_['svm__kernel']
    print(f"  Split {iSplit+1} - Best PCA components: {
          best_pca_n}, Best SVM: C={best_svm_C}, kernel={best_svm_kernel}")

    # Evaluate on test set
    Y_pred = clf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    balanced_acc = balanced_accuracy_score(Y_test, Y_pred)
    print(f"  Split {
          iSplit+1} - Test Accuracy: {accuracy:.3f}, Balanced Accuracy: {balanced_acc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)

    # Classification report
    report = classification_report(Y_test, Y_pred, output_dict=True)
    print(f"\n  Split {iSplit+1} - Classification Report:")
    print(classification_report(Y_test, Y_pred))

    df_split = pd.DataFrame(report).transpose()
    df_split['subject'] = iSubject
    df_split['split'] = iSplit + 1  # Add split number

    # No plotting or model saving inside process_split for cleaner parallel execution, can be added back if needed for each split

    return cm, balanced_acc, best_pca_n, df_split, nbSample_train


# Initialize lists to store results across subjects and splits
# List to hold results for each split for each subject
all_subjects_split_results = []
# List to hold averaged results across splits for each subject
all_subjects_average_results = []
all_average_cms = []  # List to hold average CMs for each subject
all_nDim_optimal_pca_tab = []
all_tabBAcc = []
all_nbSample_subjects = []
all_nbSample_train_subjects = []


# --- Loop over subjects ---
for i, iSubject in enumerate(subjectNbTab):
    print(f"\n=== Training for Subject {iSubject} ===")

    X_subject = tabStrf[tabSubjectNb == iSubject]
    Y_subject = class_[tabSubjectNb == iSubject]

    subject_split_results = []  # Store results for each split for the current subject
    subject_cms = []  # Store CMs for each split for the current subject
    subject_nDim_optimal_pca_tab = []
    subject_tabBAcc = []
    subject_nbSample_train_splits = []

    # --- Parallelize 50 Splits using joblib ---
    split_results = Parallel(n_jobs=8)(  # n_jobs=-1 to use all CPUs, adjust as needed
        delayed(process_split)(iSplit, iSubject, X_subject,
                               Y_subject, outFolder, cv_inner, param_grid)
        for iSplit in range(50)
    )
    # --- End Parallelization ---

    # --- Process results from parallel splits ---
    for cm, balanced_acc, best_pca_n, df_split, nbSample_train in split_results:
        subject_cms.append(cm)
        subject_tabBAcc.append(balanced_acc)
        subject_nDim_optimal_pca_tab.append(best_pca_n)
        subject_split_results.append(df_split)
        subject_nbSample_train_splits.append(nbSample_train)

    # After processing results, calculate averages (same as before - no changes needed here)
    average_cm_subject = np.mean(np.array(subject_cms), axis=0).astype(int)
    average_balanced_accuracy_subject = np.mean(subject_tabBAcc)
    average_optimal_pca_n_subject = np.mean(subject_nDim_optimal_pca_tab)
    average_nbSample_train_subject = np.mean(subject_nbSample_train_splits)

    all_average_cms.append(average_cm_subject)
    all_nDim_optimal_pca_tab.append(average_optimal_pca_n_subject)
    all_tabBAcc.append(average_balanced_accuracy_subject)
    # Total samples for subject
    all_nbSample_subjects.append(X_subject.shape[0])
    # Avg train samples across splits
    all_nbSample_train_subjects.append(average_nbSample_train_subject)

    # Average Confusion Matrix for subject - Plotting and saving remains in the main process
    plt.figure()
    plt.imshow(average_cm_subject, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Average Confusion Matrix (Subject {iSubject}, 50 Splits)")
    plt.colorbar()

    unique_labels = np.unique(Y_subject)
    tick_positions = np.arange(len(unique_labels))

    plt.xticks(ticks=tick_positions, labels=unique_labels)
    plt.yticks(ticks=tick_positions, labels=unique_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    thresh = average_cm_subject.max() / 2.
    rows, cols = average_cm_subject.shape
    for i_cm in range(rows):
        for j_cm in range(cols):
            plt.text(j_cm, i_cm, format(average_cm_subject[i_cm, j_cm], 'd'),
                     horizontalalignment="center",
                     color="white" if average_cm_subject[i_cm, j_cm] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(outFolder, f"subject_{
                iSubject}_average_cm_50splits.png"))
    plt.close()

    # Concatenate results from all splits for the current subject
    subject_results_df = pd.concat(subject_split_results)
    all_subjects_split_results.append(subject_results_df)

    # Aggregate and average classification report metrics across splits for each subject
    averaged_metrics = {}
    for class_label in np.unique(Y_subject):
        class_dfs = [df for df in subject_split_results if str(
            class_label) in df.index]

        if class_dfs:
            avg_precision = np.mean(
                [df.loc[str(class_label), 'precision'] for df in class_dfs])
            avg_recall = np.mean(
                [df.loc[str(class_label), 'recall'] for df in class_dfs])
            avg_f1_score = np.mean(
                [df.loc[str(class_label), 'f1-score'] for df in class_dfs])
            avg_support = np.mean([df.loc[str(class_label), 'support']
                                  # Support should be int
                                   for df in class_dfs]).astype(int)

            averaged_metrics[str(class_label)] = {
                'post_precision': avg_precision,
                'post_recall': avg_recall,
                'post_f1-score': avg_f1_score,
                'post_support': avg_support
            }
        else:
            averaged_metrics[str(class_label)] = {
                'post_precision': None,
                'post_recall': None,
                'post_f1-score': None,
                'post_support': None
            }

    # Handle 'accuracy', 'macro avg', 'weighted avg' - average across splits
    accuracy_values = [df.loc['accuracy', 'precision']
                       for df in subject_split_results if 'accuracy' in df.index]
    macro_avg_precision_values = [df.loc['macro avg', 'precision']
                                  for df in subject_split_results if 'macro avg' in df.index]
    macro_avg_recall_values = [df.loc['macro avg', 'recall']
                               for df in subject_split_results if 'macro avg' in df.index]
    macro_avg_f1_score_values = [df.loc['macro avg', 'f1-score']
                                 for df in subject_split_results if 'macro avg' in df.index]
    macro_avg_support_values = [df.loc['macro avg', 'support']
                                for df in subject_split_results if 'macro avg' in df.index]
    weighted_avg_precision_values = [df.loc['weighted avg', 'precision']
                                     for df in subject_split_results if 'weighted avg' in df.index]
    weighted_avg_recall_values = [df.loc['weighted avg', 'recall']
                                  for df in subject_split_results if 'weighted avg' in df.index]
    weighted_avg_f1_score_values = [df.loc['weighted avg', 'f1-score']
                                    for df in subject_split_results if 'weighted avg' in df.index]
    weighted_avg_support_values = [df.loc['weighted avg', 'support']
                                   for df in subject_split_results if 'weighted avg' in df.index]

    averaged_metrics['accuracy'] = {'post_precision': np.mean(
        accuracy_values) if accuracy_values else None}
    averaged_metrics['macro avg'] = {
        'post_precision': np.mean(macro_avg_precision_values) if macro_avg_precision_values else None,
        'post_recall': np.mean(macro_avg_recall_values) if macro_avg_recall_values else None,
        'post_f1-score': np.mean(macro_avg_f1_score_values) if macro_avg_f1_score_values else None,
        'post_support': np.mean(macro_avg_support_values) if macro_avg_support_values else None
    }
    averaged_metrics['weighted avg'] = {
        'post_precision': np.mean(weighted_avg_precision_values) if weighted_avg_precision_values else None,
        'post_recall': np.mean(weighted_avg_recall_values) if weighted_avg_recall_values else None,
        'post_f1-score': np.mean(weighted_avg_f1_score_values) if weighted_avg_f1_score_values else None,
        'post_support': np.mean(weighted_avg_support_values) if weighted_avg_support_values else None
    }

    averaged_df = pd.DataFrame(averaged_metrics).transpose()
    averaged_df['subject #'] = iSubject  # Subject Number
    # Append averaged results for subject
    all_subjects_average_results.append(averaged_df)


# Concatenate all subjects' averaged results
all_subjects_results_formatted = pd.concat(all_subjects_average_results)


# Add columns for pre_precision, pre_recall, pre_f1-score, pre_support as requested
for col_prefix in ['pre']:
    for metric in ['precision', 'recall', 'f1-score', 'support']:
        col_name = f'{col_prefix}_{metric}'
        if col_name not in all_subjects_results_formatted.columns:
            all_subjects_results_formatted[col_name] = None


# Reorder columns to match the desired format
columns_order = [
    'subject #', 'post_precision', 'post_recall', 'post_f1-score', 'post_support',
    'pre_precision', 'pre_recall', 'pre_f1-score', 'pre_support', 'accuracy',
    'macro avg', 'weighted avg'
]
all_subjects_results_formatted = all_subjects_results_formatted[columns_order]


# Save the formatted DataFrame to a CSV file
all_subjects_results_formatted.to_csv(os.path.join(
    # index=True to keep class labels as index
    outFolder, "all_subjects_results_formatted.csv"), index=True)


# Calculate overall summary statistics across subjects
print("\n=== Summary across subjects (averaged over 50 splits) ===")
print(f"Nb samples whole (avg per subject): M={np.mean(all_nbSample_subjects):.1f}, min={
      np.min(all_nbSample_subjects)}, max={np.max(all_nbSample_subjects)}")
print(f"Nb samples train (avg per subject, per split): M={np.mean(all_nbSample_train_subjects):.1f}, min={
      np.min(all_nbSample_train_subjects)}, max={np.max(all_nbSample_train_subjects)}")
print(f"Mean Balanced Accuracy (averaged across subjects and splits): {
      np.mean(all_tabBAcc):.3f} ± {np.std(all_tabBAcc):.3f}")


# Average confusion matrix across all subjects (average of average CMs per subject)
overall_average_cm = np.mean(np.array(all_average_cms), axis=0).astype(int)

# Plot and save the overall average confusion matrix
plt.figure()
plt.imshow(overall_average_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Overall Average Confusion Matrix (All Subjects, Averaged over 50 Splits)")
plt.colorbar()

unique_labels = np.unique(class_)
tick_positions = np.arange(len(unique_labels))

plt.xticks(ticks=tick_positions, labels=unique_labels)
plt.yticks(ticks=tick_positions, labels=unique_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

thresh = overall_average_cm.max() / 2.
rows, cols = overall_average_cm.shape
for i_cm in range(rows):
    for j_cm in range(cols):
        plt.text(j_cm, i_cm, format(overall_average_cm[i_cm, j_cm], 'd'),
                 horizontalalignment="center",
                 color="white" if overall_average_cm[i_cm, j_cm] > thresh else "black")

plt.tight_layout()
plt.savefig(os.path.join(
    outFolder, "overall_average_confusion_matrix_50splits.png"))
plt.close()
