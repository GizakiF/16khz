import pickle
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score

out_folder = './out/out_03_classicationSubjectLevel_AllMaps/10_17_removed_iteration/'
subject_models = {}

for filename in os.listdir(out_folder):
    if filename.endswith("_model.pkl"):  # Load models only
        subject_id = filename.split("_")[1]
        model_path = os.path.join(out_folder, filename)

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

                # --- Load X_train and Y_train ---
                X_train_path = os.path.join(out_folder, f"subject_{
                                            subject_id}_X_test.pkl")
                Y_train_path = os.path.join(out_folder, f"subject_{
                                            subject_id}_Y_test.pkl")

                try:
                    with open(X_train_path, 'rb') as f:
                        X_train = pickle.load(f)
                    with open(Y_train_path, 'rb') as f:
                        Y_train = pickle.load(f)

                    # --- Calculate Balanced Accuracy ---
                    # Predict on training data
                    Y_pred_train = model.predict(X_train)
                    balanced_acc = balanced_accuracy_score(
                        Y_train, Y_pred_train)

                    subject_models[subject_id] = {
                        'model': model, 'balanced_accuracy': balanced_acc}
                    print(f"Subject {subject_id}: Balanced Accuracy = {
                          balanced_acc:.4f}")

                except FileNotFoundError:
                    print(f"X_train or Y_train data not found for subject {
                          subject_id}. Skipping.")
                    # Store None if data isn't found
                    subject_models[subject_id] = {
                        'model': model, 'balanced_accuracy': None}

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

# --- Calculate and print average balanced accuracy ---
balanced_accuracies = [
    data['balanced_accuracy'] for data in subject_models.values() if data['balanced_accuracy'] is not None
]

if balanced_accuracies:
    avg_balanced_accuracy = np.mean(balanced_accuracies)
    print(f"\nAverage Balanced Accuracy across all subjects: {
          avg_balanced_accuracy:.4f}")
else:
    print("\nNo balanced accuracy scores could be calculated (likely missing X_train/Y_train data).")


# --- Save aggregated data (optional) ---
output_metrics_path = os.path.join(
    out_folder, "aggregated_models_with_metrics.pkl")
with open(output_metrics_path, 'wb') as f:
    pickle.dump(subject_models, f)

print(f"Aggregated models and metrics saved to: {output_metrics_path}")
