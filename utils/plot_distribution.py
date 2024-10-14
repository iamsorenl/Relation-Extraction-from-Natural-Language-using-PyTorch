import numpy as np
import matplotlib.pyplot as plt

def plot_class_distributions(predictions, submission_labels, label_classes, expected_distribution):
    """
    Plots three separate bar graphs:
    1. Predicted class distribution (model's output).
    2. Expected class distribution (ground truth based on test set).
    3. Actual predicted distribution based on generated submission (test submission).
    """
    
    # Ensure that all distributions match the length of `label_classes`
    if len(expected_distribution) != len(label_classes):
        print(f"Warning: Mismatch in label_classes ({len(label_classes)}) and expected_distribution ({len(expected_distribution)}).")

        # Keep only valid classes that are in both expected_distribution and class_labels
        valid_indices = [i for i in range(min(len(label_classes), len(expected_distribution)))]
        label_classes = [label_classes[i] for i in valid_indices]
        expected_distribution = [expected_distribution[i] for i in valid_indices]

    # Convert predictions tensor to NumPy array
    predictions_np = predictions.cpu().numpy()

    # Calculate predicted class distribution (based on the model's predictions)
    predicted_class_counts = np.zeros(len(label_classes))  # Initialize array for valid labels

    # Sum the predictions for only the valid labels
    for i in range(len(label_classes)):
        predicted_class_counts[i] = np.sum(predictions_np[:, i])

    # Normalize the predicted distribution
    predicted_distribution = predicted_class_counts / np.sum(predicted_class_counts)

    # Now calculate the distribution of actual predicted labels (submission)
    submission_class_counts = np.zeros(len(label_classes))

    # Calculate the distribution of predicted labels in the submission file
    for labels in submission_labels:
        for label in labels:
            if label in label_classes:  # Ensure the label is in `label_classes`
                submission_class_counts[label_classes.index(label)] += 1

    # Normalize the submission distribution
    submission_distribution = submission_class_counts / np.sum(submission_class_counts)

    # Set up subplots: three plots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot predicted distribution (from model)
    index = np.arange(len(label_classes))
    ax1.bar(index, predicted_distribution, color='blue', alpha=0.6)
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Distribution')
    ax1.set_title('Predicted Class Distribution (Model)')
    ax1.set_xticks(index)
    ax1.set_xticklabels(label_classes, rotation=90)

    # Plot expected distribution (ground truth)
    ax2.bar(index, expected_distribution, color='green', alpha=0.6)
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Distribution')
    ax2.set_title('Expected Class Distribution (Ground Truth)')
    ax2.set_xticks(index)
    ax2.set_xticklabels(label_classes, rotation=90)

    # Plot submission distribution (predicted labels in test set submission)
    ax3.bar(index, submission_distribution, color='orange', alpha=0.6)
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Distribution')
    ax3.set_title('Actual Predicted Class Distribution (Submission)')
    ax3.set_xticks(index)
    ax3.set_xticklabels(label_classes, rotation=90)

    # Adjust layout to make it more readable
    plt.tight_layout()
    plt.show()