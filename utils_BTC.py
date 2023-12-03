from PIL import Image
import pandas as pd
import numpy as np
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix
from monai.transforms import (
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    Resize
)
from monai.transforms import Transform, Compose
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# getting image location
def get_image(directory):
    image_paths = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith('.png')]

    return image_paths

# getting label
def get_label(csv_path):
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("The CSV does not contain a 'label' column.")

    labels = df["label"].tolist()

    print("First 5 labels:", labels[:5])

    return labels


# getting trandforms
def get_transforms(image_path):
    match = re.search(r'N(\d+)', image_path)
    if not match:
        raise ValueError("Wrong file structure")

    dimension = int(match.group(1))

    if dimension == 0:
        pass
    elif dimension == 1:
        print('start to us tranform in dimension 3')

        class DiscardAlphaChannel(Transform):
            def __call__(self, data):
                if data.shape[0] == 4:  # Check if data has 4 channels
                    return data[:3]  # Discard the alpha channel
                return data

        class PrintShape(Transform):
            def __call__(self, data):
                print(data.shape)
                return data

        train_transforms = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                DiscardAlphaChannel(),
                Resize((256, 256)),  # For RGB 2D images
                ScaleIntensity(),
                RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandFlip(spatial_axis=0, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ]
        )

        val_transforms = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                DiscardAlphaChannel(),
                Resize((256, 256)),  # For RGB 2D images
                ScaleIntensity(),
            ]
        )

        return train_transforms, val_transforms

    elif dimension == 2:
        pass

    else:
        raise ValueError("Wrong value number")


# spliting dataset
def split_dataset(files, k=5):
    def extract_number(filename):
        basename = os.path.basename(filename)
        number = int(basename.split('_')[1])
        return number

    unique_numbers = list(set(extract_number(file) for file in files))
    np.random.shuffle(unique_numbers)

    segment_length = len(unique_numbers) // k

    all_indices = []

    for i in range(k):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i != k - 1 else len(unique_numbers)

        val_numbers = unique_numbers[start_idx:end_idx]
        train_numbers = unique_numbers[:start_idx] + unique_numbers[end_idx:]

        train_files = [file for file in files if extract_number(file) in train_numbers]
        val_files = [file for file in files if extract_number(file) in val_numbers]

        train_indices = [files.index(file) for file in train_files]
        val_indices = [files.index(file) for file in val_files]

        all_indices.append((train_indices, val_indices))

    return all_indices

# creating result folder
def create_results_folder(base_directory):
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    existing_folders = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]
    folder_count = len(existing_folders)

    new_folder_name = f"training_results_{folder_count + 1}"
    new_folder_path = os.path.join(base_directory, new_folder_name)
    os.makedirs(new_folder_path)
    print(new_folder_path)

    return new_folder_path

# voting
def majority_vote(votes):
    num_ones = np.sum(votes)
    num_zeros = len(votes) - num_ones

    return 1 if num_ones > num_zeros else 0

# roc
def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
def plot_confusion_matrix(vote_label, vote_pred, save_path=None):
    cm = confusion_matrix(vote_label, vote_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('confusion metrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    plt.savefig(save_path)
    plt.close()

# f1
def plot_f1_curve(y_true, y_scores, save_path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
    f1_scores = np.nan_to_num(f1_scores)

    max_f1_index = np.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_index]
    best_threshold = thresholds[max_f1_index]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.scatter(thresholds[max_f1_index], max_f1, color='red', label=f'Best F1 Score: {max_f1:.2f}')
    plt.xlabel('Thresholds')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Thresholds')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# final acc auc 
def save_accuracy_auc_plot(accuracy_history, auc_history, save_path):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(accuracy_history, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(auc_history, label='AUC', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# final loss
def plot_loss_history(loss_history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title("Training Loss History")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(save_path)
    plt.close()





