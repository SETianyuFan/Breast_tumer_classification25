import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, f1_score
import utils
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201, DenseNet264, SEResNext101, SENet154
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    Resize
)
from monai.utils import set_determinism
from monai.transforms import Transform, Compose
from torchvision import transforms
import seaborn as sns
import argparse
import torch.nn.functional as F

print_config()

if __name__ == "__main__":
    # setting args
    parser = argparse.ArgumentParser('HBC25', add_help=False)
    # dir
    parser.add_argument('--data_dir', default='/mnt/cifs/tdsc_project/tdsc_program/data/rgb_xy_30/images', type=str)
    parser.add_argument('--csv_dir', default='/mnt/cifs/tdsc_project/tdsc_data/train/labels.csv', type=str)
    # /mnt/cifs/tdsc_project/tdsc_data/slice_data/N1_A2_S30/images
    # /mnt/cifs/tdsc_project/tdsc_data/train/labels.csv
    # splite set
    parser.add_argument('--train_ratio', default=0.6, type=float)
    parser.add_argument('--val_ratio', default=0.6, type=float)
    # dataloader
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    # train
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--val_interval', default=3, type=int)
    # result
    parser.add_argument('--result_dir', default='result', type=str)


    args = parser.parse_args()
    torch.cuda.empty_cache()
    set_determinism(seed=7)

    # getting data
    images = utils.get_image_paths(args.data_dir)
    for image_number, slices in images.items():
        print(f"3D Image {image_number}:")
        for path in slices:
            print(f"{path}")

    # getting label
    labels = utils.get_labels(args.csv_dir)
    for image_number, label in labels.items():
        print(f"3D Image {image_number}: Label = {label}")

    # setting transforms
    train_transforms, val_transforms, test_transforms = utils.get_transforms_based_on_dimension(args.data_dir)

    # splite sets
    train_set, val_set, test_set, train_labels, val_labels, test_labels = utils.split_datasets_and_labels(images, labels, train_ratio=0.6, val_ratio=0.2)

    # creating datasets
    train_datasets = utils.create_datasets(train_set, train_labels, train_transforms)
    val_datasets = utils.create_datasets(val_set, val_labels, val_transforms)
    test_datasets = utils.create_datasets(test_set, test_labels, test_transforms)
    for image_number, dataset in val_datasets.items():
        print(f"3D Image Number: {image_number}")
        print(f"Number of slices in dataset: {len(dataset)}")
        first_image, first_label = dataset[0]
        print(f"image shape: {first_image.shape}")
        print(f"image label: {first_label}")

    # creating dataloaders
    train_loader_set = utils.create_dataloaders(train_datasets, args.batch_size,
                                                True, args.num_workers)
    val_loader_set = utils.create_dataloaders(val_datasets, args.batch_size, False,
                                              args.num_workers)
    test_loader_set = utils.create_dataloaders(test_datasets, args.batch_size, False,
                                               args.num_workers)
    first_key = next(iter(val_loader_set))
    first_dataloader = val_loader_set[first_key]
    for i, (data, labels) in enumerate(first_dataloader):
        print(f"Batch {i}")
        print(f"Data shape: {data.shape}")
        print(f"Labels: {labels}")
        
    # stop using test set
    train_loader_set = train_loader_set | test_loader_set

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet201(spatial_dims=2, in_channels=3, out_channels=2,
                        pretrained=True).to(device)

    # train feature
    num_class = 2
    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    max_epochs = args.max_epochs
    val_interval = args.val_interval
    auc_metric = ROCAUCMetric()
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    accuracy_history = []
    auc_history = []
    label_mapping = {'B': 0, 'M': 1}
    result_folder = utils.create_results_folder(args.result_dir) + '/'

    # start training
    for epoch in range(max_epochs):
        model.train()
    
        for i, train_loader in train_loader_set.items(): 
            step = 0
            for batch_data in train_loader:      
                inputs = batch_data[0].to(device)  
                labels = batch_data[1]  
                labels = [label_mapping[label] for label in labels]
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                step += 1
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                softmax_outputs = torch.argmax(outputs, dim=1)
                print(softmax_outputs)
                print(labels)
                loss.backward()
                optimizer.step()
                print(f'epoch: {epoch}/{max_epochs}| ',
                      f'3d data: {i}| ',
                      f"{step}/{len(train_loader)}| ", 
                      f"train_loss: {loss.item():.4f}")

        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            
            with torch.no_grad():
                total_accuracy = 0
                total_auc = 0
                num_samples = 0
                all_preds = []
                all_predict = []
                all_labels = []

                for image_number, val_loader in val_loader_set.items():
                    batch_preds = []

                    for batch_data in val_loader:
                        val_images = batch_data[0].to(device)
                        val_labels = batch_data[1]
                        val_labels = [label_mapping[label] for label in val_labels]
                        val_labels = torch.tensor(val_labels,
                                                  dtype=torch.long).to(device)
                        outputs = model(val_images)
                        probabilitie = F.softmax(outputs, dim=1)
                        outputs_numpy = probabilitie.detach().cpu().numpy()
                        
                        labels_numpy = val_labels.cpu().numpy()
                        batch_preds.append(outputs_numpy)
                        
                    batch_labels = labels_numpy[0]
                    aggregated_outputs = utils.aggregate_votes(batch_preds)
                    aggregated_labels = batch_labels
                    print(aggregated_outputs)
                    print(aggregated_labels)
                    all_preds.append(aggregated_outputs)
                    all_labels.append(aggregated_labels)

                # accuracy
                print(all_preds)
                print(all_labels)
                accuracy = accuracy_score(all_labels, all_preds)
                # auc_roc
                auc = roc_auc_score(all_labels, all_preds)

                accuracy_history.append(accuracy)
                auc_history.append(auc)

                num_samples += 1

                print(
                    f"Validation results - AUC: {auc:.4f}, "
                    f"Average Accuracy: {accuracy:.4f}"
                )

                if auc > best_metric:
                    best_metric = auc
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(result_folder, "best_metric_model.pth"))
                    print("saved new best metric model")

                    utils.plot_roc_curve(all_labels, all_preds, result_folder + 'roc_curve.png')
                    utils.plot_f1_curve(all_labels, all_preds, result_folder + 'f1_curve.png')

            average_accuracy = total_accuracy / num_samples
            average_auc = total_auc / num_samples

    utils.save_accuracy_auc_plot(accuracy_history, auc_history, result_folder + 'accuracy_auc_plot.png')
