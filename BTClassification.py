import os
import torch
import numpy as np

from monai.data import DataLoader
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201
from monai.utils import set_determinism

import argparse
import utils_BTC
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

if __name__ == "__main__":
    # setting args
    parser = argparse.ArgumentParser('HBC25', add_help=False)
    # dir
    parser.add_argument('--data_dir', default='/mnt/cifs/tdsc_project/tdsc_data/slice_data/N1_A2_S30/images', type=str)
    parser.add_argument('--csv_dir',
                        default='/mnt/cifs/tdsc_project/tdsc_data/slice_data/N1_A2_S30/csv/output_labels.csv', type=str)
    # splite set
    parser.add_argument('--fold_num', default=0, type=int)
    # dataloader
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    # train
    parser.add_argument('--max_epochs', default=30, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--seed', default=700, type=int)
    # result
    parser.add_argument('--result_dir', default='result', type=str)
    args = parser.parse_args()

    # set up initial environment
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=args.seed)

    # load data and creating dataloader
    train_transforms, val_transforms = utils_BTC.get_transforms(args.data_dir)
    # data set
    class MedNISTDataset(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transforms):
            self.image_files = image_files
            self.labels = labels
            self.transforms = transforms

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            return self.transforms(self.image_files[index]), self.labels[index]
    image_files_list = utils_BTC.get_image(args.data_dir)
    image_class = utils_BTC.get_label(args.csv_dir)
    data_sets = utils_BTC.split_dataset(image_files_list)
    data_set = data_sets[args.fold_num]
    train_indices, val_indices = data_set
    train_x = [image_files_list[i] for i in train_indices]
    train_y = [image_class[i] for i in train_indices]
    val_x = [image_files_list[i] for i in val_indices]
    val_y = [image_class[i] for i in val_indices]
    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # set up tain\validation environment
    num_class = 2
    result_folder = utils_BTC.create_results_folder(args.result_dir) + '/'
    with open(result_folder + 'args.txt', 'w') as file:
        for arg, value in vars(args).items():
            file.write(f'{arg}: {value}\n')
    model = DenseNet201(spatial_dims=2, in_channels=3, out_channels=num_class, pretrained=False).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    max_epochs = args.max_epochs
    val_interval = args.val_interval

    best_metric = -1
    best_metric_epoch = -1
    loss_history = []
    accuracy_history = []
    auc_history = []

    # start training
    for epoch in range(max_epochs):
        print("-" * 10)
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"epoch {epoch + 1}/{max_epochs}",
                  f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        loss_history.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            group_pred = []
            group_label = []
            vote_pred = []
            vote_label = []
            val_i = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    
                    output = model(val_images)
                    probabilities = F.softmax(output, dim=1)
                    binary_output_cpu = torch.argmax(probabilities, dim=1).cpu().numpy()
                    val_labels_cpu = val_labels.cpu().numpy()
                    for i in binary_output_cpu:
                        group_pred.append(binary_output_cpu[i])
                        group_label.append(val_labels_cpu[i])

                    val_i += 1
                    if val_i % 3 == 0:
                        vote_pred.append(utils_BTC.majority_vote(group_pred))
                        vote_label.append(group_label[0])
                        group_pred = []
                        group_label = []

                accuracy = accuracy_score(vote_label, vote_pred)
                auc = roc_auc_score(vote_label, vote_pred)
                accuracy_history.append(accuracy)
                auc_history.append(auc)

                print(
                    f"Validation results - AUC: {auc:.4f}, "
                    f"Average Accuracy: {accuracy:.4f}"
                )

                if auc > best_metric:
                    best_metric = auc
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(result_folder, "best_metric_model.pth"))
                    print("saved new best metric model")

                    utils_BTC.plot_roc_curve(vote_label, vote_pred, result_folder + 'roc_curve.png')
                    utils_BTC.plot_f1_curve(vote_label, vote_pred, result_folder + 'f1_curve.png')
                    utils_BTC.plot_confusion_matrix(vote_label, vote_pred, result_folder + 'confusion_matrix.png')

    utils_BTC.save_accuracy_auc_plot(accuracy_history, auc_history, result_folder + 'accuracy_auc_plot.png')
    utils_BTC.plot_loss_history(loss_history, result_folder + 'train_loss_history.png')








