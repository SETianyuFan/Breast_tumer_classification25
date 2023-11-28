import os
import argparse
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_and_combine_slices(image_path, mask_path, axis=2, neighbors=0, slices=30):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    image_data = sitk.GetArrayFromImage(image)
    mask_data = sitk.GetArrayFromImage(mask)

    relevant_slices = []
    mask_areas = []

    for i in range(image_data.shape[axis]):
        if axis == 0:
            slice_image = image_data[i, :, :]
            slice_mask = mask_data[i, :, :]
        elif axis == 1:
            slice_image = image_data[:, i, :]
            slice_mask = mask_data[:, i, :]
        else:
            slice_image = image_data[:, :, i]
            slice_mask = mask_data[:, :, i]

        mask_area = np.sum(slice_mask)
        if mask_area > 0:
            relevant_slices.append(slice_image)
            mask_areas.append(mask_area)

    selected_indices = np.argsort(mask_areas)[-slices:]
    combined_images = []

    for idx in selected_indices:
        if neighbors == 0:
            combined_images.append(relevant_slices[idx])
        else:
            combined_image = []
            for n in range(-neighbors, neighbors + 1):
                ni = idx + n
                if 0 <= ni < len(relevant_slices):
                    combined_image.append(relevant_slices[ni])
                else:
                    combined_image.append(np.zeros_like(relevant_slices[0]))

            combined_images.append(np.stack(combined_image, axis=-1))

    return combined_images

def process_3D_images_and_save(args):
    print(f'start processing 3D images in {args.neighbors*2+1} dimension, {args.axis} axis and {args.slices} slices')
    data = pd.read_csv(args.csv_file)
    label_mapping = dict(zip(data['case_id'], data['label']))

    output_labels = []
    
    subfolder_name = f"N{args.neighbors}_A{args.axis}_S{args.slices}"
    output_base_subfolder = os.path.join(args.output_base_folder, subfolder_name)
    output_images_folder = os.path.join(output_base_subfolder, 'images')
    output_csv_folder = os.path.join(output_base_subfolder, 'csv')
    for folder in [output_base_subfolder, output_images_folder, output_csv_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)


    for folder in [args.output_base_folder, output_images_folder, output_csv_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    image_files = sorted([f for f in os.listdir(args.image_folder) if f.startswith("DATA_") and f.endswith(".nrrd")])

    for image_file in image_files:
        case_num = image_file.split('_')[1].split('.')[0]
        image_path = os.path.join(args.image_folder, image_file)
        mask_path = os.path.join(args.mask_folder, f"MASK_{case_num}.nrrd")

        combined_images = extract_and_combine_slices(image_path, mask_path, args.axis, args.neighbors, args.slices)

        for idx, img in enumerate(combined_images):
            output_name = f"DATA_{case_num}_{idx}.png"
            plt.imsave(os.path.join(output_images_folder, output_name), img)
            label = label_mapping[int(case_num)]
            if label == 'M':
                output_labels.append(1)
            else:
                output_labels.append(0)

        print(f"{image_file} finished")

    output_data = pd.DataFrame({'label': output_labels})
    output_data.to_csv(os.path.join(output_csv_folder, 'output_labels.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 3D medical images and save them along with labels.")
    parser.add_argument("--image_folder", required=False, default='/mnt/cifs/tdsc_project/tdsc_data/train/DATA', help="Path to DATA")
    parser.add_argument("--mask_folder", required=False, default='/mnt/cifs/tdsc_project/tdsc_data/train/MASK/unzip', help="Path to MASK")
    parser.add_argument("--csv_file", required=False, default='/mnt/cifs/tdsc_project/tdsc_data/train/labels.csv', help="Path to CSV")
    parser.add_argument("--output_base_folder", required=False, default='/mnt/cifs/tdsc_project/tdsc_data/slice_data', help="Path to output data")
    
    parser.add_argument("--neighbors", type=int, default=0, help="Number of neighboring slices")
    parser.add_argument("--axis", type=int, default=1, help="Axis (0, 1, or 2)(x, y, or z)")
    parser.add_argument("--slices", type=int, default=30, help="Number of slice")

    args = parser.parse_args()
    process_3D_images_and_save(args)
