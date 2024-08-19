import os
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

def str2int(car_id_num: str, dataset: str):
    if dataset == 'veri':
        if len(car_id_num) == 1:
            car_id_num = '00' + str(car_id_num)
        elif len(car_id_num) == 2:
            car_id_num = '0' + str(car_id_num)
        else:
            pass
    elif dataset == 'vehicleid' or dataset == 'veriwild':
        if len(car_id_num) == 1:
            car_id_num = '0000' + car_id_num
        elif len(car_id_num) == 2:
            car_id_num = '000' + car_id_num
        elif len(car_id_num) == 3:
            car_id_num = '00' + car_id_num
        elif len(car_id_num) == 4:
            car_id_num = '0' + car_id_num
        else:
            pass
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return car_id_num

def compute_gms_matches(orb: cv2.ORB, bf: cv2.BFMatcher, img1: np.ndarray, img2: np.ndarray, verbose: bool = False):
    # Detect and compute keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Check if descriptors were found
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        if verbose:
            print(f"Warning: No descriptors found for one of the images. Returning 0 matches.")
        return 0

    if des1.shape[1] != des2.shape[1]:
        if verbose:
            print(f"Error: Descriptor sizes don't match. Cannot proceed with matching.")
        return 0

    # Convert des1 and des2 to have the same type
    # Fixes: cv2.error: OpenCV(4.10.0) /io/opencv/modules/core/src/batch_distance.cpp:274: error: (-215:Assertion failed) type == src2.type() && src1.cols == src2.cols && (type == CV_32F || type == CV_8U) in function 'batchDistance'
    if(des1.dtype != [np.uint8, np.float32]) or (des1.dtype != [np.uint8, np.float32]):
        if verbose:
            print(f"Warning: Converting descriptors to np.uint8.")
        des1 = des1.astype(np.uint8)
        
    if(des2.dtype != [np.uint8, np.float32]) or (des2.dtype != [np.uint8, np.float32]):
        if verbose:
            print(f"Warning: Converting descriptors to np.uint8.")
        des2 = des2.astype(np.uint8)

    # Perform initial matching
    matches = bf.match(des1, des2)

    # Apply GMS matching
    gms_matches = cv2.xfeatures2d.matchGMS(size1=img1.shape[:2], size2=img2.shape[:2],
                                            keypoints1=kp1, keypoints2=kp2,
                                            matches1to2=matches, withRotation=True)
    return len(gms_matches)

def process_class(image_paths: list, image_size: tuple = (224, 224), verbose: bool = False):
    n = len(image_paths)
    width, height = image_size
    adj_matrix = np.zeros((n, n), dtype=np.int32) # Initialize the adjacency matrix

    # Iterate over all the images
    for i in range(n):
        # Read and resize, as per paper
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (width, height))
        img1 = img1.astype(np.uint8)

        # Only iterate over j > i
        for j in range(i + 1, n):
            # Read and resize, as per paper
            img2 = cv2.imread(image_paths[j], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.resize(img2, (width, height))
            img2 = img2.astype(np.uint8)

            # Compute GMS matches
            matches = compute_gms_matches(orb, bf, img1, img2, verbose)

            # Set both (i,j) and (j,i) at once
            adj_matrix[i, j] = matches
            adj_matrix[j, i] = matches

        pbar.update(1)

    return adj_matrix

def get_dict(dataset: str, train_file: str, img_dir: str):
    class_images = defaultdict(list)
    original_to_new_id = {}
    new_id_counter = 1
    
    # Read query file names
    if (dataset == 'veri'):
        # Open the file with the correct encoding
        with open(train_file, 'r', encoding='gb2312') as file:
            xml_content = file.read()
        
        # Parse the XML string
        root = ET.fromstring(xml_content)
                
        # Iterate through each Item element
        for item in root.findall('.//Item'):
            vehicle_id_str = item.get('vehicleID')
            
            car_id_num = str(int(re.search(r'\d+', vehicle_id_str).group()))
            car_id_num = str2int(car_id_num, dataset)
            
            full_image_path = os.path.join(img_dir, item.get('imageName'))
            
            class_images[car_id_num].append(full_image_path)
    elif (dataset == 'veriwild'):
        with open(train_file, 'r') as file:
            lines = [line.strip().split(' ') for line in file.readlines()]
            
        # Iterate through each Item element
        for line in tqdm(lines, desc='Splitting train images'):
            vehicle_id = line[0].split('/')[0]
            image_name = line[0].split('/')[1]
            full_image_path = os.path.join(img_dir, vehicle_id, image_name)
            
            # Here we map the original vehicle ID to a new ID
            if vehicle_id not in original_to_new_id:
                new_id = str2int(str(new_id_counter), dataset)

                original_to_new_id[vehicle_id] = new_id
                new_id_counter += 1
            
            new_vehicle_id = original_to_new_id[vehicle_id]
            class_images[new_vehicle_id].append(full_image_path)
    elif (dataset == 'vehicleid'):
        with open(train_file, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        
        # Iterate through each Item element
        for line in tqdm(lines, desc='Splitting train images'):
            image_name = line.split(' ')[0]
            vehicle_id = line.split(' ')[1]
            full_image_path = os.path.join(img_dir, image_name + '.jpg')
            
            # Here we map the original vehicle ID to a new ID
            if vehicle_id not in original_to_new_id:
                new_id = str2int(str(new_id_counter), dataset)

                original_to_new_id[vehicle_id] = new_id
                new_id_counter += 1
            
            new_vehicle_id = original_to_new_id[vehicle_id]
            class_images[new_vehicle_id].append(full_image_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Return both the dictionary and the mapping from original to new IDs
    return class_images, original_to_new_id

# ========================== MAIN ========================== #
# Set up paths
dataset = 'veri' # 'veri' (Which is: VeRi-776) / 'veriwild' / 'vehicleid'
base_datapath = 'data'
gms_path = 'gms'
image_size = (224, 224) # Before computing GMS matches, resize the images to this size (as per paper)
verbose = False # Set to True to see more detailed output, errors etc.

if (dataset == 'veri'):
    data_path = os.path.join(base_datapath, 'veri')
    img_dir = os.path.join(data_path, 'image_train')
    train_file = os.path.join(data_path, 'train_label.xml')
elif (dataset == 'veriwild'):
    data_path = os.path.join(base_datapath, 'veriwild')
    img_dir = os.path.join(data_path, 'images')
    train_file = os.path.join(data_path, 'train_test_split', 'train_list_start0.txt')
elif (dataset == 'vehicleid'):
    data_path = os.path.join(base_datapath, 'vehicleid')
    img_dir = os.path.join(data_path, 'image')
    train_file = os.path.join(data_path, 'train_test_split', 'train_list.txt')
else:
    raise ValueError(f"Unknown dataset: {dataset}")

output = os.path.join(gms_path, dataset)
if (os.path.exists(output) == False):
    os.makedirs(output)
    if verbose:
        print(f"Output directory created at: {output}")

# Instantiate the ORB and BFMatcher objects
orb = cv2.ORB_create(nfeatures = 10000, fastThreshold = 0)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

# Get the dictionary of class images
# It will contain keys as class labels and values as lists of image paths
# Example:
# {
#     '1': ['<full_path>/190472.jpg', '<full_path>/134671.jpg', ...],
#     '2': ['<full_path>/134718.jpg', '<full_path>/824511.jpg', ...],
#    ...
# }
class_images, id_mapping = get_dict(dataset, train_file, img_dir)

# # In case you want to filter the dictionary to start from a certain class (resuming from a checkpoint, basically)
# resuming_class = 5344
# class_images = {k: v for k, v in class_images.items() if int(k) >= resuming_class}

# Create the index_vp.pkl file
# dict_index should contain the name of the images as keys, and a tuple (class_label, counter) as values
dict_index = {os.path.basename(image): (class_label, counter)
              for class_label, images in class_images.items()
              for counter, image in enumerate(images)}

with open(os.path.join(output, f'index_vp_{dataset}.pkl'), 'wb') as f:
    pickle.dump(dict_index, f)
if verbose:
    print("Successfully saved the Index Pickle file.")

# Get how many iterations are needed (for tqdm)
total_iterations = sum([len(images) for images in class_images.values()])

# Process each class
with tqdm(total=total_iterations, desc="Processing pickle files") as pbar:
    for class_label, images in class_images.items():
        print(f"Processing class {class_label} with {len(images)} images")
        adj_matrix = process_class(images, image_size=image_size, verbose=verbose)
        
        # Save the adjacency matrix
        with open(os.path.join(output, f'{class_label}.pkl'), 'wb') as f:
            pickle.dump(adj_matrix, f)

    print("Processing complete. Adjacency matrices saved.")
# ========================================================== #