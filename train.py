import os
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import cv2

HOME = os.getcwd()
main_path = f"{HOME}/Data/"

def get_file(class_name, idx):
    directory = os.listdir(main_path + class_name)
    file_list = []
    dim1, dim2 = [], []
    images = []
    label = []
    for d in directory:
        file_list.append(os.path.join(main_path + class_name + '/' + d))

    for img_path in file_list:
        img = Image.open(img_path)
        img = img.resize((28, 68))
        img = np.array(img) / 255
        images.append(img)
        label.append(idx)
        
    return images, label

def get_data(data1, label1, data2, label2):
    X1 = np.array(data1)
    Y1 = np.array(label1)
    
    X2 = np.array(data2)
    Y2 = np.array(label2)
    
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    return X, Y

def main():
    empty_list, empty_label = get_file('empty', 0)
    not_empty_list, not_empty_label = get_file('not_empty', 1)

    train_data, label_data = get_data(
        empty_list,
        empty_label,
        not_empty_list,
        not_empty_label)

    n_samples = train_data.shape[0]
    X_train_flat = train_data.reshape(n_samples, -1)

    X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(
        X_train_flat, label_data, test_size=0.2, random_state=42
    )

    # model = RandomForestClassifier(n_estimators=100, random_state=42)

    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_split, Y_train_split)

    with open(f'{HOME}/Model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
