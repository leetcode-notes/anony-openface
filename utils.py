import os
import comet_ml
import sys
import math
import torch
import random
import pyprind
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
from scipy.stats import pearsonr
from itertools import combinations
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class MeanAndStd:
    """
    Class to find mean and standard deviation for custom datasets
    """

    def __init__(self, data_path):
        """
        Constructor that receives paths to dataset folder
        :param data_path: path to dataset path
        """
        self.data_path = data_path
        self.val_computer()

    def val_computer(self):
        """
        Function that computes mean and standard deviation
        :return: mean, standard deviation over given dataset
        """
        transform = transforms.Compose(
            [
                transforms.CenterCrop(1024),
                transforms.Resize(224),
                transforms.Grayscale(3),
                transforms.ToTensor(),
            ]
        )

        data_ = torchvision.datasets.ImageFolder(root=self.data_path, transform=transform)
        data_loader = data.DataLoader(data_, batch_size=10)

        mean = 0.0
        std = 0.0
        for images, _ in data_loader:
            batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)

        var = 0.0
        for images, _ in data_loader:
            batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])

        mean /= len(data_)
        std = torch.sqrt(var / (len(data_) * 224 * 224))
        return mean, std


class BP4D:
    """
    This class is responsible for pre-processing the BP4D dataset
    """

    def __init__(self, data_path, f_range=1, m_range=0):
        """
        Constructor for the BP4D class
        :param data_path: file path to the location of the dataset
        :param f_range: number of female participants you wish to load
        :param m_range: number of male participants you wish to load
        """
        self.data_path = data_path
        self.f_range = f_range + 1
        self.m_range = m_range + 1

        self.processed_dataset = open("test_BP4D_subjects.txt", 'w')

    def processor(self):
        """
        Function that performs the pre-processing for the dataset

        The function works by first finding the optimal image frames of every participant from their
        corresponding .csv files.

        It then visits each participant's expression sub-folder and finds the file path

        The file paths are subsequently appended with their sub-folder names and their AU occurrence data
        and written to the txt file created by self.processed_dataset
        :return: N/A
        """
        print ("Pre-processing BP4D...!!!")

        for i in range(67, self.f_range):
            # working on subfolder T1
            dataframe = pd.read_csv(self.data_path + "AUCoding/AU_OCC/F" + str(format(i, '03d')) + "_T1.csv")
            for _, row in dataframe.iterrows():
                row = list(row)
                frame = row.pop(0)
                row = [0 if x == 9 else x for x in row]
                line = self.data_path + "2D+3D/F" + str(format(i, '03d')) + "/T1/" + str(
                    format(frame, '04d')) + ".jpg/" + "".join(str(e) for e in row) + "\n"
                index = line.index(".jpg/")
                line = list(line)
                line.pop(index + 4 + 35)
                line.pop(index + 4 + 34)
                line.pop(index + 4 + 33)
                line.pop(index + 4 + 32)
                line.pop(index + 4 + 31)
                line.pop(index + 4 + 30)
                line.pop(index + 4 + 29)
                line.pop(index + 4 + 28)
                line.pop(index + 4 + 27)
                line.pop(index + 4 + 26)
                line.pop(index + 4 + 25)
                line.pop(index + 4 + 24)
                line.pop(index + 4 + 23)
                line.pop(index + 4 + 22)
                line.pop(index + 4 + 21)
                line.pop(index + 4 + 19)
                line.pop(index + 4 + 18)
                line.pop(index + 4 + 17)
                line.pop(index + 4 + 16)
                line.pop(index + 4 + 14)
                line.pop(index + 4 + 11)
                line.pop(index + 4 + 9)
                line.pop(index + 4 + 7)
                line.pop(index + 4 + 4)
                self.processed_dataset.write("".join(line))

            # working on subfolder T6
            dataframe = pd.read_csv(self.data_path + "AUCoding/AU_OCC/F" + str(format(i, '03d')) + "_T6.csv")
            for _, row in dataframe.iterrows():
                row = list(row)
                frame = row.pop(0)
                row = [0 if x == 9 else x for x in row]
                line = self.data_path + "2D+3D/F" + str(format(i, '03d')) + "/T6/" + str(
                    format(frame, '04d')) + ".jpg/" + "".join(str(e) for e in row) + "\n"
                index = line.index(".jpg/")
                line = list(line)
                line.pop(index + 4 + 35)
                line.pop(index + 4 + 34)
                line.pop(index + 4 + 33)
                line.pop(index + 4 + 32)
                line.pop(index + 4 + 31)
                line.pop(index + 4 + 30)
                line.pop(index + 4 + 29)
                line.pop(index + 4 + 28)
                line.pop(index + 4 + 27)
                line.pop(index + 4 + 26)
                line.pop(index + 4 + 25)
                line.pop(index + 4 + 24)
                line.pop(index + 4 + 23)
                line.pop(index + 4 + 22)
                line.pop(index + 4 + 21)
                line.pop(index + 4 + 19)
                line.pop(index + 4 + 18)
                line.pop(index + 4 + 17)
                line.pop(index + 4 + 16)
                line.pop(index + 4 + 14)
                line.pop(index + 4 + 11)
                line.pop(index + 4 + 9)
                line.pop(index + 4 + 7)
                line.pop(index + 4 + 4)
                self.processed_dataset.write("".join(line))

            # working on subfolder T7
            dataframe = pd.read_csv(self.data_path + "AUCoding/AU_OCC/F" + str(format(i, '03d')) + "_T7.csv")
            for _, row in dataframe.iterrows():
                row = list(row)
                frame = row.pop(0)
                row = [0 if x == 9 else x for x in row]
                line = self.data_path + "2D+3D/F" + str(format(i, '03d')) + "/T7/" + str(
                    format(frame, '04d')) + ".jpg/" + "".join(str(e) for e in row) + "\n"
                index = line.index(".jpg/")
                line = list(line)
                line.pop(index + 4 + 35)
                line.pop(index + 4 + 34)
                line.pop(index + 4 + 33)
                line.pop(index + 4 + 32)
                line.pop(index + 4 + 31)
                line.pop(index + 4 + 30)
                line.pop(index + 4 + 29)
                line.pop(index + 4 + 28)
                line.pop(index + 4 + 27)
                line.pop(index + 4 + 26)
                line.pop(index + 4 + 25)
                line.pop(index + 4 + 24)
                line.pop(index + 4 + 23)
                line.pop(index + 4 + 22)
                line.pop(index + 4 + 21)
                line.pop(index + 4 + 19)
                line.pop(index + 4 + 18)
                line.pop(index + 4 + 17)
                line.pop(index + 4 + 16)
                line.pop(index + 4 + 14)
                line.pop(index + 4 + 11)
                line.pop(index + 4 + 9)
                line.pop(index + 4 + 7)
                line.pop(index + 4 + 4)
                self.processed_dataset.write("".join(line))

            # working on subfolder T8
            dataframe = pd.read_csv(self.data_path + "AUCoding/AU_OCC/F" + str(format(i, '03d')) + "_T8.csv")
            for _, row in dataframe.iterrows():
                row = list(row)
                frame = row.pop(0)
                row = [0 if x == 9 else x for x in row]
                line = self.data_path + "2D+3D/F" + str(format(i, '03d')) + "/T8/" + str(
                    format(frame, '03d')) + ".jpg/" + "".join(str(e) for e in row) + "\n"
                index = line.index(".jpg/")
                line = list(line)
                line.pop(index + 4 + 35)
                line.pop(index + 4 + 34)
                line.pop(index + 4 + 33)
                line.pop(index + 4 + 32)
                line.pop(index + 4 + 31)
                line.pop(index + 4 + 30)
                line.pop(index + 4 + 29)
                line.pop(index + 4 + 28)
                line.pop(index + 4 + 27)
                line.pop(index + 4 + 26)
                line.pop(index + 4 + 25)
                line.pop(index + 4 + 24)
                line.pop(index + 4 + 23)
                line.pop(index + 4 + 22)
                line.pop(index + 4 + 21)
                line.pop(index + 4 + 19)
                line.pop(index + 4 + 18)
                line.pop(index + 4 + 17)
                line.pop(index + 4 + 16)
                line.pop(index + 4 + 14)
                line.pop(index + 4 + 11)
                line.pop(index + 4 + 9)
                line.pop(index + 4 + 7)
                line.pop(index + 4 + 4)
                self.processed_dataset.write("".join(line))

        for i in range(48, self.m_range):
            # working on subfolder T1
            dataframe = pd.read_csv(self.data_path + "AUCoding/AU_OCC/M" + str(format(i, '03d')) + "_T1.csv")
            for _, row in dataframe.iterrows():
                row = list(row)
                frame = row.pop(0)
                row = [0 if x == 9 else x for x in row]
                line = self.data_path + "2D+3D/M" + str(format(i, '03d')) + "/T1/" + str(
                    format(frame, '04d')) + ".jpg/" + "".join(str(e) for e in row) + "\n"
                index = line.index(".jpg/")
                line = list(line)
                line.pop(index + 4 + 35)
                line.pop(index + 4 + 34)
                line.pop(index + 4 + 33)
                line.pop(index + 4 + 32)
                line.pop(index + 4 + 31)
                line.pop(index + 4 + 30)
                line.pop(index + 4 + 29)
                line.pop(index + 4 + 28)
                line.pop(index + 4 + 27)
                line.pop(index + 4 + 26)
                line.pop(index + 4 + 25)
                line.pop(index + 4 + 24)
                line.pop(index + 4 + 23)
                line.pop(index + 4 + 22)
                line.pop(index + 4 + 21)
                line.pop(index + 4 + 19)
                line.pop(index + 4 + 18)
                line.pop(index + 4 + 17)
                line.pop(index + 4 + 16)
                line.pop(index + 4 + 14)
                line.pop(index + 4 + 11)
                line.pop(index + 4 + 9)
                line.pop(index + 4 + 7)
                line.pop(index + 4 + 4)
                self.processed_dataset.write("".join(line))

            # working on subfolder T6
            dataframe = pd.read_csv(self.data_path + "AUCoding/AU_OCC/M" + str(format(i, '03d')) + "_T6.csv")
            for _, row in dataframe.iterrows():
                row = list(row)
                frame = row.pop(0)
                row = [0 if x == 9 else x for x in row]
                line = self.data_path + "2D+3D/M" + str(format(i, '03d')) + "/T6/" + str(
                    format(frame, '04d')) + ".jpg/" + "".join(str(e) for e in row) + "\n"
                index = line.index(".jpg/")
                line = list(line)
                line.pop(index + 4 + 35)
                line.pop(index + 4 + 34)
                line.pop(index + 4 + 33)
                line.pop(index + 4 + 32)
                line.pop(index + 4 + 31)
                line.pop(index + 4 + 30)
                line.pop(index + 4 + 29)
                line.pop(index + 4 + 28)
                line.pop(index + 4 + 27)
                line.pop(index + 4 + 26)
                line.pop(index + 4 + 25)
                line.pop(index + 4 + 24)
                line.pop(index + 4 + 23)
                line.pop(index + 4 + 22)
                line.pop(index + 4 + 21)
                line.pop(index + 4 + 19)
                line.pop(index + 4 + 18)
                line.pop(index + 4 + 17)
                line.pop(index + 4 + 16)
                line.pop(index + 4 + 14)
                line.pop(index + 4 + 11)
                line.pop(index + 4 + 9)
                line.pop(index + 4 + 7)
                line.pop(index + 4 + 4)
                self.processed_dataset.write("".join(line))

            # working on subfolder T7
            dataframe = pd.read_csv(self.data_path + "AUCoding/AU_OCC/M" + str(format(i, '03d')) + "_T7.csv")
            for _, row in dataframe.iterrows():
                row = list(row)
                frame = row.pop(0)
                row = [0 if x == 9 else x for x in row]
                line = self.data_path + "2D+3D/M" + str(format(i, '03d')) + "/T7/" + str(
                    format(frame, '04d')) + ".jpg/" + "".join(str(e) for e in row) + "\n"
                index = line.index(".jpg/")
                line = list(line)
                line.pop(index + 4 + 35)
                line.pop(index + 4 + 34)
                line.pop(index + 4 + 33)
                line.pop(index + 4 + 32)
                line.pop(index + 4 + 31)
                line.pop(index + 4 + 30)
                line.pop(index + 4 + 29)
                line.pop(index + 4 + 28)
                line.pop(index + 4 + 27)
                line.pop(index + 4 + 26)
                line.pop(index + 4 + 25)
                line.pop(index + 4 + 24)
                line.pop(index + 4 + 23)
                line.pop(index + 4 + 22)
                line.pop(index + 4 + 21)
                line.pop(index + 4 + 19)
                line.pop(index + 4 + 18)
                line.pop(index + 4 + 17)
                line.pop(index + 4 + 16)
                line.pop(index + 4 + 14)
                line.pop(index + 4 + 11)
                line.pop(index + 4 + 9)
                line.pop(index + 4 + 7)
                line.pop(index + 4 + 4)
                self.processed_dataset.write("".join(line))

            # working on subfolder T8
            dataframe = pd.read_csv(self.data_path + "AUCoding/AU_OCC/M" + str(format(i, '03d')) + "_T8.csv")
            for _, row in dataframe.iterrows():
                row = list(row)
                frame = row.pop(0)
                row = [0 if x == 9 else x for x in row]
                line = self.data_path + "2D+3D/M" + str(format(i, '03d')) + "/T8/" + str(
                    format(frame, '03d')) + ".jpg/" + "".join(str(e) for e in row) + "\n"
                index = line.index(".jpg/")
                line = list(line)
                line.pop(index + 4 + 35)
                line.pop(index + 4 + 34)
                line.pop(index + 4 + 33)
                line.pop(index + 4 + 32)
                line.pop(index + 4 + 31)
                line.pop(index + 4 + 30)
                line.pop(index + 4 + 29)
                line.pop(index + 4 + 28)
                line.pop(index + 4 + 27)
                line.pop(index + 4 + 26)
                line.pop(index + 4 + 25)
                line.pop(index + 4 + 24)
                line.pop(index + 4 + 23)
                line.pop(index + 4 + 22)
                line.pop(index + 4 + 21)
                line.pop(index + 4 + 19)
                line.pop(index + 4 + 18)
                line.pop(index + 4 + 17)
                line.pop(index + 4 + 16)
                line.pop(index + 4 + 14)
                line.pop(index + 4 + 11)
                line.pop(index + 4 + 9)
                line.pop(index + 4 + 7)
                line.pop(index + 4 + 4)
                self.processed_dataset.write("".join(line))
        self.processed_dataset.close()

        # randomizing the copied file paths and creating the final processed file
        with open("test_BP4D_subjects.txt", 'r') as source:
            data = [(random.random(), line) for line in source]
        data.sort()
        with open('test_pro_BP4D_subjects.txt', 'w') as target:
            for _, line in data:
                target.write(line)
        os.remove("test_BP4D_subjects.txt")
        print ("Done...!!!")

    def split_train_test_val(self, train_per=0.7, test_per=0.3):
        """
        This function creates the train and test splits of the processed dataset
        :return: N/A
        """
        src_file = open("processed_BP4D_subjects.txt", 'r')
        train_file = open("train_BP4D_subjects.txt", "w")
        test_file = open("test_BP4D_subjects.txt", "w")
        file_length = src_file.readlines()
        train_len = int(train_per * len(file_length))
        test_len = int(test_per * len(file_length))
        for i in range(train_len):
            train_file.write(file_length[i])
        for j in range(train_len, test_len + train_len):
            test_file.write(file_length[j])
        print ("Split complete...!!!")


def m_and_f_sort(train_per=0.7, test_per=0.3):
    """
    This function creates splits by male and female subjects
    :return: N/A
    """
    src_file = open("processed_BP4D.txt", 'r')
    male_file = open("male_BP4D.txt", 'w')
    female_file = open("female_BP4D.txt", 'w')

    file_paths = src_file.readlines()
    print ("Sorting by male and female subjects...!!!")
    for i in range(len(file_paths)):
        if "M" in file_paths[i]:
            male_file.write(file_paths[i])
        elif "F" in file_paths[i]:
            female_file.write(file_paths[i])

    male_file = open("male_BP4D.txt", 'r')
    female_file = open("female_BP4D.txt", 'r')

    train_m_file = open("train_male_BP4D.txt", "w")
    test_m_file = open("test_male_BP4D.txt", "w")
    male_file_length = male_file.readlines()

    train_f_file = open("train_female_BP4D.txt", "w")
    test_f_file = open("test_female_BP4D.txt", "w")
    female_file_length = female_file.readlines()

    train_len = int(train_per * len(male_file_length))
    test_len = int(test_per * len(male_file_length))
    for i in range(train_len):
        train_m_file.write(male_file_length[i])
    for j in range(train_len, test_len + train_len):
        test_m_file.write(male_file_length[j])

    train_len = int(train_per * len(female_file_length))
    test_len = int(test_per * len(female_file_length))
    for i in range(train_len):
        train_f_file.write(female_file_length[i])
    for j in range(train_len, test_len + train_len):
        test_f_file.write(female_file_length[j])

    print ("Split complete...!!!")


class AUMean:
    """
    This class calculates the mean probability of every Action Unit
    """

    def __init__(self, data_path):
        """
        Constructor for AUMean
        :param data_path: file path to processed BP4D dataset
        """
        self.data_path = data_path
        self.calc_mean()

    def calc_mean(self):
        """
        Function that calculates the mean of every AU
        :return: mean: list containing the mean of every probability
        """
        index = 0
        mean = []
        with open(self.data_path, 'r') as src:
            readlines = src.readlines()
            for i in range(11):
                sum = 0.0
                for j in range(len(readlines)):
                    sum += float(readlines[j][readlines[j].index(".jpg/") + 5 + index])
                mean.append(sum / len(readlines))
                index += 1
        return mean


class DataLoader:
    """
    Custom dataloader for the BP4D dataset that works with the .txt files containing the file paths to
    every image in the dataset
    """

    def __init__(self, data_path, batch_size, mean=[0.1732, 0.1732, 0.1732], std=[0.1505, 0.1505, 0.1505]):
        """
        Constructor for the DataLoader class
        :param data_path: file path to the .txt file containing the image paths
        :param batch_size: batch size of every image batch
        :param mean: mean values for the normalization transform
        :param std: standard deviation values for the normalization transform
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.mean = mean
        self.std = std

    def loaded_data(self):
        """
        Function that prepares the batches
        :return: data: a list containing all the batches. Each element in the list is a tuple of the following:
                    >> batch of images
                    >> batch of AU indexes
                    >> batch of expression indexes
        """
        data = list()
        batch = 0
        with open(self.data_path, 'r') as src:
            print ("Loading dataset...!!!")
            readlines = src.readlines()
            bar = pyprind.ProgBar(len(readlines), stream=sys.stdout)
            batch_tensor = torch.Tensor(self.batch_size)
            truth_tensor = torch.Tensor(self.batch_size)
            label_tensor = torch.Tensor(self.batch_size)
            for image in range(len(readlines)):
                try:
                    img = Image.open(readlines[image][:readlines[image].index(".jpg/") + 4])
                except FileNotFoundError:
                    continue
                img = F.crop(img, 0, 0, 1250, 1024)
                img = F.resize(img, (224, 224))
                img = F.to_grayscale(img, 3)
                img = F.to_tensor(img)
                img = F.normalize(img, mean=self.mean, std=self.std)
                truth = list(map(int, list(readlines[image][
                                           readlines[image].index(".jpg/") + 5: readlines[image].index("\n")])))
                truth = torch.from_numpy(np.array(truth))
                label = torch.from_numpy(np.array(
                    float(readlines[image][readlines[image].index("/T") + 2: readlines[image].index("/T") + 3]))
                )
                if batch == 0:
                    batch_tensor = img
                    truth_tensor = truth
                    label_tensor = label
                    batch += 1
                elif batch == 1:
                    batch_tensor = torch.cat((batch_tensor.unsqueeze(0), img.unsqueeze(0)), dim=0)
                    truth_tensor = torch.cat((truth_tensor.unsqueeze(0), truth.unsqueeze(0)), dim=0)
                    label_tensor = torch.cat((label_tensor.unsqueeze(0), label.unsqueeze(0)), dim=0)
                    batch += 1
                elif 1 < batch < self.batch_size:
                    batch_tensor = torch.cat((batch_tensor, img.unsqueeze(0)), dim=0)
                    truth_tensor = torch.cat((truth_tensor, truth.unsqueeze(0)), dim=0)
                    label_tensor = torch.cat((label_tensor, label.unsqueeze(0)), dim=0)
                    batch += 1
                else:
                    data.append((batch_tensor, truth_tensor, label_tensor))
                    batch = 1
                    batch_tensor = img
                    truth_tensor = truth
                    label_tensor = label
                bar.update()
            print ("Done...!!!")
        return data


class DataLoader2:
    """
        Custom dataloader for the BP4D dataset that works with the .txt files containing the file paths to
        every image in the dataset
        """

    def __init__(self, data_path, index, batch_size, mean=[0.1732, 0.1732, 0.1732], std=[0.1505, 0.1505, 0.1505]):
        """
        Constructor for the DataLoader class
        :param data_path: file path to the .txt file containing the image paths
        :param batch_size: batch size of every image batch
        :param mean: mean values for the normalization transform
        :param std: standard deviation values for the normalization transform
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.index = index
        self.mean = mean
        self.std = std

    def loaded_data(self):
        """
        Function that prepares the batches
        :return: data: a list containing all the batches. Each element in the list is a tuple of the following:
                    >> batch of images
                    >> batch of AU indexes
                    >> batch of expression indexes
        """
        data = list()
        batch = 0
        with open(self.data_path, 'r') as src:
            readlines = src.readlines()
            batch_tensor = torch.Tensor(self.batch_size)
            truth_tensor = torch.Tensor(self.batch_size)
            label_tensor = torch.Tensor(self.batch_size)
            counter = 0
            image = self.index
            if image >= len(readlines) - self.batch_size:
                image = 0
            while len(readlines) > 0:
                if counter == self.batch_size + 1:
                    break
                try:
                    img = Image.open(readlines[image][:readlines[image].index(".jpg/") + 4])
                    counter += 1
                except FileNotFoundError:
                    image += 1
                    if image == len(readlines):
                        image = 0
                    continue
                img = F.crop(img, 0, 0, 1250, 1024)
                img = F.resize(img, (224, 224))
                img = F.to_grayscale(img, 3)
                img = F.to_tensor(img)
                img = F.normalize(img, mean=self.mean, std=self.std)
                truth = list(map(int, list(readlines[image][
                                           readlines[image].index(".jpg/") + 5: readlines[image].index("\n")])))

                truth = torch.from_numpy(np.array(truth)).float()
                label = torch.from_numpy(np.array(
                    float(readlines[image][readlines[image].index("/T") + 2: readlines[image].index("/T") + 3]))
                )
                if batch == 0:
                    batch_tensor = img
                    truth_tensor = truth
                    label_tensor = label
                    batch += 1
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
                elif batch == 1:
                    batch_tensor = torch.cat((batch_tensor.unsqueeze(0), img.unsqueeze(0)), dim=0)
                    truth_tensor = torch.cat((truth_tensor.unsqueeze(0), truth.unsqueeze(0)), dim=0)
                    label_tensor = torch.cat((label_tensor.unsqueeze(0), label.unsqueeze(0)), dim=0)
                    batch += 1
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
                elif 1 < batch < self.batch_size:
                    batch_tensor = torch.cat((batch_tensor, img.unsqueeze(0)), dim=0)
                    truth_tensor = torch.cat((truth_tensor, truth.unsqueeze(0)), dim=0)
                    label_tensor = torch.cat((label_tensor, label.unsqueeze(0)), dim=0)
                    batch += 1
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
                else:
                    data.append((batch_tensor, truth_tensor, label_tensor))
                    batch = 1
                    batch_tensor = img
                    truth_tensor = truth
                    label_tensor = label
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
        return data


class Correlate:
    """
    This class finds the correlations between the different Action Unit pairs
    """

    def __init__(self, batch_size):
        self.data_path = "train_BP4D.txt"
        self.batch_size = batch_size
        self.au_index = [x for x in range(35)]
        self.au_index.pop(34)
        self.au_index.pop(33)
        self.au_index.pop(32)
        self.au_index.pop(31)
        self.au_index.pop(30)
        self.au_index.pop(29)
        self.au_index.pop(28)
        self.au_index.pop(27)
        self.au_index.pop(26)
        self.au_index.pop(25)
        self.au_index.pop(24)
        self.au_index.pop(23)
        self.au_index.pop(22)
        self.au_index.pop(21)
        self.au_index.pop(20)
        self.au_index.pop(18)
        self.au_index.pop(17)
        self.au_index.pop(16)
        self.au_index.pop(15)
        self.au_index.pop(13)
        self.au_index.pop(10)
        self.au_index.pop(8)
        self.au_index.pop(6)
        self.au_index.pop(3)
        self.comb = list(combinations(self.au_index, 2))

    def correlate(self):
        """
        This function finds the correlations between Action Units by finding all possible pair combinations
        of Action Units
        The resultant value is stored in a .pt file in order to save time while using this value
        However the function can be called to return a value
        Additionally this function returns class weights
        :return: N/A
        """
        class_weights = torch.zeros(11)
        corr_list = list()
        with open(self.data_path) as src:
            readlines = src.readlines()
            for i in range(len(self.au_index)):
                x = list()
                for line in readlines:
                    x.append(int(line[line.index(".jpg/") + 5 + i][0]))
                sum = 0
                neg_sum = 0
                for j in x:
                    if j == 1:
                        sum += 1
                    else:
                        neg_sum += 1
                class_weights[i] = float(neg_sum / sum)
                print ("Action Unit " + str(self.au_index[i] + 1) + "\tNumber of positives=" + str(sum) + "\tTotal\
                length=" + str(len(x)) + "\tClass Weight=" + str(class_weights[i]))
            # print (class_weights.shape)
            torch.save(class_weights, 'class_weight.pt')
            # exit(0)
            for au in self.comb:
                try:
                    x = list()
                    y = list()
                    for line in readlines:
                        # print (line)
                        x.append(int(line[line.index(".jpg/") + 5 + au[0]][0]))
                        y.append(int(line[line.index(".jpg/") + 5 + au[1]][0]))

                    corr = pearsonr(np.array(x), np.array(y))[0]
                    if math.isnan(corr) is True:
                        corr = 0.009621640951724951
                        print ("found")
                        # exit(0)
                    print (corr)
                    corr_list.append(corr)
                except:
                    corr = 0.009621640951724951
                    corr_list.append(corr)
        corr_list = np.array(corr_list)
        print (corr_list.size)
        corr_ten = torch.from_numpy(corr_list).reshape(11, 5)
        corr_tensor = corr_ten.unsqueeze(0)
        for batch in range(self.batch_size - 1):
            corr_tensor = torch.cat((corr_tensor, corr_ten.unsqueeze(0)), dim=0)
        print (corr_tensor.shape)
        torch.save(corr_tensor.float(), "corr_tensor.npy")
        # torch.save(class_weights, 'class_weights.pt')


class DataSampler:
    """
    This class creates a random sample of the dataset
    """

    def __init__(self, sampler_per, data_path):
        self.sampler_per = sampler_per
        self.data_path = data_path
        self.sampled_dataset = open("sampled.txt", 'w')
        self.sampler()

    def sampler(self):
        with open(self.data_path) as source:
            lines = [(random.random(), line) for line in source]
        lines.sort()
        limit = int(len(lines) * self.sampler_per)
        count = 0
        for _, line in lines:
            if count == limit:
                break
            else:
                self.sampled_dataset.write(line)
                count += 1
