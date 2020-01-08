# from comet_ml import Experiment as comex
import os
import cv2
import torch
import ffmpy
from tqdm import tqdm
import base_model
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import torch.optim as optim
import torchvision.transforms.functional as F


class Bosphorous:
    """
    This class is responsible for pre-processing the Bosphorous dataset
    """

    def __init__(self):
        self.data_path = ""
        self.index = 0
        self.batch_size = 0
        self.mean = []
        self.std = []

    def processor(self):
        """
        Function that performs the pre-processing for the dataset
        :return: N/A
        """
        print("Pre-processing Bosphorous...!!!")
        processed_dataset = open("pro_Bosphorous.txt", 'w')
        for i in range(0, 105):
            path = self.data_path + "bs" + str(format(i, '03d')) + "/"
            files = os.listdir(path)
            for file in files:
                if file.endswith(".png"):
                    if "CAU" in file:
                        au = ""
                        if "A12" in file:
                            au = au + "_12"
                        if "A15" in file:
                            au = au + "_15"
                        if au is not "":
                            processed_dataset.write(path + file + "/" + au + "\n")
                        continue
                    elif "UFAU" in file:
                        au = ""
                        if "_2" in file:
                            au = au + "_2"
                        if "_4" in file:
                            au = au + "_4"
                        if "_1" in file:
                            au = au + "_1"
                        if au is not "":
                            processed_dataset.write(path + file + "/" + au + "\n")
                        continue
                    elif "LFAU" in file:
                        au = ""
                        if "_12" in file:
                            au = au + "_12"
                        if "_14" in file:
                            au = au + "_14"
                        if "_15" in file:
                            au = au + "_15"
                        if "_17" in file:
                            au = au + "_17"
                        if "_23" in file:
                            au = au + "_23"
                        if "_10" in file:
                            au = au + "_10"
                        if au is not "":
                            processed_dataset.write(path + file + "/" + au + "\n")
                        continue

    def evaluate_loader(self, data_path="/projects/dataset_processed/Bosphorus/BosphorusDB/BosphorusDB/",
                        batch_size=8, index=0, mean=[0.1732, 0.1732, 0.1732], std=[0.1505, 0.1505, 0.1505]):
        self.data_path = data_path
        self.index = index
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        data = list()
        batch = 0
        AU_labels = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23"]
        with open("pro_Bosphorous.txt", 'r') as src:
            readlines = src.readlines()
            batch_tensor = torch.Tensor(self.batch_size)
            truth_tensor = torch.Tensor(self.batch_size)
            counter = 0
            image = self.index
            if image >= len(readlines) - self.batch_size:
                image = 0
            while len(readlines) > 0:
                if counter == self.batch_size + 1:
                    break
                try:
                    img = Image.open(readlines[image][:readlines[image].index(".png/") + 4])
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
                truth = [0 for _ in range(11)]
                aus = readlines[image][readlines[image].index(".png/") + 5: readlines[image].index("\n")].split("_")
                for au in aus:
                    if au.isdigit():
                        truth[AU_labels.index(au)] = 1
                truth = torch.from_numpy(np.array(truth))
                if batch == 0:
                    batch_tensor = img
                    truth_tensor = truth
                    batch += 1
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
                elif batch == 1:
                    batch_tensor = torch.cat((batch_tensor.unsqueeze(0), img.unsqueeze(0)), dim=0)
                    truth_tensor = torch.cat((truth_tensor.unsqueeze(0), truth.unsqueeze(0)), dim=0)
                    batch += 1
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
                elif 1 < batch < self.batch_size:
                    batch_tensor = torch.cat((batch_tensor, img.unsqueeze(0)), dim=0)
                    truth_tensor = torch.cat((truth_tensor, truth.unsqueeze(0)), dim=0)
                    batch += 1
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
                else:
                    data.append((batch_tensor, truth_tensor))
                    batch = 1
                    batch_tensor = img
                    truth_tensor = truth
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
        return data

    def train_cnn(self, au_net, config, model_save_path="EmoNet.pth", train_path="train_Bosphorous.txt",
                  test_path="test_Bosphorous.txt", loss_weight=0.65):
        EPOCHS = config.num_epochs
        BATCH_SIZE = config.batch_size
        LEARNING_RATE = config.lr
        TRAIN_DATA_PATH = train_path
        TEST_DATA_PATH = test_path

        print ("Selected params: \nEpochs %d \nBatch Size %d \nLearning Rate %.6f \nTrain path %s \nTest path %s"
               % (EPOCHS, BATCH_SIZE, LEARNING_RATE, TRAIN_DATA_PATH, TEST_DATA_PATH))

        # to find means of Action Units
        train_AU_mean = torch.from_numpy(np.array([0.5 for _ in range(11)]))
        test_AU_mean = torch.from_numpy(np.array([0.5 for _ in range(11)]))

        AU_labels = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23"]

        # setting networks to cuda if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        print('device selected: ', device)
        au_net.to(device)

        # class_weights = torch.load("class_weight.pt").to(device)
        AU_criterion = nn.BCELoss()
        corr_net = base_model.CorrCNN().to(device)
        params = list(au_net.parameters()) + list(corr_net.parameters())
        optimizer = optim.Adam(params, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=0.1)

        epoch = 0

        emo_file = model_save_path
        if torch.cuda.is_available() is True:
            model_zoo_ = torch.load(emo_file)
        else:
            model_zoo_ = torch.load(emo_file, map_location='cpu')
        model_zoo_ = {k.replace('model.', ''): v for k, v in model_zoo_.items()}
        model_zoo_2 = dict()
        for k, v in model_zoo_.items():
            if "classifier" not in k:
                model_zoo_2.update({k: v})

        # Setting empty weights to custom classifier for AUnet for 11 classes
        model_zoo_2.update({"classifier.0.weight": model_zoo_["classifier.0.weight"]})
        model_zoo_2.update({"classifier.0.bias": model_zoo_["classifier.0.bias"]})
        model_zoo_2.update({"classifier.3.weight": torch.randn((11, 4096))})
        model_zoo_2.update({"classifier.3.bias": torch.randn(11)})

        au_net.load_state_dict(model_zoo_2)

        # checkpoint = torch.load("BP4D_sub_CNN_28-9-19.pth")
        # au_net.load_state_dict(checkpoint['au_final_state'])
        # corr_net.load_state_dict(checkpoint['corr_final_state'])

        del_cor = torch.load("corr_tensor.npy").to(device)  # delta correlaton factor
        print ("TRAINING STARTS...!!!")
        with open("Bosphorus_CNN_10-10-19.txt", 'w') as src:
            for epoch in range(epoch, EPOCHS):
                running_loss = 0.0
                labels1 = np.array([])
                labels2 = np.array([])
                labels3 = np.array([])
                labels4 = np.array([])
                labels5 = np.array([])
                labels6 = np.array([])
                labels7 = np.array([])
                labels8 = np.array([])
                labels9 = np.array([])
                preds1 = np.array([])
                preds2 = np.array([])
                preds3 = np.array([])
                preds4 = np.array([])
                preds5 = np.array([])
                preds6 = np.array([])
                preds7 = np.array([])
                preds8 = np.array([])
                preds9 = np.array([])
                for i in range(0, len(open(TRAIN_DATA_PATH, 'r').readlines()), BATCH_SIZE):
                    print(epoch + 1, "/", i, "Bosphorus CNN 10-10-19")
                    train_data = self.evaluate_loader(data_path=TRAIN_DATA_PATH, batch_size=BATCH_SIZE, index=i)
                    inputs, truth = train_data[0][0], train_data[0][1]
                    inputs = inputs.to(device)
                    truth = truth.to(device)
                    truth = truth.float()
                    optimizer.zero_grad()
                    outputs = au_net(inputs).to(device)
                    corr_factor = corr_net(inputs).reshape([BATCH_SIZE, 11, 5]).to(device)
                    corr_factor = torch.bmm(corr_factor, corr_factor.permute(0, 2, 1))
                    del_corr = torch.bmm(del_cor, del_cor.permute(0, 2, 1))
                    """
                    loss_2 is the CorrNet loss
                    """
                    loss_2 = torch.Tensor(11, 11).fill_(-1.0).to(device) * (corr_factor + del_corr) * torch.bmm(
                        (outputs - train_AU_mean.float().to(device)).unsqueeze(1),
                        (outputs - train_AU_mean.float().to(device)).unsqueeze(1).permute(0, 2, 1))

                    corr_loss = torch.mean(loss_2[torch.triu(torch.ones(BATCH_SIZE, 11, 11)) == 1]).to(device)

                    print ("corr loss", corr_loss)
                    loss = torch.zeros(1).to(device)
                    for index in range(BATCH_SIZE):
                        loss += AU_criterion(outputs[index][0], truth[index][0])
                        loss += AU_criterion(outputs[index][1], truth[index][1])
                        loss += AU_criterion(outputs[index][2], truth[index][2])
                        loss += AU_criterion(outputs[index][5], truth[index][5])
                        loss += AU_criterion(outputs[index][6], truth[index][6])
                        loss += AU_criterion(outputs[index][7], truth[index][7])
                        loss += AU_criterion(outputs[index][8], truth[index][8])
                        loss += AU_criterion(outputs[index][9], truth[index][9])
                        loss += AU_criterion(outputs[index][10], truth[index][10])
                    loss = loss + 4.0 + loss_weight * corr_loss
                    loss.backward()
                    print ("total loss", loss)
                    optimizer.step()
                    running_loss += loss.item()

                    j = i
                    if j >= len(open(TRAIN_DATA_PATH, 'r').readlines()):
                        j = 0
                    ##########
                    # RUNNING  VALIDATION
                    #########
                    au_net.eval()
                    val_data = self.evaluate_loader(data_path=TEST_DATA_PATH, batch_size=BATCH_SIZE, index=j)
                    val_inputs, val_truth = val_data[0][0], val_data[0][1]

                    val_inputs = val_inputs.to(device)
                    val_truth = val_truth.to(device)
                    val_truth = val_truth.float()
                    val_outputs = au_net(val_inputs).to(device)
                    # Accuracy
                    val_outputs = (val_outputs > 0.5).float()
                    for index in range(BATCH_SIZE):
                        label = np.array(val_truth[index].cpu()).flatten()
                        labels1 = np.append(labels1, label[0])
                        labels2 = np.append(labels2, label[1])
                        labels3 = np.append(labels3, label[2])
                        labels4 = np.append(labels4, label[5])
                        labels5 = np.append(labels5, label[6])
                        labels6 = np.append(labels6, label[7])
                        labels7 = np.append(labels7, label[8])
                        labels8 = np.append(labels8, label[9])
                        labels9 = np.append(labels9, label[10])
                        pred = np.array(val_outputs[index].detach().cpu()).flatten()
                        preds1 = np.append(preds1, pred[0])
                        preds2 = np.append(preds2, pred[1])
                        preds3 = np.append(preds3, pred[2])
                        preds4 = np.append(preds4, pred[5])
                        preds5 = np.append(preds5, pred[6])
                        preds6 = np.append(preds6, pred[7])
                        preds7 = np.append(preds7, pred[8])
                        preds8 = np.append(preds8, pred[9])
                        preds9 = np.append(preds9, pred[10])

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[0], pearsonr(preds1, labels1)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[0],
                                                           f1_score(preds1, labels1, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[1], pearsonr(preds2, labels2)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[1],
                                                           f1_score(preds2, labels2, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[2], pearsonr(preds3, labels3)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[2],
                                                           f1_score(preds3, labels3, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[5], pearsonr(preds4, labels4)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[5],
                                                           f1_score(preds4, labels4, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[6], pearsonr(preds5, labels5)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[6],
                                                           f1_score(preds5, labels5, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[7], pearsonr(preds6, labels6)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[7],
                                                           f1_score(preds6, labels6, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[8], pearsonr(preds7, labels7)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[8],
                                                           f1_score(preds7, labels7, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[9], pearsonr(preds8, labels8)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[9],
                                                           f1_score(preds8, labels8, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[10], pearsonr(preds9, labels9)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[10],
                                                           f1_score(preds9, labels9, average='micro')))

                print("\n\n\n\n")

                torch.save({'epoch': epoch,
                            'au_model_state_dict': au_net.state_dict(),
                            'corr_model_state_dict': corr_net.state_dict()
                            }, 'checkpoint_Bosphorus_CNN_10-10-19.pth')

        print("Finished Training")
        src.close()
        torch.save({'au_final_state': au_net.state_dict(),
                    'corr_final_state': corr_net.state_dict(),
                    }, 'final_Bosphorus_CNN_10-10-19_state.pth')

    def test_cnn(self, au_net, config, model_save_path="EmoNet.pth", test_path="pro_Bosphorous.txt"):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        print('device selected: ', device)
        au_net.to(device)
        BATCH_SIZE = config.batch_size
        TEST_DATA_PATH = test_path
        corr_net = base_model.CorrCNN().to(device)

        emo_file = model_save_path
        if torch.cuda.is_available() is True:
            model_zoo_ = torch.load(emo_file)
        else:
            model_zoo_ = torch.load(emo_file, map_location='cpu')
        model_zoo_ = {k.replace('model.', ''): v for k, v in model_zoo_.items()}
        model_zoo_2 = dict()
        for k, v in model_zoo_.items():
            if "classifier" not in k:
                model_zoo_2.update({k: v})

        # Setting empty weights to custom classifier for AUnet for 11 classes
        model_zoo_2.update({"classifier.0.weight": model_zoo_["classifier.0.weight"]})
        model_zoo_2.update({"classifier.0.bias": model_zoo_["classifier.0.bias"]})
        model_zoo_2.update({"classifier.3.weight": torch.randn((11, 4096))})
        model_zoo_2.update({"classifier.3.bias": torch.randn(11)})

        au_net.load_state_dict(model_zoo_2)

        checkpoint = torch.load("BP4D_sub_CNN_28-9-19.pth")
        au_net.load_state_dict(checkpoint['au_final_state'])
        corr_net.load_state_dict(checkpoint['corr_final_state'])
        au_net.eval()
        labels1 = np.array([])
        labels2 = np.array([])
        labels3 = np.array([])
        labels4 = np.array([])
        labels5 = np.array([])
        labels6 = np.array([])
        labels7 = np.array([])
        labels8 = np.array([])
        labels9 = np.array([])
        preds1 = np.array([])
        preds2 = np.array([])
        preds3 = np.array([])
        preds4 = np.array([])
        preds5 = np.array([])
        preds6 = np.array([])
        preds7 = np.array([])
        preds8 = np.array([])
        preds9 = np.array([])
        for j in tqdm(range(0, len(open(TEST_DATA_PATH, 'r').readlines()), BATCH_SIZE)):
            val_data = self.evaluate_loader(data_path=TEST_DATA_PATH, batch_size=BATCH_SIZE, index=j)
            val_inputs, val_truth = val_data[0][0], val_data[0][1]

            val_inputs = val_inputs.to(device)
            val_truth = val_truth.to(device)
            val_truth = val_truth.float()
            val_outputs = au_net(val_inputs).to(device)
            # Accuracy
            val_outputs = (val_outputs > 0.5).float()
            for index in range(BATCH_SIZE):
                label = np.array(val_truth[index].cpu()).flatten()
                labels1 = np.append(labels1, label[0])
                labels2 = np.append(labels2, label[1])
                labels3 = np.append(labels3, label[2])
                labels4 = np.append(labels4, label[5])
                labels5 = np.append(labels5, label[6])
                labels6 = np.append(labels6, label[7])
                labels7 = np.append(labels7, label[8])
                labels8 = np.append(labels8, label[9])
                labels9 = np.append(labels9, label[10])
                pred = np.array(val_outputs[index].detach().cpu()).flatten()
                preds1 = np.append(preds1, pred[0])
                preds2 = np.append(preds2, pred[1])
                preds3 = np.append(preds3, pred[2])
                preds4 = np.append(preds4, pred[5])
                preds5 = np.append(preds5, pred[6])
                preds6 = np.append(preds6, pred[7])
                preds7 = np.append(preds7, pred[8])
                preds8 = np.append(preds8, pred[9])
                preds9 = np.append(preds9, pred[10])
        AU_labels = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23"]
        print("AU{} Pearson Corr = {}".format(AU_labels[0], pearsonr(preds1, labels1)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[0], f1_score(preds1, labels1, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[1], pearsonr(preds2, labels2)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[1], f1_score(preds2, labels2, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[2], pearsonr(preds3, labels3)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[2], f1_score(preds3, labels3, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[5], pearsonr(preds4, labels4)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[5], f1_score(preds4, labels4, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[6], pearsonr(preds5, labels5)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[6], f1_score(preds5, labels5, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[7], pearsonr(preds6, labels6)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[7], f1_score(preds6, labels6, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[8], pearsonr(preds7, labels7)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[8], f1_score(preds7, labels7, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[9], pearsonr(preds8, labels8)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[9], f1_score(preds8, labels8, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[10], pearsonr(preds9, labels9)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[10], f1_score(preds9, labels9, average='micro')))


class DISFA:
    def __init__(self, path="/projects/dataset_processed/DISFA/"):
        self.path = path

    def truth_values(self):
        for i in range(1, 33):
            dir = self.path + "ActionUnit_Labels/SN" + str(format(i, '03d')) + "/SN" + str(format(i, '03d')) + "_au"
            with open('disfa_au_truth_SN' + str(format(i, '03d')) + ".csv", 'w') as target:
                data = dict()
                for j in range(1, 27):
                    csvpath = dir + str(j) + ".txt"
                    try:
                        readlines = open(csvpath, "r").readlines()
                    except FileNotFoundError:
                        continue
                    output = list()
                    for k in range(len(readlines)):
                        element = readlines[k].split(',')[1]
                        if int(list(element)[0]) > 0:
                            element = 1
                        else:
                            element = 0
                        output.append(element)
                    data[j] = output
                df = pd.DataFrame(data)
                df.to_csv(target, index=False)
            target.close()
            try:
                df = pd.read_csv('disfa_au_truth_SN' + str(format(i, '03d')) + ".csv")
            except:
                os.remove('disfa_au_truth_SN' + str(format(i, '03d')) + ".csv")

    def extract_disfa(self):
        dir = self.path + "Video_RightCamera/"
        for i in range(1, 33):
            if len(os.listdir("frames")) != 0:
                os.system("rm frames/*")
            # cmd = "ffmpeg -i " + dir + "RightVideoSN" + str(format(i, '03d')) + "_Comp.avi" + \
            #       " -qscale:v 2 frames/SN" + str(format(i, '03d')) + "_%03d.jpg"
            # print (cmd)
            # os.system(cmd)
            ff = ffmpy.FFmpeg(
                inputs={dir + "RightVideoSN" + str(format(i, '03d')) + "_Comp.avi": None},
                outputs={"frames/SN" + str(format(i, '03d')) + "_%03d.jpg": ["-qscale:v", "2"]}
            )
            ff.run()

    def processor(self):
        print ("Pre-processing DISFA...!!!")
        processed_dataset = open("pro_DISFA.txt", 'w')
        for i in range(1, 33):
            try:
                dataframe = pd.read_csv('disfa_au_truth_SN' + str(format(i, '03d')) + ".csv")
            except FileNotFoundError:
                continue
            for j, row in dataframe.iterrows():
                row = list(row)
                line = "/projects/dataset_processed/DISFA_frames/SN" + str(format(i, '03d')) + "_" + str(
                    format(j, '03d')) + ".jpg/" + "".join(str(e) for e in row) + "\n"
                index = line.index(".jpg/")
                line = list(line)
                line.pop(index + 4 + 11)
                line.pop(index + 4 + 10)
                line.pop(index + 4 + 9)
                line.pop(index + 4 + 5)
                line.pop(index + 4 + 3)
                processed_dataset.write("".join(line))

    def evaluate_loader(self, data_path="/projects/dataset_processed/DISFA_frames/", batch_size=16, index=0,
                        mean=[0.1732, 0.1732, 0.1732], std=[0.1505, 0.1505, 0.1505]):
        self.data_path = data_path
        self.index = index
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        data = list()
        batch = 0
        AU_labels = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23"]
        with open("pro_DISFA.txt", 'r') as src:
            readlines = src.readlines()
            batch_tensor = torch.Tensor(self.batch_size)
            truth_tensor = torch.Tensor(self.batch_size)
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
                img = F.crop(img, 200, 200, 800, 500)
                img = F.resize(img, (224, 224))
                img = F.to_grayscale(img, 3)
                img = F.to_tensor(img)
                img = F.normalize(img, mean=self.mean, std=self.std)
                truth = list(map(int, list(readlines[image][
                                           readlines[image].index(".jpg/") + 5: readlines[image].index("\n")])))
                truth = torch.from_numpy(np.array(truth)).float()
                if batch == 0:
                    batch_tensor = img
                    truth_tensor = truth
                    batch += 1
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
                elif batch == 1:
                    batch_tensor = torch.cat((batch_tensor.unsqueeze(0), img.unsqueeze(0)), dim=0)
                    truth_tensor = torch.cat((truth_tensor.unsqueeze(0), truth.unsqueeze(0)), dim=0)
                    batch += 1
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
                elif 1 < batch < self.batch_size:
                    batch_tensor = torch.cat((batch_tensor, img.unsqueeze(0)), dim=0)
                    truth_tensor = torch.cat((truth_tensor, truth.unsqueeze(0)), dim=0)
                    batch += 1
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
                else:
                    data.append((batch_tensor, truth_tensor))
                    batch = 1
                    batch_tensor = img
                    truth_tensor = truth
                    if image + 1 == len(readlines):
                        image = 0
                    else:
                        image += 1
        return data

    def train_cnn(self, au_net, config, model_save_path="EmoNet.pth", train_path="train_DISFA.txt",
                  test_path="test_DISFA.txt", loss_weight=0.65):
        EPOCHS = config.num_epochs
        BATCH_SIZE = config.batch_size
        LEARNING_RATE = config.lr
        TRAIN_DATA_PATH = train_path
        TEST_DATA_PATH = test_path

        print ("Selected params: \nEpochs %d \nBatch Size %d \nLearning Rate %.6f \nTrain path %s \nTest path %s"
               % (EPOCHS, BATCH_SIZE, LEARNING_RATE, TRAIN_DATA_PATH, TEST_DATA_PATH))

        # to find means of Action Units
        train_AU_mean = torch.from_numpy(np.array([0.5 for _ in range(11)]))
        test_AU_mean = torch.from_numpy(np.array([0.5 for _ in range(11)]))

        AU_labels = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23"]

        # setting networks to cuda if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        print('device selected: ', device)
        au_net.to(device)

        # class_weights = torch.load("class_weight.pt").to(device)
        AU_criterion = nn.BCELoss()
        corr_net = base_model.CorrCNN().to(device)
        params = list(au_net.parameters()) + list(corr_net.parameters())
        optimizer = optim.Adam(params, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=0.1)

        epoch = 0

        emo_file = model_save_path
        if torch.cuda.is_available() is True:
            model_zoo_ = torch.load(emo_file)
        else:
            model_zoo_ = torch.load(emo_file, map_location='cpu')
        model_zoo_ = {k.replace('model.', ''): v for k, v in model_zoo_.items()}
        model_zoo_2 = dict()
        for k, v in model_zoo_.items():
            if "classifier" not in k:
                model_zoo_2.update({k: v})

        # Setting empty weights to custom classifier for AUnet for 11 classes
        model_zoo_2.update({"classifier.0.weight": model_zoo_["classifier.0.weight"]})
        model_zoo_2.update({"classifier.0.bias": model_zoo_["classifier.0.bias"]})
        model_zoo_2.update({"classifier.3.weight": torch.randn((11, 4096))})
        model_zoo_2.update({"classifier.3.bias": torch.randn(11)})

        au_net.load_state_dict(model_zoo_2)

        checkpoint = torch.load("BP4D_CNN_19-9-19_state.pth")
        au_net.load_state_dict(checkpoint['au_final_state'])
        corr_net.load_state_dict(checkpoint['corr_final_state'])

        del_cor = torch.load("corr_tensor.npy").to(device)  # delta correlaton factor
        print ("TRAINING STARTS...!!!")
        with open("DISFA_CNN_19-9-19.txt", 'w') as src:
            for epoch in range(epoch, EPOCHS):
                running_loss = 0.0
                labels1 = np.array([])
                labels2 = np.array([])
                labels3 = np.array([])
                labels4 = np.array([])
                labels5 = np.array([])
                labels6 = np.array([])
                labels7 = np.array([])
                preds1 = np.array([])
                preds2 = np.array([])
                preds3 = np.array([])
                preds4 = np.array([])
                preds5 = np.array([])
                preds6 = np.array([])
                preds7 = np.array([])
                for i in range(0, len(open(TRAIN_DATA_PATH, 'r').readlines()), BATCH_SIZE):
                    print(epoch + 1, "/", i, "DISFA CNN 19-9-19")
                    train_data = self.evaluate_loader(data_path=TRAIN_DATA_PATH, batch_size=BATCH_SIZE, index=i)
                    inputs, truth = train_data[0][0], train_data[0][1]
                    inputs = inputs.to(device)
                    truth = truth.to(device)
                    truth = truth.float()
                    optimizer.zero_grad()
                    outputs = au_net(inputs).to(device)
                    corr_factor = corr_net(inputs).reshape([BATCH_SIZE, 11, 5]).to(device)
                    corr_factor = torch.bmm(corr_factor, corr_factor.permute(0, 2, 1))
                    del_corr = torch.bmm(del_cor, del_cor.permute(0, 2, 1))
                    """
                    loss_2 is the CorrNet loss
                    """
                    loss_2 = torch.Tensor(11, 11).fill_(-1.0).to(device) * (corr_factor + del_corr) * torch.bmm(
                        (outputs - train_AU_mean.float().to(device)).unsqueeze(1),
                        (outputs - train_AU_mean.float().to(device)).unsqueeze(1).permute(0, 2, 1))

                    corr_loss = torch.mean(loss_2[torch.triu(torch.ones(BATCH_SIZE, 11, 11)) == 1]).to(device)

                    print ("corr loss", corr_loss)
                    loss = torch.zeros(1).to(device)
                    for index in range(BATCH_SIZE):
                        loss += AU_criterion(outputs[index][0], truth[index][0])
                        loss += AU_criterion(outputs[index][1], truth[index][1])
                        loss += AU_criterion(outputs[index][2], truth[index][2])
                        loss += AU_criterion(outputs[index][3], truth[index][3])
                        loss += AU_criterion(outputs[index][6], truth[index][4])
                        loss += AU_criterion(outputs[index][8], truth[index][5])
                        loss += AU_criterion(outputs[index][9], truth[index][6])
                    loss = loss + 4.0 + loss_weight * corr_loss
                    loss.backward()
                    print ("total loss", loss)
                    optimizer.step()
                    running_loss += loss.item()

                    j = i
                    if j >= len(open(TRAIN_DATA_PATH, 'r').readlines()):
                        j = 0

                    ##########
                    # RUNNING  VALIDATION
                    #########
                    au_net.eval()
                    val_data = self.evaluate_loader(data_path=TEST_DATA_PATH, batch_size=BATCH_SIZE, index=j)
                    val_inputs, val_truth = val_data[0][0], val_data[0][1]

                    val_inputs = val_inputs.to(device)
                    val_truth = val_truth.to(device)
                    val_truth = val_truth.float()
                    val_outputs = au_net(val_inputs).to(device)
                    # Accuracy
                    val_outputs = (val_outputs > 0.5).float()
                    for index in range(BATCH_SIZE):
                        label = np.array(val_truth[index].cpu()).flatten()
                        labels1 = np.append(labels1, label[0])
                        labels2 = np.append(labels2, label[1])
                        labels3 = np.append(labels3, label[2])
                        labels4 = np.append(labels4, label[3])
                        labels5 = np.append(labels5, label[4])
                        labels6 = np.append(labels6, label[5])
                        labels7 = np.append(labels7, label[6])
                        pred = np.array(val_outputs[index].detach().cpu()).flatten()
                        preds1 = np.append(preds1, pred[0])
                        preds2 = np.append(preds2, pred[1])
                        preds3 = np.append(preds3, pred[2])
                        preds4 = np.append(preds4, pred[3])
                        preds5 = np.append(preds5, pred[6])
                        preds6 = np.append(preds6, pred[8])
                        preds7 = np.append(preds7, pred[9])
                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[0], pearsonr(preds1, labels1)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[0],
                                                           f1_score(preds1, labels1, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[1], pearsonr(preds2, labels2)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[1],
                                                           f1_score(preds2, labels2, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[2], pearsonr(preds3, labels3)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[2],
                                                           f1_score(preds3, labels3, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[3], pearsonr(preds4, labels4)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[3],
                                                           f1_score(preds4, labels4, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[6], pearsonr(preds5, labels5)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[6],
                                                           f1_score(preds5, labels5, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[8], pearsonr(preds6, labels6)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[8],
                                                           f1_score(preds6, labels6, average='micro')))

                print("Epoch{} AU{} Pearson Corr = {}".format(epoch + 1, AU_labels[9], pearsonr(preds7, labels7)[0]))
                print ("Epoch{} AU{} F1 Score = {}".format(epoch + 1, AU_labels[9],
                                                           f1_score(preds7, labels7, average='micro')))

                print("\n\n\n\n")

                torch.save({'epoch': epoch,
                            'au_model_state_dict': au_net.state_dict(),
                            'corr_model_state_dict': corr_net.state_dict()
                            }, 'checkpoint_DISFA_CNN_19-9-19.pth')

        print("Finished Training")
        src.close()
        torch.save({'au_final_state': au_net.state_dict(),
                    'corr_final_state': corr_net.state_dict(),
                    }, 'final_DISFA_CNN_19-9-19_state.pth')

    def test_cnn(self, au_net, config, model_save_path="EmoNet.pth", test_path="pro_DISFA.txt"):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        print('device selected: ', device)
        au_net.to(device)
        BATCH_SIZE = config.batch_size
        TEST_DATA_PATH = test_path
        corr_net = base_model.CorrCNN().to(device)

        emo_file = model_save_path
        if torch.cuda.is_available() is True:
            model_zoo_ = torch.load(emo_file)
        else:
            model_zoo_ = torch.load(emo_file, map_location='cpu')
        model_zoo_ = {k.replace('model.', ''): v for k, v in model_zoo_.items()}
        model_zoo_2 = dict()
        for k, v in model_zoo_.items():
            if "classifier" not in k:
                model_zoo_2.update({k: v})

        # Setting empty weights to custom classifier for AUnet for 11 classes
        model_zoo_2.update({"classifier.0.weight": model_zoo_["classifier.0.weight"]})
        model_zoo_2.update({"classifier.0.bias": model_zoo_["classifier.0.bias"]})
        model_zoo_2.update({"classifier.3.weight": torch.randn((11, 4096))})
        model_zoo_2.update({"classifier.3.bias": torch.randn(11)})

        au_net.load_state_dict(model_zoo_2)

        checkpoint = torch.load("BP4D_sub_CNN_28-9-19.pth")
        au_net.load_state_dict(checkpoint['au_final_state'])
        corr_net.load_state_dict(checkpoint['corr_final_state'])
        au_net.eval()
        labels1 = np.array([])
        labels2 = np.array([])
        labels3 = np.array([])
        labels4 = np.array([])
        labels5 = np.array([])
        labels6 = np.array([])
        labels7 = np.array([])
        preds1 = np.array([])
        preds2 = np.array([])
        preds3 = np.array([])
        preds4 = np.array([])
        preds5 = np.array([])
        preds6 = np.array([])
        preds7 = np.array([])
        for j in tqdm(range(0, len(open(TEST_DATA_PATH, 'r').readlines()), BATCH_SIZE)):
            val_data = self.evaluate_loader(data_path=TEST_DATA_PATH, batch_size=BATCH_SIZE, index=j)
            val_inputs, val_truth = val_data[0][0], val_data[0][1]

            val_inputs = val_inputs.to(device)
            val_truth = val_truth.to(device)
            val_truth = val_truth.float()
            val_outputs = au_net(val_inputs).to(device)
            # Accuracy
            val_outputs = (val_outputs > 0.5).float()
            for index in range(BATCH_SIZE):
                label = np.array(val_truth[index].cpu()).flatten()
                labels1 = np.append(labels1, label[0])
                labels2 = np.append(labels2, label[1])
                labels3 = np.append(labels3, label[2])
                labels4 = np.append(labels4, label[3])
                labels5 = np.append(labels5, label[4])
                labels6 = np.append(labels6, label[5])
                labels7 = np.append(labels7, label[6])
                pred = np.array(val_outputs[index].detach().cpu()).flatten()
                preds1 = np.append(preds1, pred[0])
                preds2 = np.append(preds2, pred[1])
                preds3 = np.append(preds3, pred[2])
                preds4 = np.append(preds4, pred[3])
                preds5 = np.append(preds5, pred[6])
                preds6 = np.append(preds6, pred[8])
                preds7 = np.append(preds7, pred[9])

        AU_labels = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23"]
        print("AU{} Pearson Corr = {}".format(AU_labels[0], pearsonr(preds1, labels1)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[0], f1_score(preds1, labels1, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[1], pearsonr(preds2, labels2)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[1], f1_score(preds2, labels2, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[2], pearsonr(preds3, labels3)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[2], f1_score(preds3, labels3, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[3], pearsonr(preds4, labels4)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[3], f1_score(preds4, labels4, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[4], pearsonr(preds5, labels5)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[4], f1_score(preds5, labels5, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[5], pearsonr(preds6, labels6)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[5], f1_score(preds6, labels6, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[6], pearsonr(preds7, labels7)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[6], f1_score(preds7, labels7, average='micro')))


class BP4D:

    def __init__(self):
        self.data_path = ""
        self.index = 0
        self.batch_size = 0
        self.mean = []
        self.std = []

    def evaluate_loader(self, data_path="", batch_size=16, index=0, mean=[0.1732, 0.1732, 0.1732],
                        std=[0.1505, 0.1505, 0.1505]):
        self.data_path = data_path
        self.index = index
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
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

    def test_cnn(self, au_net, config, model_save_path="EmoNet.pth", test_path="test_pro_BP4D_subjects.txt"):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        print('device selected: ', device)
        au_net.to(device)
        BATCH_SIZE = config.batch_size
        TEST_DATA_PATH = test_path
        corr_net = base_model.CorrCNN().to(device)

        emo_file = model_save_path
        if torch.cuda.is_available() is True:
            model_zoo_ = torch.load(emo_file)
        else:
            model_zoo_ = torch.load(emo_file, map_location='cpu')
        model_zoo_ = {k.replace('model.', ''): v for k, v in model_zoo_.items()}
        model_zoo_2 = dict()
        for k, v in model_zoo_.items():
            if "classifier" not in k:
                model_zoo_2.update({k: v})

        # Setting empty weights to custom classifier for AUnet for 11 classes
        model_zoo_2.update({"classifier.0.weight": model_zoo_["classifier.0.weight"]})
        model_zoo_2.update({"classifier.0.bias": model_zoo_["classifier.0.bias"]})
        model_zoo_2.update({"classifier.3.weight": torch.randn((11, 4096))})
        model_zoo_2.update({"classifier.3.bias": torch.randn(11)})

        au_net.load_state_dict(model_zoo_2)

        checkpoint = torch.load("BP4D_sub_CNN_28-9-19_state.pth")
        au_net.load_state_dict(checkpoint['au_final_state'])
        corr_net.load_state_dict(checkpoint['corr_final_state'])
        au_net.eval()
        labels1 = np.array([])
        labels2 = np.array([])
        labels3 = np.array([])
        labels4 = np.array([])
        labels5 = np.array([])
        labels6 = np.array([])
        labels7 = np.array([])
        labels8 = np.array([])
        labels9 = np.array([])
        labels10 = np.array([])
        labels11 = np.array([])
        preds1 = np.array([])
        preds2 = np.array([])
        preds3 = np.array([])
        preds4 = np.array([])
        preds5 = np.array([])
        preds6 = np.array([])
        preds7 = np.array([])
        preds8 = np.array([])
        preds9 = np.array([])
        preds10 = np.array([])
        preds11 = np.array([])
        for j in tqdm(range(0, len(open(TEST_DATA_PATH, 'r').readlines()), BATCH_SIZE)):
            val_data = self.evaluate_loader(data_path=TEST_DATA_PATH, batch_size=BATCH_SIZE, index=j)
            val_inputs, val_truth = val_data[0][0], val_data[0][1]

            val_inputs = val_inputs.to(device)
            val_truth = val_truth.to(device)
            val_truth = val_truth.float()
            val_outputs = au_net(val_inputs).to(device)

            # Accuracy
            val_predictions = (val_outputs > 0.5).float()
            for index in range(BATCH_SIZE):
                label = np.array(val_truth[index].cpu()).flatten()
                labels1 = np.append(labels1, label[0])
                labels2 = np.append(labels2, label[1])
                labels3 = np.append(labels3, label[2])
                labels4 = np.append(labels4, label[3])
                labels5 = np.append(labels5, label[4])
                labels6 = np.append(labels6, label[5])
                labels7 = np.append(labels7, label[6])
                labels8 = np.append(labels8, label[7])
                labels9 = np.append(labels9, label[8])
                labels10 = np.append(labels10, label[9])
                labels11 = np.append(labels11, label[10])
                pred = np.array(val_predictions[index].detach().cpu()).flatten()
                preds1 = np.append(preds1, pred[0])
                preds2 = np.append(preds2, pred[1])
                preds3 = np.append(preds3, pred[2])
                preds4 = np.append(preds4, pred[3])
                preds5 = np.append(preds5, pred[4])
                preds6 = np.append(preds6, pred[5])
                preds7 = np.append(preds7, pred[6])
                preds8 = np.append(preds8, pred[7])
                preds9 = np.append(preds9, pred[8])
                preds10 = np.append(preds10, pred[9])
                preds11 = np.append(preds11, pred[10])

        AU_labels = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23"]
        print("AU{} Pearson Corr = {}".format(AU_labels[0], pearsonr(preds1, labels1)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[0], f1_score(preds1, labels1, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[1], pearsonr(preds2, labels2)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[1], f1_score(preds2, labels2, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[2], pearsonr(preds3, labels3)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[2], f1_score(preds3, labels3, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[3], pearsonr(preds4, labels4)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[3], f1_score(preds4, labels4, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[4], pearsonr(preds5, labels5)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[4], f1_score(preds5, labels5, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[5], pearsonr(preds6, labels6)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[5], f1_score(preds6, labels6, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[6], pearsonr(preds7, labels7)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[6], f1_score(preds7, labels7, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[7], pearsonr(preds8, labels8)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[7], f1_score(preds8, labels8, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[8], pearsonr(preds9, labels9)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[8], f1_score(preds9, labels9, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[9], pearsonr(preds10, labels10)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[9], f1_score(preds10, labels10, average='micro')))

        print("AU{} Pearson Corr = {}".format(AU_labels[10], pearsonr(preds11, labels11)[0]))
        print ("AU{} F1 Score = {}".format(AU_labels[10], f1_score(preds11, labels11, average='micro')))
