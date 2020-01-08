from comet_ml import Experiment as comex
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import base_model
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import utils

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


def train_cnn(au_net, config, model_save_path="EmoNet.pth", train_path="processed_BP4D_subjects.txt",
              test_path="test_pro_BP4D_subjects.txt", load_model=False, find_au_mean=True, loss_weight=0.65):

    # defining hyper-parameters
    EPOCHS = config.num_epochs
    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.lr
    TRAIN_DATA_PATH = train_path
    TEST_DATA_PATH = test_path

    print ("Selected params: \nEpochs %d \nBatch Size %d \nLearning Rate %.6f \nTrain path %s \nTest path %s"
           % (EPOCHS, BATCH_SIZE, LEARNING_RATE, TRAIN_DATA_PATH, TEST_DATA_PATH))

    # for fast training
    cudnn.benchmark = True

    # to find means of Action Units
    if find_au_mean is True:
        train_AU_mean = torch.from_numpy(np.array(utils.AUMean(TRAIN_DATA_PATH).calc_mean()))
        test_AU_mean = torch.from_numpy(np.array(utils.AUMean(TEST_DATA_PATH).calc_mean()))
    else:
        train_AU_mean = torch.from_numpy(np.array([0.5 for _ in range(11)]))
        test_AU_mean = torch.from_numpy(np.array([0.5 for _ in range(11)]))

    AU_labels = ["1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23"]

    # setting networks to cuda if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    print('device selected: ', device)
    au_net.to(device)

    class_weights = torch.load("class_weight.pt").to(device)
    AU_criterion = nn.BCELoss(weight=class_weights)
    corr_net = base_model.CorrCNN().to(device)
    params = list(au_net.parameters()) + list(corr_net.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=0.1)

    epoch = 0

    # loading pre-trained weights from EmotioNet
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

    # au_net.load_state_dict(model_zoo_2)

    # loading from checkpoint for warm start
    if load_model is True:
        checkpoint = torch.load("./checkpoint_DriverVGG4.pth")
        au_net.load_state_dict(checkpoint['au_model_state_dict'])
        corr_net.load_state_dict(checkpoint['corr_model_state_dict'])
        au_net.train()

    del_cor = torch.load("corr_tensor.npy").to(device)  # delta correlaton factor
    print ("TRAINING STARTS...!!!")

    with torch.autograd.detect_anomaly():
        with open("BP4D_pure_CNN_2-11-19.txt", 'w') as src:
            for epoch in range(epoch, EPOCHS):
                running_loss = 0.0
                val_running_loss = 0.0
                val_running_acc = 0.0
                labels = np.array([])
                preds = np.array([])
                for i in range(0, len(open(TRAIN_DATA_PATH, 'r').readlines()), BATCH_SIZE):
                    print (epoch + 1, "/", i, "Corr Pure CNN 2-11-19")
                    train_data = utils.DataLoader2(data_path=TRAIN_DATA_PATH, batch_size=BATCH_SIZE,
                                                   index=i).loaded_data()
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
                    for index in range(len(truth)):
                        loss += AU_criterion(outputs[index], truth[index])

                    loss = loss + 4.0 + loss_weight * corr_loss
                    experiment.log_metric("loss", loss.item(), step=i)
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
                    val_data = utils.DataLoader2(data_path=TEST_DATA_PATH, batch_size=BATCH_SIZE, index=j).loaded_data()
                    val_inputs, val_truth = val_data[0][0], val_data[0][1]

                    val_inputs = val_inputs.to(device)
                    val_truth = val_truth.to(device)
                    val_truth = val_truth.float()

                    # pdb.set_trace()
                    val_outputs = au_net(val_inputs).to(device)
                    corr_factor = corr_net(val_inputs).reshape([BATCH_SIZE, 11, 5]).to(device)
                    corr_factor = torch.bmm(corr_factor, corr_factor.permute(0, 2, 1)).to(device)
                    del_corr = torch.bmm(del_cor, del_cor.permute(0, 2, 1))
                    loss_2 = torch.Tensor(11, 11).fill_(-1.0).to(device) * (corr_factor + del_corr) * torch.bmm(
                        (val_outputs - train_AU_mean.float().to(device)).unsqueeze(1),
                        (val_outputs - train_AU_mean.float().to(device)).unsqueeze(1).permute(0, 2, 1))

                    corr_loss = torch.mean(loss_2[torch.triu(torch.ones(BATCH_SIZE, 11, 11)) == 1]).to(device)
                    val_loss = torch.zeros(1).to(device)
                    val_loss.to(device)
                    for index in range(BATCH_SIZE):
                        val_loss += AU_criterion(val_outputs[index], val_truth[index])

                    val_loss = val_loss + 4.0 + loss_weight * corr_loss
                    val_running_loss += val_loss.item()

                    # Accuracy
                    val_correct = 0
                    val_predictions = (val_outputs > 0.5).float()
                    for index in range(BATCH_SIZE):
                        for x in range(11):
                            val_correct += val_predictions[index][x] == val_truth[index][x]
                    val_running_acc += (val_correct / (11.0 * BATCH_SIZE)) * 100.0
                    print ("Correct Classifications: ", int(val_correct), " / ", (11.0 * BATCH_SIZE))
                    print ("Accuracy: ", (int(val_correct) / (11.0 * BATCH_SIZE)) * 100.0)
                    val_running_acc += (int(val_correct) / (11.0 * BATCH_SIZE)) * 100.0

                    # Adding labels and predictions to numpy array for every epoch
                    for index in range(BATCH_SIZE):
                        label = np.array(val_truth[index].cpu()).flatten()
                        labels = np.append(labels, label)
                        pred = np.array(val_predictions[index].detach().cpu()).flatten()
                        preds = np.append(preds, pred)
                    au_net.train()
                    #########

                pearson_corr = pearsonr(preds, labels)[0]
                f1 = f1_score(preds, labels, average='micro')
                print('EPOCH: %d, LOSS: %.5f, VAL ACCURACY: %.5f, VAL F1 SCORE: %.5f, PearCorr: %.5f' %
                      (epoch + 1, running_loss / (epoch + 1), val_running_acc / (epoch + 1), f1, pearson_corr), "\n\n")

                torch.save({'epoch': epoch,
                            'au_model_state_dict': au_net.state_dict(),
                            'corr_model_state_dict': corr_net.state_dict()
                            }, 'BP4D_pure_CNN_2-11-19.pth')

    print("Finished Training")
    src.close()
    torch.save({'au_final_state': au_net.state_dict(),
                'corr_final_state': corr_net.state_dict(),
                }, 'BP4D_pure_CNN_2-11-19_state.pth')
