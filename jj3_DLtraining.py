import sys
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import pandas as pd
from termcolor import colored
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef
from model_util import *
import os

'''
This function is used to evaluate the accuracy of the model
'''
def evaluate_accuracy(data_iter, net, torchdevice = "mps"):
    #device = torch.device("mps", 0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else torchdevice)

    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        x, y = x.float().to(device), y.float().to(device)
        outputs = net(x)
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

'''
This function is used to log the results of the training process
'''
def to_log(log, output_dir):
    with open(output_dir + "modelLog.log", "a") as f:
        f.write(log + '\n')

'''
This function is used to train the model
If load_model_dir is empty, the model is trained from scratch
If load_model_dir is not empty, the model is fine-tuned. In this case input_modelstring describes the parameters of the fine-tuned model
'''
def DL_train(cfg, EPOCHS, LR, BATCH_SIZE, logger = "", load_model_dir = "", input_modelstring = "", excludeSpeciesGroup = -1, _SILENT_RUN = True, trainFlag = "ask"):
    screen = sys.stdout
    if _SILENT_RUN:
        # Store references to original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

    #device = torch.device("mps", 0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else cfg["ENVIRONMENT"]["torchdevice"])

    #Load the file named dataset_TT.csv in the 0.DataSet folder of the expConf["Folder"] folder
    if load_model_dir == "":
        datasetFile = os.path.join("experiments", cfg["GENERAL"]["folder"], "0.DataSet", "dataset_TT.csv")
    else:
        datasetFile = os.path.join("experiments", cfg["GENERAL"]["folder"], "0.DataSet", "dataset_FT.csv")

    model_dir_path = os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training", "Model")
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    if load_model_dir == "":
        logits_output = os.path.join(model_dir_path, f'M_TT_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TRAINTEST"]["model_name"]}_batch_{BATCH_SIZE}_exclude_{excludeSpeciesGroup}_logits.csv')
        model_filename = f'M_TT_{cfg["GENERAL"]["acronym"]}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TRAINTEST"]["model_name"]}_batch_{BATCH_SIZE}_exclude_{excludeSpeciesGroup}.pl'
        model_loc = os.path.join(model_dir_path, model_filename)
    else:
        logits_output = os.path.join(model_dir_path, f'M_FT_origin_{input_modelstring}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TRAINTEST"]["model_name"]}_batch_{BATCH_SIZE}_exclude_{excludeSpeciesGroup}_logits.csv')
        model_filename = f'M_FT_origin_{input_modelstring}_epochs_{EPOCHS}_lr_{LR:.7f}_model_{cfg["TRAINTEST"]["model_name"]}_batch_{BATCH_SIZE}_exclude_{excludeSpeciesGroup}.pl'
        model_loc = os.path.join(model_dir_path, model_filename)

    logger.log_message (f"Output Model: {model_filename}")

    #if I am not in Leave One Out and a model with the same name already exists...
    if os.path.exists(model_loc) and trainFlag == "no":
       return model_loc

    jj_all_data = pd.read_csv(datasetFile)

    #if excludeSpecies is not empty, exclude from the dataset the species in the list - This is for LEAVE ONE OUT
    if excludeSpeciesGroup >= 0:
        jj_all_data = jj_all_data[~jj_all_data['Species'].isin(cfg["TRAINTEST"]["leaveoneoutspecies"][excludeSpeciesGroup])]
        logger.log_message("*" * 80)
        logger.log_message (f"Leave-One-Out Species: {excludeSpeciesGroup} ({cfg['TRAINTEST']['leaveoneoutspecies'][excludeSpeciesGroup]})")

    logger.log_message (f"Total number of Label 1 in the Test/Train dataset: {len(jj_all_data[jj_all_data['Label'] == 1])}")
    logger.log_message (f"Total number of Label 0 in the Test/Train dataset: {len(jj_all_data[jj_all_data['Label'] == 0])}")
    #logger.log_message the total number of rows that have Label = 1 and bin = train
    logger.log_message (f"\tTrain: 1: {len(jj_all_data[(jj_all_data['Label'] == 1) & (jj_all_data['bin'] == 'train')])} - 0: {len(jj_all_data[(jj_all_data['Label'] == 0) & (jj_all_data['bin'] == 'train')])}")
    logger.log_message (f"\tTest: 1: {len(jj_all_data[(jj_all_data['Label'] == 1) & (jj_all_data['bin'] == 'test')])} - 0: {len(jj_all_data[(jj_all_data['Label'] == 0) & (jj_all_data['bin'] == 'test')])}")
    logger.log_message (f"\tVal: 1: {len(jj_all_data[(jj_all_data['Label'] == 1) & (jj_all_data['bin'] == 'val')])} - 0: {len(jj_all_data[(jj_all_data['Label'] == 0) & (jj_all_data['bin'] == 'val')])}")
    logger.log_message ("*" * 80)
    logger.log_message(f"Batch Size: {BATCH_SIZE}")
    logger.log_message ("*" * 80)
    logger.log_message(f"Phase: {'Fine Tuning' if load_model_dir != '' else 'Training'}")
    logger.log_message ("*" * 80)

    label = torch.tensor(jj_all_data.loc[:,'Label'].values)
    data = torch.tensor(jj_all_data.loc[:, '0':'1023'].values)

    print(label.shape, data.shape)

    train_data = torch.tensor(jj_all_data.loc[jj_all_data['bin'] == 'train', '0':'1023'].values).double()
    train_label = torch.tensor(jj_all_data.loc[jj_all_data['bin'] == 'train', 'Label'].values)

    test_data = torch.tensor(jj_all_data.loc[jj_all_data['bin'] == 'test', '0':'1023'].values).double()
    test_label = torch.tensor(jj_all_data.loc[jj_all_data['bin'] == 'test', 'Label'].values)

    jjfinal_test_data = torch.tensor(jj_all_data.loc[jj_all_data['bin'] == 'val', '0':'1023'].values).double()
    jjfinal_test_label = torch.tensor(jj_all_data.loc[jj_all_data['bin'] == 'val', 'Label'].values)

    trP = np.count_nonzero(train_label == 1)
    trN = np.count_nonzero(train_label == 0)

    LOSS_WEIGHT_POSITIVE = (int(trP) + int(trN)) / (2.0 * int(trP))
    LOSS_WEIGHT_NEGATIVE = (int(trP) + int(trN)) / (2.0 * int(trN))

    # https://towardsdatascience.com/deep-learning-with-weighted-cross-entropy-loss-on-imbalanced-tabular-data-using-fastai-fe1c009e184c
    soft_max = nn.Softmax(dim=1)
    # class_weights=torch.FloatTensor([w_0, w_1]).cuda()
    weig = torch.FloatTensor([LOSS_WEIGHT_NEGATIVE, LOSS_WEIGHT_POSITIVE]).float().to(device)

    # train_data,train_label=genData("./train_peptide.csv",260)
    # test_data,test_label=genData("./test_peptide.csv",260)

    train_dataset = Data.TensorDataset(train_data, train_label)
    test_dataset = Data.TensorDataset(test_data, test_label)
    batch_size = BATCH_SIZE
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    Emb_dim = data.shape[1]


    ############################
    model_name = cfg["TRAINTEST"]["model_name"]
    model_func = globals()[model_name]
    net = model_func().float().to(device)
    logger.log_message(f'Trained model: {cfg["TRAINTEST"]["model_name"]}')

    # if cfg["TRAINTEST"]["big_or_small_model"] == 0:
    #     logger.log_message("Model 1")
    #     net = newModel1().float().to(device)
    # else:
    #     logger.log_message("Model 2")
    #     net = newModel2().float().to(device)
        # state_dict=torch.load('/content/Model/pretrain.pl')
        # net.load_state_dict(state_dict['model'])

    logger.log_message(f"Initial Learning Rate: {LR}")

    if load_model_dir != "":
        state_dict = torch.load(load_model_dir)
        net.load_state_dict(state_dict['model'])
        logger.log_message(f"Input Model: {os.path.basename(load_model_dir)}")
    # lr = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.75, verbose=True)
    # https://discuss.pytorch.org/t/reducelronplateau-not-doing-anything/24575/10
    # criterion = ContrastiveLoss()
    # criterion_model = nn.CrossEntropyLoss(reduction='sum')

    criterion_model = nn.CrossEntropyLoss(weight=weig, reduction='mean')
    best_bacc = 0
    best_aupr = 0

    CUDA_LAUNCH_BLOCKING = 1
    for epoch in range(EPOCHS):
        loss_ls = []

        t0 = time.time()
        net.train()
        # for seq1,seq2,label,label1,label2 in train_iter_cont:
        for seq, label in train_iter:
            # logger.log_message(seq1.shape,seq2.shape,label.shape,label1.shape,label2.shape)
            seq, label = seq.float().to(device), label.to(device)
            output = net(seq)
            loss = criterion_model(output, label)
            #             logger.log_message(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
        lr_scheduler.step(loss)

        output_dir = os.path.join("experiments", cfg["GENERAL"]["folder"], "1.DL_Training")

        if epoch % 100 == 0:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(output_dir, 'ckpt_{}.pl'.format(epoch)))
        net.eval()
        with torch.no_grad():
            train_acc = evaluate_accuracy(train_iter, net, torchdevice = cfg["ENVIRONMENT"]["torchdevice"])
            # test_acc=evaluate_accuracy(test_iter,net)
            test_data_gpu = test_data.float().to(device)
            test_logits = net(test_data_gpu)
            outcome = np.argmax(test_logits.detach().cpu(), axis=1)
            test_bacc = balanced_accuracy_score(test_label, outcome)
            precision, recall, thresholds = precision_recall_curve(test_label, soft_max(test_logits.cpu())[:, 1])
            test_aupr = auc(recall, precision)
        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        # results=f"epoch: {epoch+1}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, test_aupr: {colored(test_aupr, "red")},test_bacc: {colored(test_bacc, "red")}, time: {time.time() - t0:.2f}'
        logger.log_message(results, epoch/EPOCHS)
        to_log(results, output_dir)
        if test_aupr > best_aupr:
            best_aupr = test_aupr
            #torch.save({"best_aupr": best_aupr, "model": net.state_dict(), 'args': jj_arg}, model_loc)
            torch.save({"best_aupr": best_aupr, "model": net.state_dict(), 'args': {}}, model_loc)
            logger.log_message(f"best_aupr: {best_aupr}")
    state_dict = torch.load(model_loc)

    # state_dict=torch.load('/content/Model/pretrain.pl')
    net.load_state_dict(state_dict['model'])
    #pro = pd.read_csv(args.pro_label_dir)
    #label = torch.tensor(pro['label'].values)
    # final_test_data,final_test_label=data[9655+1068:].double(),label[9655+1068:]
    # train_data,train_label=data[:6011].double(),label[:6011]
    #final_test_data, final_test_label = data[-int(teP) - int(teN):].double(), label[-int(teP) - int(teN):]
    final_test_data, final_test_label = jjfinal_test_data, jjfinal_test_label
    final_test_data = final_test_data.float().to(device)
    net.eval()
    with torch.no_grad():
        logits = net(final_test_data)

    # logits_output=os.path.split(rep_file)[1].replace('.csv','_logtis.csv')
    logits_cpu = logits.cpu().detach().numpy()
    logits_cpu_pd = pd.DataFrame(logits_cpu)
    logits_cpu_pd.to_csv(logits_output, index=False)

    outcome = np.argmax(logits.cpu().detach().numpy(), axis=1)
    MCC = matthews_corrcoef(final_test_label.cpu(), outcome)
    acc = accuracy_score(final_test_label, outcome)
    bacc = balanced_accuracy_score(final_test_label, outcome)
    precision1, recall1, thresholds1 = precision_recall_curve(final_test_label,
                                                              soft_max(torch.tensor(logits_cpu))[:, 1])
    final_test_aupr = auc(recall1, precision1)
    final_auc_roc = roc_auc_score(final_test_label, soft_max(torch.tensor(logits_cpu))[:, 1])
    # final_test_aupr=0
    logger.log_message('bacc,MCC,final_test_aupr,final_auc_roc')
    logger.log_message(f"{str(bacc).replace('.', ',')} {str(MCC).replace('.', ',')} {str(final_test_aupr).replace('.', ',')} {str(final_auc_roc).replace('.', ',')}")

    #if returnResults == True:
    #    return ([model_loc, str(bacc).replace('.', ','), str(MCC).replace('.', ','), str(final_test_aupr).replace('.', ','), str(final_auc_roc).replace('.', ',')])
    #else:
    return (model_loc)




