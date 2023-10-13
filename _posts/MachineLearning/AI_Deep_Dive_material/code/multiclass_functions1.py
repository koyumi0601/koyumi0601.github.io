import torch
from torch import nn, optim
from torchvision import datasets,transforms
import numpy as np # confusion matrix 사용시
import matplotlib.pyplot as plt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def Train(model, train_DL, criterion, optimizer, LR, EPOCH):

    NoT=len(train_DL.dataset) # Number of training data
    loss_history = []

    model.train() # train mode로!
    for ep in range(EPOCH):
        rloss = 0
        for x_batch, y_batch in train_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # cross entropy loss
            loss = criterion(y_hat, y_batch)
            # update
            optimizer.zero_grad() # gradient 누적을 막기 위한 초기화
            loss.backward() # backpropagation
            optimizer.step() # weight update
            # loss accumulation
            loss_b = loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE 로 하면 마지막 16개도 32개로 계산해버림
            rloss += loss_b # running loss
        # print loss
        loss_e = rloss/NoT # epoch loss
        loss_history += [loss_e]
        print(f"Epoch: {ep+1} train loss: {round(loss_e,3)}")
        print("-"*20)
        
    return loss_history

def Test(model,test_DL):
    model.eval()
    with torch.no_grad():
        rcorrect = 0
        # rloss = 0
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # loss 
            # loss = criterion(y_hat, y_batch) # 만약 쓰려면 매개변수에 criterion 추가
            # loss accumulation
            # loss_b = loss.item() * x_batch.shape[0] # BATCH_SIZE 로 하면 마지막 16개도 32개로 계산해버림
            # rloss += loss_b # running loss
            # accuracy accumulation
            pred = y_hat.argmax(dim=1)
            corrects_b = torch.sum(pred == y_batch).item()
            rcorrect += corrects_b
        # loss_e = rloss/NoTes
        accuracy_e = rcorrect/len(test_DL.dataset)*100
    # print("test loss: ", round(loss_e,2))
    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(accuracy_e,1)} %)")
    return round(accuracy_e,1)

def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to("cpu")

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1,2,0).squeeze(), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = "g" if pred_class==true_class else "r") 

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num


def get_conf(model, test_DL):
    model.eval()
    with torch.no_grad():
        confusion = torch.zeros(10,10)
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # accuracy
            pred = y_hat.argmax(dim=1)

            for i in range(y_batch.shape[0]):
                confusion[pred[i], y_batch[i]] += 1
            # confusion[pred, y_batch] += 1
        
    confusion = confusion.numpy()

    return confusion

def plot_confusion_matrix(confusion, classes=None):
    accuracy=np.trace(confusion)/np.sum(confusion) * 100
    
    confusion = confusion/np.sum(confusion, axis=0)
    plt.figure(figsize=(15,10))
    plt.imshow(confusion, cmap="Blues")
    plt.title("confusion matrix")
    plt.colorbar()

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j,i, round(confusion[i,j],2), 
                     horizontalalignment="center", 
                     color="white" if confusion[i,j] > np.max(confusion) / 1.5 else "black")

    if classes is not None:
        plt.xticks(range(10), classes)
        plt.yticks(range(10), classes)
    else:
        plt.xticks(range(10))
        plt.yticks(range(10))

    plt.ylabel("Predicted label")
    plt.xlabel(f"True label \n accuracy = {round(accuracy,1)} %")
    plt.show()

