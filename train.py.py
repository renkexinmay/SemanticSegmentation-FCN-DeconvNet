import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets
from torchvision import transforms, utils

import os
import time
import matplotlib.pyplot as plt
import argparse
import numpy as np
import PIL

import voc
from model import conv_deconv, FCN
#from load_dataset import load_dataset

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset_year', choices = dict(pascal_2011="2011", pascal_2012="2012"), default = "2012", action = LookupChoices)
parser.add_argument('--model', choices = dict(conv_deconv=conv_deconv(), FCN=FCN()), default = FCN(), action = LookupChoices)
parser.add_argument('--data', default = './data')
parser.add_argument('--log', default = '/log/log.txt')
parser.add_argument('--epochs', default = 100, type = int)
parser.add_argument('--batch', default = 64, type = int)
parser.add_argument('--load', default = False, type = bool)
parser.add_argument('--check_every', default = 5, type = int)
parser.add_argument('--save_every', default = 5, type = int)
opts = parser.parse_args()

def color_map(N=256, normalized=True):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def cmap_func(cmap, img):
        n = 224
        # print(img.size())
        # print(cmap[5])
        tic = time.time()
        res = torch.zeros(3, n, n)
        for i in range(n):
                for j in range(n):
                        colors = cmap[img[0][i][j]]
                        res[0][i][j] = float(colors[0])
                        res[1][i][j] = float(colors[1])
                        res[2][i][j] = float(colors[2])
        print(time.time() - tic)
        return res


def train(model, mode, num_epoch, dataset_train, train_loader, val_loader, optimizer):
        count = 0
        print_label = True
        print_imgs = True
        save_path = "results/" + model.name
        if not os.path.isdir(save_path):
                os.mkdir(save_path)
                os.mkdir(save_path + "/log")
                os.mkdir(save_path + "/checkpoints")
                os.mkdir(save_path + "/saved_images")
        log = open(save_path + opts.log, 'w')
        train_loss = []
        val_loss = []
        cmap = color_map()
                

        for epoch in range(num_epoch):
                loss_epoch = 0
                print("\nEPOCH " +str(epoch)+" of "+str(num_epoch)+"\n")

                model.train()
                scheduler.step()
                for batch_idx, batch in enumerate(train_loader):

                        if batch_idx % 5 == 0:
                                print("EPOCH:", str(epoch), "batch_idx:", str(batch_idx), "out of:", len(train_loader))

                        # Load datas and labels
                        inputs, labels = [torch.autograd.Variable(tensor.to(device)) for tensor in batch]

                        # Feed the network with the datas
                        if model.name == 'FCN_net' or model.name == 'FCN_net_v2' or model.name == 'FCN_net_v3':
                                outputs, fs4, fs3, fs2, fs1, gcfm1 = model(inputs)
                        elif model.name == 'DeconvNet':
                                outputs = model(inputs)

                        # Printing label once to get the form of it
                        if print_label:
                                print('label')
                                # print(labels[0].detach().cpu())
                                print(labels.size())
                                print_label = False

                        # if print_imgs:
                        #         print('img')
                        #         # print(imgs[0].detach().cpu())
                        #         print(imgs.size())
                        #         print_imgs = False

                        # Saving images to have visual results
                        if batch_idx % 5 == 0:
                                test = torch.argmax(outputs[0], dim=0, keepdim=True).squeeze().detach().cpu()
                                # print(test)
                                test = dataset_train.decode_segmap(np.array(test)).astype(np.uint8)
                                # print(test)
                                print('test')
                                print(test.shape)
                                img1 = transforms.ToPILImage()(test)
                                img1.save(save_path + "/saved_images/" + str(count) + "_output_" + str(epoch) + "_" + str(batch_idx) + ".png")
                                img2 = transforms.ToPILImage()(dataset_train.decode_segmap(np.array(labels[0].detach().cpu())).astype(np.uint8))
                                # img2 = PIL.Image.fromarray(imgs[0].detach().cpu())
                                img2.save(save_path + "/saved_images/" + str(count) + "_label_" + str(epoch) + "_" + str(batch_idx) + ".png")
                                count += 1

                        # Compute loss and compute backward
                        print(torch.min(labels), torch.max(labels), labels.size(), outputs.size())
                        print(torch.min(outputs), torch.max(outputs))
                        # print(labels.size())
                        loss = model.criterion(outputs, labels)

                        if loss > 0:
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                        loss_epoch += loss.item()

                print("Epoch :", str(epoch), "; train loss :", str(loss_epoch))
                log.write('train_loss epoch {}: {:.06f}\n'.format(epoch, loss_epoch))
                #train_loss.append(loss_epoch)
                
                if epoch % check_every == 0:
                        model.eval()
                        loss_validation = 0
                        for batch_idx, batch in enumerate(val_loader):

                                if batch_idx % 5 == 0:
                                        print("EVAL MODE; ", "batch_idx:", str(batch_idx), "out of:", len(val_loader))

                                inputs, labels = [torch.autograd.Variable(tensor.to(device)) for tensor in batch]
                                with torch.no_grad():
                                        if model.name == 'FCN_net' or model.name == 'FCN_net_v2' or model.name == 'FCN_net_v3':
                                                outputs, fs4, fs3, fs2, fs1, gcfm1 = model(inputs)
                                        elif model.name == 'DeconvNet':
                                                outputs = model(inputs)
                                loss = model.criterion(outputs, labels)
                                loss_validation += loss.item()

                        print("epoch: ", str(epoch), "; Train loss:", str(loss_epoch), "; Test loss: ", str(loss_validation))
                        log.write('val_loss epoch {}: {:.06f}\n'.format(epoch, loss_validation))
                        #loss_val.append(loss_validation)

                if epoch % save_every == 0:
                        torch.save({
                                'epoch': epoch,
                                'model': opts.model,
                                #'loss_val': loss_validation,
                                'state_dict': model.state_dict()},
                                save_path + "/checkpoints/model_epoch_"+str(epoch)+".pt")
                        print("checkpoint: saved model")

        

if __name__ == '__main__':

        ## Initialisation of global variables
        data_path = opts.data
        year = opts.dataset_year
        check_every = opts.check_every
        save_every = opts.save_every
        num_epoch = opts.epochs

        ## Make sure PIL Version is 5.4.1 and not 4.0.0
        print('PIL Version:', PIL.PILLOW_VERSION, 'WARNING : make sure it is 5.4.1 and not 4.0.0')

        ## Make sure we are actually using CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        print("we are actually using :", device)

        # Create the model before the transformation to get the size
        if not opts.load:
                model = opts.model
        else:
                model = opts.model ## TO BE CHANGED

        ## Transformations for datasets
        size = model.input_size
        transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor()])
        transform_label = transforms.Compose([
                transforms.Resize((size, size)),#])
                transforms.ToTensor()]) #NO MORE NEEDED CAUSE ITS DONE IN VOC.PY
        
        dataset_train = voc.VOCSegmentation(root=data_path, year=year, image_set="train",
                                  download=False, transform = transform, target_transform=transform_label)
        dataset_val = voc.VOCSegmentation(root=data_path, year=year, image_set="val",
                                  download=False, transform = transform, target_transform=transform_label)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=True, num_workers=2)
        
        print("Number of training images:", len(dataset_train))
        print("Number of validation images:", len(dataset_val))
        print("Model name:", model.name)
        print("Loss name:", model.loss_name)

        
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        train(model, "fine_tuning", num_epoch, dataset_train, train_loader, val_loader, optimizer)