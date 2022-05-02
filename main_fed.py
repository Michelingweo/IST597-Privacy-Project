# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet, weights_init
from models.Fed import FedAvg, gaussian_img
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    #random number init
    seed = 1234  # experimental result
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #path define
    dataset = args.dataset
    root_path = '.'
    data_path = os.path.join(root_path, './data').replace('\\', '/')
    save_path = os.path.join(root_path, 'save/DLG_%s' % dataset).replace('\\', '/')

    #DLG define
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    #path print
    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)



    # load dataset and split users
    if args.dataset == 'mnist':
        image_shape = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        image_shape = (32, 32)
        num_classes = 10
        channel = 3
        hidden = 768
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    #DLG net init
    net_DLG = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
    net_DLG.apply(weights_init)
    net_DLG = net_DLG.to(device)

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))


    print('DLG Testing')



    for idx_net in range(args.num_exp): # replace with communication rounds
        net = net_DLG

        print('running %d|%d experiment' % (idx_net, args.num_exp))

        print('%s, Try to generate %d images' % ('DLG', args.num_dummy))

        criterion = nn.CrossEntropyLoss().to(device)
        imidx_list = []

        for imidx in range(args.num_dummy):
            idx = args.index
            imidx_list.append(idx)

            if args.noise == 0:
                noised_img = dataset_train[idx][0]
            else:
                noised_img = gaussian_img(dataset_train[idx][0], stddev=args.noise)


            tmp_datum = noised_img.float().to(device) # put the data to the GPU
            tmp_datum = tmp_datum.view(1, *tmp_datum.size()) # reshape tmp_datum
            tmp_label = torch.Tensor([dataset_train[idx][1]]).long().to(device) # data label to tensor
            tmp_label = tmp_label.view(1, )
            if imidx == 0:
                gt_data = tmp_datum # ground truth data
                gt_label = tmp_label # ground truth label
            else:
                gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                gt_label = torch.cat((gt_label, tmp_label), dim=0)

            # compute original gradient
            out = net_DLG(gt_data) # prediction
            y = criterion(out, gt_label) # cross entropy for the diff between real label and prediction
            dy_dx = torch.autograd.grad(y, net_DLG.parameters())

            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=args.DLG_lr)

        history = []
        history_iters = []
        grad_difference = []
        data_difference = []
        train_iters = []

        print('lr =', args.DLG_lr)
        for iters in range(args.iteration):

            def closure():
                optimizer.zero_grad()
                pred = net(dummy_data)


                dummy_loss = -torch.mean(
                        torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))



                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)
            current_loss = closure().item()
            train_iters.append(iters)
            grad_difference.append(current_loss)
            data_difference.append(torch.mean((dummy_data - gt_data) ** 2).item())

            if iters % int(args.iteration / 30) == 0:
                print(iters, 'grad diff = %.8f, data diff = %.8f' % (current_loss, data_difference[-1]))
                history.append([tp(dummy_data[imidx].cpu()) for imidx in range(args.num_dummy)])
                history_iters.append(iters)

                for imidx in range(args.num_dummy):
                    plt.figure(figsize=(12, 8))
                    plt.subplot(3, 10, 1)
                    for i in range(min(len(history), 29)):
                        plt.subplot(3, 10, i + 2)
                        plt.imshow(history[i][imidx], cmap='gray')
                        plt.title('iter=%d' % (history_iters[i]))
                        plt.axis('off')

                    plt.savefig('save/DLG_on_%s_%05d.png' % (imidx_list, imidx_list[imidx]))
                    plt.close()

                if current_loss < 0.000001:  # converge
                    break

            loss_DLG = grad_difference
            label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
            mse_DLG = data_difference

    print('image_idx_list :', imidx_list)
    print('gradient diff :', loss_DLG[-1])
    print('data diff :', mse_DLG[-1])
    print('gt_label :', gt_label.detach().cpu().data.numpy(), 'DLG_label: ', label_DLG)

    print('----------------------\n\n')

