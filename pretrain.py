import os
import numpy as np
import argparse
import torch
from pprint import pprint
from data.pretraining import DataReaderPlainImg, custom_collate
from data.transforms import get_transforms_pretraining
from utils import check_dir, accuracy, get_logger
from models.pretraining_backbone import ResNet18Backbone

global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--weights-init', type=str,
                        default="random")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--bs', type=int, default=6, help='batch_size')
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "size"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):



    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)

    # build model and load weights
    model = ResNet18Backbone(pretrained=False).cuda()

    #sd = torch.load('pretrain_weights_init.pth')
    model.load_state_dict(torch.load('pretrain_weights_init.pth')['model'])
    #model.eval()
    #raise NotImplementedError("TODO: load weight initialization")

    # load dataset
    data_root = args.data_folder
    train_transform, val_transform = get_transforms_pretraining(args)
    train_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "train"), transform=train_transform)
    val_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "val"), transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                               pin_memory=True, drop_last=True, collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True, collate_fn=custom_collate)

    # TODO: loss function
    criterion = None
    criterion = torch.nn.CrossEntropyLoss()
    #raise NotImplementedError("TODO: loss function")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    # Train-validate for one epoch. You don't have to run it for 100 epochs, preferably until it starts overfitting.
    for epoch in range(100):
        print("Epoch {}".format(epoch))
        train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = validate(val_loader, model, criterion)

        # save model
        if val_loss < best_val_loss:
            PATH = 'better_model.pth'
            torch.save(model.state_dict(),PATH)
            #raise NotImplementedError("TODO: save model if a new best validation error was reached")


# train one epoch over the whole training dataset. You can change the method's signature.
def train(loader, model, criterion, optimizer):
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(loader):
            print('batch=',i)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()
            #if i % 2000 == 1999:
            #    print('[%d , %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            #    running_loss = 0.0

    print('finished training')
    #raise NotImplementedError("TODO: training routine")


# validation function. you can change the method's signature.
def validate(loader, model, criterion):
    correct = 0
    total = 0
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss = running_loss + loss.item()
            #acc = accuracy(outputs,labels)
    mean_val_accuracy = (100 * correct / total)
    mean_val_loss = (100.0 * running_loss / total)
    mean_val_accuracy = accuracy(outputs,labels)
    print('Accuracy of the network on test images: %d %%' % (mean_val_accuracy))
    print('Loss of the network on test images: %d %%' % (100.0 * running_loss / total))

    #raise NotImplementedError("TODO: validation routine")
    return mean_val_loss, mean_val_accuracy


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running on device: {device}")

    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
