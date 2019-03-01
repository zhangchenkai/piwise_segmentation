import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
from torchvision.transforms import ToTensor, ToPILImage, Resize

sys.path.append("/home/nico/PycharmProjects/project-marvel/defect-detection")
from defect_detection.evaluator.evaluation import save_metrics_on_results
from piwise.criterion import CrossEntropyLoss2d
from piwise.dataset import VOCTrain, VOCTest
from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet
from piwise.transform import ToLabel, Colorize
from piwise.visualize import Dashboard

NUM_CHANNELS = 3
NUM_CLASSES = 16

color_transform = Colorize(n=NUM_CLASSES)
image_transform = ToPILImage()
input_transform = Compose([
    Resize(256),
    ToTensor(),
    # Normalize([.485, .456, .406], [.229, .224, .225]),
    Normalize([.5, .5, .5], [.5, .5, .5]),
])
target_transform = Compose([
    Resize(256),
    ToLabel(),
    # Relabel(255, 21),
])


def train(args, model):
    model.train()

    weight = torch.ones(NUM_CLASSES)
    weight[0] = 0.1

    loader = DataLoader(VOCTrain(args.datadir, 'train', input_transform, target_transform),
                        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)

    optimizer = Adam(model.parameters(), lr=1e-5)
    # if args.model.startswith('FCN'):
    #     optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)
    # if args.model.startswith('PSP'):
    #     optimizer = SGD(model.parameters(), 1e-2, .9, 1e-4)
    # if args.model.startswith('Seg'):
    #     optimizer = SGD(model.parameters(), 1e-3, .9)

    if args.steps_plot > 0:
        board = Dashboard(args.port)

    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = []

        for step, (images, labels) in enumerate(loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            if args.steps_plot > 0 and step % args.steps_plot == 0:
                image = inputs[0].cpu().data
                image[0] = image[0] * .5 + .5
                image[1] = image[1] * .5 + .5
                image[2] = image[2] * .5 + .5
                board.image(image,
                            f'input (epoch: {epoch}, step: {step})')
                board.image(color_transform(outputs[0].cpu().max(0, keepdim=True)[1].data),
                            f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                            f'target (epoch: {epoch}, step: {step})')
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average} (epoch: {epoch}, step: {step})')
            if args.steps_save > 0 and step % args.steps_save == 0:
                filename = f'{args.model}-{epoch:03}-{step:04}.pth'
                torch.save(model.state_dict(), filename)
                print(f'save: {filename} (epoch: {epoch}, step: {step})')


def evaluate(args, model):
    save_dir = '/home/nico/Desktop/FCN-8s/'
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    all_metrics_list = []
    for p_id in range(1, 16):
        print('=====pattern %d=====' % p_id)
        loader = DataLoader(VOCTest(args.datadir, p_id, input_transform, target_transform),
                            num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

        targets = []
        preds = []
        for step, (image, label) in enumerate(loader):
            if args.cuda:
                image = image.cuda()
            # inputs = Variable(image)
            targets.append(label.numpy().astype(np.uint8))
            outputs = model(image)
            pred = outputs.detach().cpu().numpy().argmax(axis=1)
            preds.append(pred.astype(np.uint8))

        targets = np.concatenate(targets).flatten() == p_id
        preds = np.concatenate(preds).flatten() == p_id

        print('======start evaluation======')
        metrics_dict = save_metrics_on_results(label_pred=None, label_true=None,
                                               binary_result=preds, binary_mask=targets,
                                               model_name='fcn8s-p%d' % p_id, save_dir=save_dir)

        all_metrics_list.append(metrics_dict)
        df = pd.DataFrame(all_metrics_list)
        df.to_csv('~/Desktop/all_metrics_of_%s.csv' % 'fcn8s')


def main(args):
    Net = None
    if args.model == 'fcn8':
        Net = FCN8
    if args.model == 'fcn16':
        Net = FCN16
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'unet':
        Net = UNet
    if args.model == 'pspnet':
        Net = PSPNet
    if args.model == 'segnet':
        Net = SegNet
    assert Net is not None, f'model {args.model} not available'

    model = Net(NUM_CLASSES)

    if args.cuda:
        model = model.cuda()
    if args.state:
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                                             map_location=lambda storage, loc: storage))

    if args.mode == 'eval':
        evaluate(args, model)
    if args.mode == 'train':
        train(args, model)


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--state')

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('--datadir', default='data')
    parser_eval.add_argument('--batch-size', type=int, default=4)
    parser_eval.add_argument('--num-workers', type=int, default=4)
    # parser_eval.add_argument('image')
    # parser_eval.add_argument('label')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--datadir', default='data')
    parser_train.add_argument('--port', type=int, default=5000)
    parser_train.add_argument('--num-epochs', type=int, default=32)
    parser_train.add_argument('--num-workers', type=int, default=4)
    parser_train.add_argument('--batch-size', type=int, default=4)
    parser_train.add_argument('--steps-loss', type=int, default=50)
    parser_train.add_argument('--steps-plot', type=int, default=100)
    parser_train.add_argument('--steps-save', type=int, default=500)

    main(parser.parse_args())
