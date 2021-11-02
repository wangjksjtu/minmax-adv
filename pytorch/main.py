import argparse
import json
import time

import torch
import torch.nn as nn

import torchvision.utils
import torchvision.transforms as transforms

import torchattacks
from tqdm import tqdm
import timm

from utils import imshow, image_folder_custom_label


def list_model(keyword=''):
    print (timm.list_models('*%s*' % keyword, pretrained=True))


def get_model(model='inception_v3'):
    # load pretrained model

    ## Pretrained Models ##
    # vgg13, vgg16, vgg19
    # inception_v3, inception_v4, adv_inception_v3, ens_adv_inception_resnet_v2
    # resnet18, resnet34, resnet60
    # efficientnet_b0 - efficientnet_b7
    # vit_{SCALE}_patch16_224, vit_{SCALE}_patch16_384; SCALE=small/base/large

    try:
        class Normalize(nn.Module) :
            def __init__(self, mean, std) :
                super(Normalize, self).__init__()
                self.register_buffer('mean', torch.Tensor(mean))
                self.register_buffer('std', torch.Tensor(std))
                
            def forward(self, input):
                # Broadcasting
                mean = self.mean.reshape(1, 3, 1, 1)
                std = self.std.reshape(1, 3, 1, 1)
                return (input - mean) / std

        # Adding a normalization layer for Inception v3.
        # We can't use torch.transforms because it supports only non-batch images.
        norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        model = nn.Sequential(
            norm_layer,
            timm.create_model(model, pretrained=True)
        )

    except:
        raise NotImplementedError

    return model


def attack(dataloader, model, device, batch_size, atk=None):
    start = time.time()
    correct = 0
    total = 0
    
    pbar = tqdm(total=len(dataloader))
    for i_iter, (images, labels) in enumerate(dataloader):
        
        images = images.to(device)
        labels = labels.to(device)

        if atk is None:
            outputs = model(images)
        else:
            adv_images = atk(images, labels)
            outputs = model(adv_images)

        _, pre = torch.max(outputs.data, 1)

        total += batch_size
        correct += (pre == labels).sum()

        # imshow(torchvision.utils.make_grid(adv_images.cpu().data, normalize=True), 
        #        [label_mapping[dataset.classes[i]] for i in pre])
        pbar.update(1)
    
    pbar.close()
    print('Total elapsed time (sec) : %.2f' % (time.time() - start))
    desc = 'Natural' if atk is None else 'Robust'
    print('%s accuracy: %.2f %%' % (desc, 100 * float(correct) / total))


def ensemble_attack(dataloader, models, device, batch_size, K, atk=None):
    start = time.time()
    correct = 0
    total = 0
    
    pbar = tqdm(total=len(dataloader))
    for i_iter, (images, labels) in enumerate(dataloader):
        
        images = images.to(device)
        labels = labels.to(device)
        
        outputs_K = []
        if atk is None:
            for model in models:
                outputs = model(images)
                outputs_K.append(outputs)
        else:
            adv_images = atk(images, labels)
            for model in models:
                outputs = model(adv_images)
                outputs_K.append(outputs)

        for outputs in outputs_K:
            _, pre = torch.max(outputs.data, 1)

            total += batch_size
            correct += (pre == labels).sum()

        # imshow(torchvision.utils.make_grid(adv_images.cpu().data, normalize=True), 
        #        [label_mapping[dataset.classes[i]] for i in pre])
        pbar.update(1)
    
    pbar.close()
    print('Total elapsed time (sec) : %.2f' % (time.time() - start))
    desc = 'Natural' if atk is None else 'Robust'
    print('%s accuracy: %.2f %%' % (desc, 100 * float(correct) / total))


def main(args):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 25

    class_idx = json.load(open("./data/imagenet_class_index.json"))

    idx2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    label_mapping = {}
    for k in range(len(class_idx)):
        label_mapping[class_idx[str(k)][0]] = class_idx[str(k)][1]

    # NOTE: only if the dataset category is placed by real class name (instead of id, e.g., n07579787)
    # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))] 

    if "inception" in args.model:
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
            
            # Using normalization for pretrained model
            # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406],                     
            #                          std=[0.229, 0.224, 0.225])
                
            # However, DO NOT USE normalization transforms in this section.
            # torchattacks only supports images with a range between 0 and 1.
            # Thus, please refer to the model construction section.
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


    dataset = image_folder_custom_label(root='./data/imagenet-adv/val/', transform=transform, idx2label=idx2label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # images, labels = iter(dataloader).next()
    # print("True Image & True Label")
    # print (images.shape, labels.shape)
    # imshow(torchvision.utils.make_grid(images, normalize=True), [label_mapping[dataset.classes[i]] for i in labels])

    # loading model
    model = get_model(args.model).to(device)
    model.eval()

    atks = [torchattacks.FGSM(model, eps=8/255),
            torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=7),
            torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01),
            torchattacks.RFGSM(model, eps=8/255, alpha=4/255, steps=1),
            torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7),
            torchattacks.FFGSM(model, eps=8/255, alpha=12/255),
            torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=7),
            torchattacks.MIFGSM(model, eps=8/255, decay=1.0, steps=5),
            torchattacks.APGD(model, eps=8/255, steps=10),
            torchattacks.FAB(model, eps=8/255),
            torchattacks.Square(model, eps=8/255),
           ]

    atks = [torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7)]
    print("Standard evaluation with clean images")
    attack(dataloader, model, device, batch_size, atk=None)

    print("Performing adversarial attacks")
    for atk in atks :
        
        print("-"*70)
        print(atk)
        attack(dataloader, model, device, batch_size, atk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating the robustness on ImageNet-1k-val')
    parser.add_argument('-m', '--model', default='inception_v3')
    args = parser.parse_args()

    main(args)
