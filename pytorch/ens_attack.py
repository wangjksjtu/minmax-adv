import argparse
import json
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchattacks
from tqdm import tqdm
import timm

from utils import image_folder_custom_label


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


def ens_attack(dataloader, models, device, batch_size, atk=None):
    start = time.time()
    K = len(models)
    corrects = [0] * K
    totals = [0] * K
    adv_maps = [[] for _ in range(K)]

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

        for i, outputs in enumerate(outputs_K):
            _, pre = torch.max(outputs.data, 1)

            totals[i] += batch_size
            corrects[i] += (pre == labels).sum()        
            adv_maps[i].append(pre != labels)

        pbar.update(1)
    
    pbar.close()
    print('Total elapsed time (sec) : %.2f' % (time.time() - start))
    desc = 'Natural' if atk is None else 'Robust'
    for i in range(K):
        print('[model %d] %s accuracy: %.2f %%' % (i, desc, 100 * float(corrects[i]) / totals[i]))
        adv_maps[i] = torch.cat(adv_maps[i])

    print (len(adv_maps))
    asr_all = torch.ones(totals[0], dtype=torch.bool).to(labels.device)
    for adv_map in adv_maps:
        print (torch.sum(asr_all))
        asr_all = torch.logical_and(adv_map, asr_all)

    print ('ASR_all: %.2f %%' % (100 * torch.sum(asr_all) / float(totals[0])))


def main(args):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    class_idx = json.load(open("./data/imagenet_class_index.json"))

    idx2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    label_mapping = {}
    for k in range(len(class_idx)):
        label_mapping[class_idx[str(k)][0]] = class_idx[str(k)][1]

    # NOTE: only if the dataset category is placed by real class name (instead of id, e.g., n07579787)
    # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))] 

    batch_size = args.batch_size
    models_name = args.models.split(",")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = image_folder_custom_label(root='./data/imagenet-adv/val/', transform=transform, idx2label=idx2label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # loading model
    models = []
    for model_name in models_name:
        model = get_model(model_name).to(device)
        model.eval()
        models.append(model)

    atks = [torchattacks.AVGPGD(models, eps=4/255, alpha=1/255, steps=7),
            torchattacks.APGDA(models, eps=4/255, alpha=1/255, steps=7, beta=100, gamma=10)
           ]

    print("Standard evaluation with clean images")
    ens_attack(dataloader, models, device, batch_size, atk=None)

    print("Performing adversarial attacks")
    for atk in atks :
        print("-"*70)
        print(atk)
        ens_attack(dataloader, models, device, batch_size, atk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating the robustness on ImageNet-1k-val')
    parser.add_argument('-m', '--models', default='vgg16,resnet34,efficientnet_b1')
    parser.add_argument('-b', '--batch_size', default=10)
    args = parser.parse_args()

    main(args)
