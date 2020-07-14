import torch
import os
from torchvision import datasets, transforms


def get_oversample(max_class, img_list):
    print(len(img_list))
    if len(img_list)==max_class:
        return img_list
    indx_list= list(range(0, len(img_list)))
    indx_list= pd.DataFrame({'idx':indx_list})
    indx_list= indx_list.sample(max_class, replace=True, random_state=42)
    indx_list= indx_list['idx'].values.tolist()
    img_list_return=[]
    for indx in indx_list:
        img_list_return.append(img_list[indx])
    return img_list_return


def get_undersample(min_class, img_list):
    print(len(img_list))
    if len(img_list)<=min_class:
        return img_list
    img_list= img_list[:min_class]
    return img_list

def main():
    train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)

    data_list =  train_data.data   ## Assuming that we have an imbalanced dataset
    targets_list= train_data.targets  ### Dataset targets

    cifar_dict={}
    for i in range(num_classes):
        cifar_dict[i]=[]

    for i,cls_idx in enumerate(targets_list):
        cifar_dict[cls_idx].append(data_list[i])

    max_class_required=500  ### If we require each class to be having 500 classes.

    """ If we have to do overrsampling
    you can oversample depending on which class you want"""


    im_data = get_oversample(max_class_required, cifar_dict[cls_idx]) ### cls_idx is any particular class you want to oversample

    """ undersampling """

    min_class_required=50

    im_data = get_undersample(min_class_required, cifar_dict[cls_idx])


    """ Balanced Sampling"""

    targets = train_data.targets
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = sampler, shuffle=False, num_workers=workers, pin_memory=True)


    """ Reweighting """

    if reweight:
        print('using re-weighting using beta value : {}'.format(args.beta))
        img_num_per_cls = get_img_num_per_cls(no_of_classes, args.imb_factor)
        effective_num = 1.0 - np.power(args.beta, img_num_per_cls)
        weights = (1.0 - args.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * int(no_of_classes)
        weights = torch.tensor(weights).float()
        criterion = nn.CrossEntropyLoss(weight=weights).cuda()
    # weights = weights.unsqueeze(0)





    ### Do not pass this sampler to test or validation.
