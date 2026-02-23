import os
import sys
import json
from tqdm import tqdm
import torch
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     #transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   #transforms.Grayscale(num_output_channels=1),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])}

    DATA_SET_PATH = 'Poite'
    
    train_dataset = datasets.ImageFolder(root=DATA_SET_PATH,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=DATA_SET_PATH,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation with {} classes".format(train_num,
                                                                           val_num, len(flower_list)))
    epochs = 100
    best_acc = 0.0 

    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            print(type(images))
            print(images.shape)
            to_pil = ToPILImage()
            image_pil = to_pil(images[0])
            image_pil.show()
            img_gray_avg = images.mean(axis =1, keepdim=True)
            print(img_gray_avg.shape)
            to_pil = ToPILImage()
            image_pil = to_pil(img_gray_avg[0])
            image_pil.show()

            img_gray_avg_R = img_gray_avg[0]
            img_gray_avg_G = img_gray_avg[0]
            img_gray_avg_B = img_gray_avg[0]

            images[0,0,:,:] = img_gray_avg[0,:,:]
            images[0,1,:,:] = img_gray_avg[0,:,:]
            images[0,2,:,:] = img_gray_avg[0,:,:]


            print(images.shape)

            to_pil = ToPILImage()
            image_pil = to_pil(images[0,0])
            image_pil.show()
            to_pil = ToPILImage()
            image_pil = to_pil(images[0,1])
            image_pil.show()
            to_pil = ToPILImage()
            image_pil = to_pil(images[0,2])
            image_pil.show()

if __name__ == '__main__':
    main()