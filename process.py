#process.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torchvision import transforms
from PIL import Image

from os import listdir
from json import dumps
import sys

class Custom_Dataset(Dataset):
    def __init__(self, folder_path, transforms=None):
        self.folder_path = folder_path
        self.image_list = listdir(folder_path)
        self.data_len = len(self.image_list)
        self.transforms = transforms

    def __getitem__(self, index):
        
        single_image_path = self.image_list[index]
        im_as_im = Image.open(self.folder_path+ '/' + single_image_path)
        if self.transforms is not None:
            im_as_ten = self.transforms(im_as_im)
        return (im_as_ten, single_image_path)

    def __len__(self):
        
        return self.data_len

def load_Model(model_path = '../input/model-testt/aerialmodel.pth'):
    model=torch.load(model_path,map_location=device)
    model.eval()
    return model

def dataset_Loader(data_path):
    transformations = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                        ])
    custom_dataset = Custom_Dataset(data_path, transformations)
    dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,batch_size=1,shuffle=False)
    return dataset_loader

def make_Process_results(data_path):
    to_json = {}
    model = load_Model()
    for i in dataset_Loader(data_path):
        
        pred = model(i[0].to(device)).data.cpu().numpy()
        to_json[i[1][0]] = int(1-pred.argmax())*'fe' + 'male'
    with open('process_results.json', 'w') as f:
        f.write(dumps(to_json))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    make_Process_results('../input/face-named/Test_face')
    if len (sys.argv) > 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        make_Process_results(sys.argv[1])
        print ('Файл создан')
    else:
        print ('Укажите путь к данным')