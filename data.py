import os
import time
import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.datasets.utils import download_url


dtype = torch.float32


class ImageData(Dataset):
    def __init__(self, list_url, list_goodsNo, shape, timeout=30, max_retry=3):
        self.list_url = list_url
        self.list_goodsNo = list_goodsNo
        self.shape = shape
        self.timeout = timeout
        self.max_retry = max_retry
        
        self.transforms = transforms.Compose([
            transforms.Resize((shape[1], shape[2])),
            transforms.ToTensor()
        ])

        resource_dir = "./"
        self.image_dir = os.path.join(resource_dir, "_image/")
        self.failed = []
    
    def __len__(self):
        return len(self.list_url)
        
    def __getitem__(self, index):
        url = self.list_url[index].strip()
        goodsNo = self.list_goodsNo[index]

        file_path = self.__save(url, goodsNo)
        img = self.__load(file_path)

        item = self.transforms(img)
        return item.type(dtype)
        
    def __save(self, url, goodsNo):
        last_element = url.split("/")[-1]
        file_name_orig = last_element.split("?")[0]
        file_name_ext = file_name_orig.split(".")[-1]
        file_name = f"{goodsNo}.{file_name_ext}"
        file_path = os.path.join(self.image_dir, file_name)

        if os.path.exists(file_path):
            return file_path

        retry = 0
        while retry < self.max_retry:
            try:

                # import urllib, shutil
                # with urllib.request.urlopen(url, timeout=self.timeout) as response:
                #     with open(file_path, 'wb') as f:
                #         shutil.copyfileobj(response, f)

                download_url(url, self.image_dir, filename=file_name)
                is_except = False
                break
            except Exception as e:
                print(f"{goodsNo} >>> {e}")
                time.sleep(2 ** retry)
                retry += 1
                is_except = True
        
        if is_except:
            self.failed.append(goodsNo)
        return file_path
        
    def __load(self, file_path):
        try:
            img = transforms.functional.Image.open(file_path, mode="r")
        except FileNotFoundError:
            img = transforms.ToPILImage()(
                torch.full(self.shape, fill_value=127.5)
            )
        return img.convert("RGB")
