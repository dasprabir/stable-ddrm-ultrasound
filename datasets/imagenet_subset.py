import torch.utils.data as data
import torchvision.transforms as transforms
import os
import scipy.io
import numpy as np
from PIL import Image

class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):#
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def default_loader2(path):
    image = scipy.io.loadmat(path)['data']
    image = image.astype(np.float32)
    #image = image
    #image = (image - image.min()) / (image.max() - image.min())
    #image = np.expand_dims(image, axis=0)
    #image = image / np.abs(image.max())
    image = np.stack((image,) * 3, axis=-1)
    return image

class ImageDataset(data.Dataset):

    def __init__(self,
                 root_dir,
                 meta_file,
                 transform=None,
                 image_size=128,
                 normalize=True):
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transform = transforms.Compose([
                    CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transform = transforms.Compose([
                    CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ])
        with open(meta_file) as f:
            lines = f.readlines()
        print("building dataset from %s" % meta_file)
        self.num = len(lines)
        self.metas = []
        self.classifier = None
        suffix =  ".JPEG"
        for line in lines:
            line_split = line.rstrip().split()
            if len(line_split) == 2:
                self.metas.append((line_split[0] , int(line_split[1])))
            else:
                self.metas.append((line_split[0] , -1))
        print("read meta done")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # Extract the base filename without an extension
        base_filename = os.path.splitext(self.metas[idx][0])[0]
        cls = self.metas[idx][1]

        # Attempt to load the image with each possible extension
        for ext in ['.mat']: #'.jpg','.mat', '.JPEG', '.png'
            filename = os.path.join(self.root_dir, base_filename + ext)
            if os.path.isfile(filename):
                img = default_loader2(filename)
                break
        else:
            raise FileNotFoundError(f"No image found for {base_filename} with extensions .jpg, .JPEG, or .png")

        # transform
        if self.transform is not None:
            img = self.transform(img)

        return img, cls