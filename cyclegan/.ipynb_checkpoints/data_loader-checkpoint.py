from PIL import Image
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, dataset_name, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_img(self, path):
        img = Image.open(path).resize(self.img_res)
        return img

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        image_filenames = glob('/home/ubuntu/colorizer-applied-dl/{}/*'.format(data_type))
        batch_images = np.random.choice(image_filenames, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.load_img(img_path)
            if domain == 'B':
                imgs.append(np.array(img)[np.newaxis, :, :, :])
            else:
                bw_img = np.array(img.convert('L')) 
                imgs.append(bw_img[np.newaxis,:,:,np.newaxis]) # B/W
            # if not is_testing:
            #     if np.random.random() > 0.5:
            #         img = np.fliplr(img)
        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        #path_A = glob('./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        #path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))
        image_filenames = glob('/home/ubuntu/colorizer-applied-dl/{}/*'.format(data_type))
        self.n_batches = int(len(image_filenames) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        paths = np.random.choice(image_filenames, total_samples, replace=False)

        for i in range(self.n_batches-1):
            curr_batch = paths[i*batch_size: (i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_path in curr_batch:
                img = self.load_img(img_path)
                bw_img = np.array(img.convert('L')) 
                
                img_A = bw_img[np.newaxis,:,:,np.newaxis] # B/W
                img_B = np.array(img)[np.newaxis, :, :, :] # Color

                # if not is_testing and np.random.random() > 0.5:
                #     img_A = np.fliplr(img_A)
                #     img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
            yield imgs_A, imgs_B