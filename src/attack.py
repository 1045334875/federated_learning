import os
from array import *
from skimage import io
import torchvision.datasets.mnist as mnist
import torch
from torchvision import datasets, transforms

# def attack(args):
# if args.dataset == 'mnist':
#     data_dir = '../data/mnist/'
# else:
#     data_dir = '../data/fmnist/'

root='../data/mnist/'
apply_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

    # train_set = datasets.MNIST(data_dir, train=True, download=False,
    #                                 transform=apply_transform)

    # test_set = datasets.MNIST(data_dir, train=False, download=False,
    #                                 transform=apply_transform)

    # root="../data/fmnist/"
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
        )
print("training set :",train_set[0].size())
print("test set :",test_set[0].size())


def convert_to_img(train=True):
    if(train):
        f=open(root+'train_test1.txt','w')
        data_path=root+'/train_false_test/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        false_set = array('B')
        lowsum=0
        sum=0
        for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
            if(label==7): 
                newlabel=1
                lowsum+=1
            else: 
                newlabel=label

            sum+=1
            false_set.append(newlabel)
            img_path=data_path+str(i)+'_'+str(label.item())+'.jpg'
            io.imsave(img_path,img.numpy())
            f.write(img_path+' '+str(newlabel)+'\n')
            
        f.write("lowsum: "+str(lowsum)+'\n')
        f.write("sum: "+str(sum)+'\n')
        f.close()
        hexval = "{0:#0{1}x}".format(sum,6) # number of files in HEX
        hexval = '0x' + hexval[2:].zfill(8)
        header = array('B')
        header.extend([0,0,8,1])
        header.append(int('0x'+hexval[2:][0:2],16))
        header.append(int('0x'+hexval[2:][2:4],16))
        header.append(int('0x'+hexval[2:][4:6],16))
        header.append(int('0x'+hexval[2:][6:8],16))
        false_set = header + false_set
        output_file = open(root+'false-labels.idx1-ubyte', 'wb')
        false_set.tofile(output_file)
        output_file.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = data_path+ str(i) +'_'+str(label)+ '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()

convert_to_img(True)
# convert_to_img(False)