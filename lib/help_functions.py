import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def load_hdf5(infile):        #加载hdf5文件
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands # "image"是写入的时候规定的字段 key-value
    return f["image"][()]           # 调用方法 train_imgs_original = load_hdf5( file_dir )

def write_hdf5(arr,outfile):  #写入hdf5文件
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#convert RGB image in black and white
# 将RGB图像转换为Gray图像
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays #[Nimgs, channels, height, width]
    assert (rgb.shape[1]==3)    #确定是否为RGB图像
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3])) # 确保张量形式正确
    return bn_imgs

#group a set of images row per columns
#利用已知信息进行分组显示
#对数据集划分，进行分组显示，totimg图像阵列
def group_images(data,per_row):  # data：数据  per_row：每行显示的图像个数
    assert data.shape[0]%per_row==0 # data=[Nimgs, channels, height, width]
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow   # 用于显示
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)): # data.shape[0]/per_row 行数
        stripe = data[i*per_row]  # 相当于matlab中的 data(i*per_row, :, :, :) 一张图像
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)  # 每per_row张图像拼成一行
        all_stripe.append(stripe)  # 加入列表
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0) # 每行图像进行拼凑 共len(all_stripe)行
    return totimg


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png') #保存
    return img


#prepare the mask in the right shape for the Unet
# 将金标准图像改写成模型输出形式
def masks_Unet(masks): # size=[Npatches, 1, patch_height, patch_width]
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w)) # 单像素建模
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))   # 二分类输出
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1  # 金标准图像的反转
                new_masks[i,j,1]=0  # 金标准图像
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
    return new_masks

# 网络输出转换成图像子块
# 网络输出 size=[Npatches, patch_height*patch_width, 2]
def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2  # 确认是否为二分类
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original": # 网络概率输出
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1] #pred[:, :, 0] 是反分割图像输出 pred[:, :, 1]是分割输出
    elif mode=="threshold": # 网络概率-阈值输出
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
        # 改写成(Npatches,1, height, width)
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images
