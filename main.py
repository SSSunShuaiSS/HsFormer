import argparse
import os
import shutil
import socket
import time
import torch
import cv2
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
from math import exp
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import vgg16
import torch.nn.functional as F
import torchvision.utils as vutils
from pytorch_msssim import SSIM
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from Data.MyData import print_log, print_network, save_result_pic, save_current_codes, get_img_transforms_train, \
    get_img_transforms_val
from Data.MyData import MyData, AverageMeter, FileSave
from models.HidingNet import HidingNet
from models.RevealNet import RevealNet

DATA_DIR = "/home/u2308283114/imageNet/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建命令解析器
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
# parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the number of frames')
# parser.add_argument('--imageSize', type=int, default=1024,help='the number of frames')
parser.add_argument('--niter', type=int, default=2000,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')  # 训练后的的图片结果位置
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')  # 针对某一个载体秘密信息单独训练
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')  # 用训练好的模型进行测试后的图片结果位置
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')  # 训练后的最优模型位置存放处
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')
# parser.add_argument('--test', default="/home/u2308283114/cocoImage/test/", help='test mode, you need give the test pics dirs in this param')
# parser.add_argument('--test', default="/home/u2308283114/div2k/test/", help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=320, help='the frequency of print the log on the console')
# parser.add_argument('--logFrequency', type=int, default=2000, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=320, help='the frequency of save the resultPic')


# parser.add_argument('--resultPicFrequency', type=int, default=2500, help='the frequency of save the resultPic')


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# save code of current experiment
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)
    cur_work_dir, mainfile = os.path.split(main_file_path)

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


class LaplacianPyramidLoss(nn.Module):
    def __init__(self, max_levels=3):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False, groups=3)
        # Gaussian kernel for pyramid
        self.gaussian_filter.weight.data = self._gaussian_kernel()
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, input, target):
        laplacian_loss = 0.0

        for _ in range(self.max_levels):
            input_down, input_up = self._down_up(input)
            target_down, target_up = self._down_up(target)

            laplacian_input = input - input_up
            laplacian_target = target - target_up

            laplacian_loss += F.mse_loss(laplacian_input, laplacian_target)

            input = input_down
            target = target_down

        laplacian_loss += F.mse_loss(input, target)
        return laplacian_loss

    def _down_up(self, x):
        down = F.avg_pool2d(x, kernel_size=2, stride=2)
        up = F.interpolate(down, scale_factor=2, mode='bilinear', align_corners=True)
        return down, up

    def _gaussian_kernel(self, channels=3, kernel_size=5):
        kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel


class CombinedLaplacianMSELoss(nn.Module):
    def __init__(self, lambda_mse=1, lambda_laplacian=2, max_levels=3):
        super(CombinedLaplacianMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.laplacian_loss = LaplacianPyramidLoss(max_levels=max_levels)
        self.lambda_mse = lambda_mse
        self.lambda_laplacian = lambda_laplacian

    def forward(self, input, target):
        # 计算 MSE 损失
        mse_loss = self.mse_loss(input, target)

        # 计算拉普拉斯金字塔损失
        laplacian_loss = self.laplacian_loss(input, target)

        # 组合两种损失
        total_loss = self.lambda_mse * mse_loss + self.lambda_laplacian * laplacian_loss
        return total_loss


def main():
    # define global parameters
    global opt, optimizerH, optimizerR, writer, logPath, schedulerH, schedulerR, val_loader, smallestLoss, iters_per_epoch

    #  output configuration
    # 就相当于创建了一个对象，把parser上面定义的一些值都可以使用opt进行调用
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    cudnn.benchmark = True  # 用于提高模型训练速度的一个设置

    #  create dirs to save the result
    if not opt.debug:  # debug为false的时候执行下面的代码
        try:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())  # 当前时间
            experiment_dir = opt.hostname + "_" + cur_time + opt.remark  # 训练的东西
            opt.outckpts += experiment_dir + "/checkPoints"  # 训练的损失数值，可以通过命令行输入进去做到可视化
            opt.trainpics += experiment_dir + "/trainPics"  # 训练的图片效果保存地方
            opt.validationpics += experiment_dir + "/validationPics"  # 测试机训练的效果
            opt.outlogs += experiment_dir + "/trainingLogs"  # 训练的网络结构以及损失等各种详细参数
            opt.outcodes += experiment_dir + "/codes"  # 整个代码保存
            opt.testPics += experiment_dir + "/testPics"  # 验证机图片存放

            # 在原先定义的路径下，添加experiment_dir的路径，在添加最后加上的路径。用来保存各种结果
            if not os.path.exists(opt.outckpts):
                os.makedirs(opt.outckpts)
                # 在上述示例中，如果 opt.outckpts 目录不存在，则使用 os.makedirs() 函数来创建该目录。
                # 请注意，在使用这段代码之前，确保 opt.outckpts 变量已经被正确初始化，并且 os 模块已经导入。
            if not os.path.exists(opt.trainpics):
                os.makedirs(opt.trainpics)
            if not os.path.exists(opt.validationpics):
                os.makedirs(opt.validationpics)
            if not os.path.exists(opt.outlogs):
                os.makedirs(opt.outlogs)
            if not os.path.exists(opt.outcodes):
                os.makedirs(opt.outcodes)
            if (not os.path.exists(opt.testPics)) and opt.test != '':
                os.makedirs(opt.testPics)

            if opt.test != '':
                imgs_dirs = ['cover', 'stego', 'secret', 'revSec']
                test_imgs_dirs = [os.path.join(opt.testPics, x) for x in imgs_dirs]
                for path in test_imgs_dirs:
                    os.makedirs(path)
                opt.testPics = test_imgs_dirs

        except OSError:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)

    if opt.test == '':
        # tensorboardX writer
        writer = SummaryWriter()

        #   get dataset
        train_cover_dataset = MyData(DATA_DIR, 'train', get_img_transforms_train(opt.imageSize))
        train_secret_dataset = MyData(DATA_DIR, 'train', get_img_transforms_train(opt.imageSize))

        val_dataset = MyData(DATA_DIR, 'val', get_img_transforms_val(opt.imageSize))

    else:
        opt.Hnet = "/home/u2308283114/udh/training/g02n3_2024-09-26-19_24_02/checkPoints/netH_epoch_284,sumloss=0.000070,Hloss=0.000031.pth"
        opt.Rnet = "/home/u2308283114/udh/training/g02n3_2024-09-26-19_24_02/checkPoints/netR_epoch_284,sumloss=0.000070,Rloss=0.000051.pth"
        testdir = opt.test
        test_cover_dataset = MyData(testdir, '', get_img_transforms_train(opt.imageSize))

    # HidingNet
    Hnet = HidingNet()  # 创建隐藏网络对象，也就是类的对象
    if opt.cuda:
        Hnet.cuda()  # 将隐藏网络放进cuda，使用gpu训练隐藏网络
    # device = torch.device("cpu")
    # Hnet.to(device)
    # Hnet.cuda()
    # Hnet.initialize_weights()  # 调用初始化参数的方法，用来初始化隐藏网络的参数
    # whether to load pre-trained model
    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet, map_location=device))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).to(device)
    print_network(Hnet)
    # RevealNet
    Rnet = RevealNet()
    if opt.cuda:
        Rnet.cuda()
    # device = torch.device("cpu")
    # Rnet.to(device)
    # Rnet.cuda()
    # Rnet.initialize_weights()
    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet, map_location=device))
    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet).to(device)
    print_network(Rnet)

    # MSE loss
    criterion = CombinedLaplacianMSELoss().cuda()
    # criterion = nn.MSELoss().cuda()

    # training mode
    if opt.test == '':
        # setup optimizer
        optimizerH = optim.AdamW(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.96, patience=6, verbose=True)

        optimizerR = optim.AdamW(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.96, patience=9, verbose=True)

        train_cover_loader = DataLoader(train_cover_dataset, batch_size=opt.batchSize,
                                        shuffle=True, num_workers=int(opt.workers))
        train_secret_loader = DataLoader(train_secret_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize * 2,
                                shuffle=False, num_workers=int(opt.workers))

        iters_per_epoch = len(train_cover_loader)

        smallestLoss = 10000
        print_log("==========  training is beginning  ==========", logPath)
        for epoch in range(opt.niter):
            train_loader = zip(train_cover_loader, train_secret_loader)
            # niter=200,一开始就定义了这个参数，这就是epoch轮数
            # train
            train(train_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
            # 现在开始运行训练的模块，把数据集，轮次数，两个网络以及优化方法都传进去了，criterion = nn.MSELoss().cuda()表明使用的MSE损失函数

            # validation
            val_hloss, val_rloss, val_sumloss = validation(val_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
            # 这个是用来测试损失情况的，这个时候是训练网络一次，就进行一次损失的测试
            # adjust learning rate
            schedulerH.step(val_sumloss)
            # 对隐藏网络损失进行优化，也就是梯度下降，上面已经定义好了学习率优化器等等的，都封装在了schedulerH
            schedulerR.step(val_rloss)

            # save the best model parameters
            if val_sumloss < globals()["smallestLoss"]:
                globals()["smallestLoss"] = val_sumloss
                for file in os.listdir(opt.outckpts):
                    if file.startswith('netH_epoch_') or file.startswith('netR_epoch_'):
                        os.remove(os.path.join(opt.outckpts, file))
                # do checkPointing
                torch.save(Hnet.state_dict(),
                           '%s/netH_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (
                               opt.outckpts, epoch, val_sumloss, val_hloss))
                torch.save(Rnet.state_dict(),
                           '%s/netR_epoch_%d,sumloss=%.6f,Rloss=%.6f.pth' % (
                               opt.outckpts, epoch, val_sumloss, val_rloss))

        writer.close()

    # test mode
    else:
        test_loader = DataLoader(test_cover_dataset, batch_size=opt.batchSize,
                                 shuffle=False, num_workers=int(opt.workers))

        test(test_loader, 0, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
        print(
            "==========  test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/  ==========")


def train(train_loader, epoch, Hnet, Rnet, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  # record loss of H-net
    Rlosses = AverageMeter()  # record loss of R-net
    SumLosses = AverageMeter()  # record Hloss + β*Rloss

    # switch to train mode
    Hnet.train()
    Rnet.train()
    # 上面的都是一些参数，对象等等的设置。现在这个网络里面是空的，什么都没有呢，只是从现在才打开了训练按钮

    start_time = time.time()
    for i, (cover_img, secret_img) in enumerate(train_loader, 0):
        # 在每次迭代中，enumerate函数会返回一个包含两个值的元组：索引i和从train_loader中获取的数据data
        # train_loader里面已经定义了批次大小，所以这儿取出来的数据数目是定义好的，从0开始依次取出
        data_time.update(time.time() - start_time)
        # data_time和start_time是用于计算数据加载时间的计时器。
        # time.time()函数返回当前时间的时间戳，通过计算差值可以得到数据加载所需的时间。

        Hnet.zero_grad()
        Rnet.zero_grad()
        # 在每个批次的训练开始时，需要清除之前计算的梯度，以避免梯度叠加
        this_batch_size = int(cover_img.size()[0])

        # 将载体图像和秘密图像各占总图的一半
        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            # 图片都放进cuda进行运算

        container_img = Hnet(
            torch.cat((cover_img, secret_img), 1))  # put concat_image into H-net and get container image
        errH = criterion(container_img, cover_img)  # loss between cover and container
        Hlosses.update(errH.item(), this_batch_size)

        rev_secret_img = Rnet(container_img)  # put concatenated image into R-net and get revealed secret image

        errR = criterion(rev_secret_img, secret_img)  # loss between secret image and revealed secret image
        Rlosses.update(errR.item(), this_batch_size)

        betaerrR_secret = opt.beta * errR
        # beta在参数里面设定为了0.7，也就是恢复网络的错误率占*0.7
        err_sum = errH + betaerrR_secret
        SumLosses.update(err_sum.item(), this_batch_size)

        err_sum.backward()

        optimizerH.step()
        optimizerR.step()

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\tLoss_H: %.6f Loss_R: %.6f  Loss_sum: %.6f \tdatatime: %.6f \tbatchtime: %.6f' % (
            epoch, opt.niter, i, iters_per_epoch,
            Hlosses.val, Rlosses.val, SumLosses.val, data_time.val, batch_time.val)

        if i % opt.logFrequency == 0:
            # logFrequency在初始系统参数的时候设置的为10，也就是说10个批次打印一次损失的信息
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)

        # genereate a picture every resultPicFrequency steps
        if i % opt.resultPicFrequency == 0:
            # resultPicFrequency设置的是100，每一个轮次对1取余都是0，但是i只有等于0或者100的时候才能打印出来，所以这儿要进行调节
            save_result_pic(this_batch_size,
                            cover_img, container_img.data,
                            secret_img, rev_secret_img.data,
                            epoch, i, opt.trainpics)

    # epcoh log
    epoch_log = "==========  one epoch time is %.4f  ==========" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f      optimizerR_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_Hloss=%.6f\tepoch_Rloss=%.6f\tepoch_sumLoss=%.6f" % (
        Hlosses.avg, Rlosses.avg, SumLosses.avg)
    print_log(epoch_log, logPath)

    if not opt.debug:
        # record lr
        writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
        # record loss
        writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
        writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
        writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)


def validation(val_loader, epoch, Hnet, Rnet, criterion):
    print("==========  validation begin  ==========")
    start_time = time.time()
    Hnet.eval()  # 是将Hnet模型切换到评估模式
    Rnet.eval()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    for i, data in enumerate(val_loader, 0):

        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data
        this_batch_size = int(all_pics.size()[0] / 2)

        cover_img = all_pics[0:this_batch_size, :, :, :]
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        # 数据放入GPU
        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()

        with torch.no_grad():

            container_img = Hnet(torch.cat((cover_img, secret_img), 1))
            # 现在才是真正的开始训练了，上面只是打开了训练的按钮，但是还没有往里面传入数据，现在才开始把图片传进去
            errH = criterion(container_img, cover_img)
            #
            Hlosses.update(errH.item(), this_batch_size)

            rev_secret_img = Rnet(container_img)

            errR = criterion(rev_secret_img, secret_img)
            #
            Rlosses.update(errR.item(), this_batch_size)

            # if i % 50 == 0:
            if i % opt.resultPicFrequency == 0:
                save_result_pic(this_batch_size,
                                cover_img, container_img.data,
                                secret_img, rev_secret_img.data,
                                epoch, i, opt.validationpics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
        writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
        writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    print("==========  validation end  ==========")
    return val_hloss, val_rloss, val_sumloss


def test(test_loader, epoch, Hnet, Rnet, criterion):
    print("==========  test begin  ==========")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()  # record the Hloss in one epoch
    Rlosses = AverageMeter()  # record the Rloss in one epoch

    for i, data in enumerate(test_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step
        # first half of images will become cover images, the rest are treated as secret images
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchSize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()

        with torch.no_grad():

            container_img = Hnet(
                torch.cat((cover_img, secret_img), 1))  # take concat_img as input of H-net and get the container_img
            errH = criterion(container_img, cover_img)  # H-net reconstructed error
            Hlosses.update(errH.item(), this_batch_size)

            rev_secret_img = Rnet(container_img)  # containerImg as input of R-net and get "rev_secret_img"

            errR = criterion(rev_secret_img, secret_img)  # R-net reconstructed error
            Rlosses.update(errR.item(), this_batch_size)
            save_result_pic_test(this_batch_size,
                                 cover_img, container_img.data,
                                 secret_img, rev_secret_img.data,
                                 i, opt.testPics)
            # save_result_pic(this_batch_size,
            #                 cover_img, container_img.data,
            #                 secret_img, rev_secret_img.data,
            #                 epoch, i, opt.testPics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    print("==========  test end  ==========")
    return val_hloss, val_rloss, val_sumloss


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # debug mode will not write logs into files
    if not opt.debug:
        # write logs into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pics, coverImg filePath and secretImg filePath
def save_result_pic(this_batch_size,
                    originalLabelv, ContainerImg,
                    secretLabelv, RevSecImg,
                    epoch, i, save_path):
    if not opt.debug:
        originalFrames = originalLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        containerFrames = ContainerImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        secretFrames = secretLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        revSecFrames = RevSecImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

        showContainer = torch.cat([originalFrames, containerFrames], 0)
        showReveal = torch.cat([secretFrames, revSecFrames], 0)
        # resultImg contains four rows: coverImg, containerImg, secretImg, RevSecImg, total this_batch_size columns
        resultImg = torch.cat([showContainer, showReveal], 0)
        resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(resultImg, resultImgName, nrow=this_batch_size, padding=1, normalize=True)


def save_result_pic_test(this_batch_size,
                         originalLabelv, ContainerImg,
                         secretLabelv, RevSecImg,
                         i, save_path):
    # For testing, save a single picture
    if not opt.debug:
        originalFrames = originalLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        containerFrames = ContainerImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        secretFrames = secretLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        revSecFrames = RevSecImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

        originalName = '%s/cover%d.png' % (save_path[0], i)
        vutils.save_image(originalFrames, originalName, nrow=this_batch_size, padding=1, normalize=True)
        containerName = '%s/stego%d.png' % (save_path[1], i)
        vutils.save_image(containerFrames, containerName, nrow=this_batch_size, padding=1, normalize=True)
        secretName = '%s/secret%d.png' % (save_path[2], i)
        vutils.save_image(secretFrames, secretName, nrow=this_batch_size, padding=1, normalize=True)
        revSecName = '%s/revSec%d.png' % (save_path[3], i)
        vutils.save_image(revSecFrames, revSecName, nrow=this_batch_size, padding=1, normalize=True)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()

