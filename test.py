from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from vgg import *



"""
###### in your dataloader

tsfm = transforms.Compose([
    transforms.ToTensor(),            
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

img = cv2.imread('xxx.png')
img_resize = cv2.resize((240, 240))
img_tensor = tsfm(img_resize) # (channels, 240, 240)

######

# shape: (batch_size, channels, 240, 240)


out_ours = model(img_tensor, img2_tensor)
out_vgg = vgg(img_tensor)
out2_vgg = vgg(img2_tensor)


"""



# gamma: (N, C, H, W)
def flow_transfer(i0, i1, gamma):
    delta = gamma[:,:2,:,:]
    ### TODO ###
    out0 = i0
    out1 = i1
    return out0, out1

def image_synthesis(i0, i1, gamma):
    # eq. (2)
    ### TODO ###
    return i0

class Block(nn.Module):
    def __init__(self, Ni, No, strides=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(Ni, 32, 3, strides)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 1)
        self.conv6 = nn.Conv2d(32, No, 3, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flow_refine1 = Block(Ni=9, No=3)
        self.flow_refine2 = Block(Ni=9, No=3)
        self.syn_refine = Block(Ni=9, No=3)
        self.flow_coarse = Block(Ni=6, No=3)

    """
    input: (N, C, H, W)
    """
    def forward(self, i0, i1):
        downsampling = nn.AvgPool2d((3, 3), stride=(2, 2), padding=1)
        upsampling   = nn.Upsample(scale_factor=2, mode='bilinear')
        upsampling4  = nn.Upsample(scale_factor=4, mode='bilinear')
        upsampling8  = nn.Upsample(scale_factor=8, mode='bilinear')

        #x = torch.cat((i0, i1), dim=1) # (N, 2C, H, W)

        i0_down2 = downsampling(i0)
        i1_down2 = downsampling(i1)
        i0_down4 = downsampling(i0_down2)
        i1_down4 = downsampling(i1_down2)
        i0_down8 = downsampling(i0_down4)
        i1_down8 = downsampling(i1_down4)

        gamma3 = F.tanh (self.flow_coarse( torch.cat((i0_down8, i1_down8), dim=1) ))
        U_gamma3 = upsampling(gamma3)
        i0_trans, i1_trans = flow_transfer(i0_down4, i1_down4, U_gamma3)
        gamma3_res = self.flow_refine2( torch.cat((i0_trans, i1_trans, U_gamma3), dim=1)) )
        
        gamma2 = F.tanh(U_gamma3 + gamma3_res)
        U_gamma2 = upsampling(gamma2)
        i0_trans, i1_trans = flow_transfer(i0_down2, i1_down2, U_gamma2)
        gamma2_res = self.flow_refine1( torch.cat((i0_trans, i1_trans, U_gamma2), dim=1)) )

        gamma1 = F.tanh(U_gamma2 + gamma2_res)

        O_gamma3 = upsampling8(gamma3)
        O_gamma2 = upsampling4(gamma2)
        O_gamma1 = upsampling(gamma1)

        I1 = image_synthesis(i0, i1, O_gamma1)
        I2 = image_synthesis(i0, i1, O_gamma2)
        I3 = image_synthesis(i0, i1, O_gamma3)

        I_refine = syn_refine( torch.cat((i0, i1, I1), dim=1) )

        if self.training:
            return I_refine, I1, I2, I3        
        else:
            return I_refine

def tau(a,b,vgg,k=0.001):
    return F.l1_loss(a, b) + k*(F.l2_loss(vgg(a), vgg(b))**2)
    
def train(args, model, vgg, device, train_loader, optimizer, epoch):
    model.train()
    vgg.eval() # ???
    for batch_idx, (i0, i1, y) in enumerate(train_loader):
        i0, i1, y = i0.to(device), i1.to(device), y.to(device)
        optimizer.zero_grad()

        I_refine, I1, I2, I3 = model(i0, i1) 

        ### TODO ###
        loss_refine = tau(I_refine, y, vgg)
        loss_1 = tau(I1, y, vgg)
        loss_2 = tau(I2, y, vgg)
        loss_3 = tau(I3, y, vgg)
        loss = loss_1 + 0.5 * (loss_2 + loss_3) + loss_refine
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        ### TODO ###
        # save model
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
                



def test(args, model, vgg, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    vgg   = vgg16(pretrained=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, vgg, device, train_loader, optimizer, epoch)
        test(args, model, vgg, device, test_loader)
        # save

    # if (args.save_model):
    #     torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()