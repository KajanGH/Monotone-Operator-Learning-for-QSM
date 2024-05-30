from torch import optim
import numpy as np
import os, torch, time
import torch.nn as nn
from tqdm import tqdm


from datetime import datetime
import torch
import torch.autograd as autograd
import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import nibabel as nib
import matplotlib.pyplot as plt


torch.backends.cuda.max_split_size_mb = 512

#@title Set parameters for training
training_epochs = 20 #@param {type:"integer"}
learning_rate = 0.0005 #@param {type:"number"}
learning_rate_lam = 1.0 #@param {type:"number"}
number_of_feature_filters = 35 #@param {type:"integer"}
number_of_layers = 8 #@param {type:"integer"}

### Parameters set for training



input_channels = 1     # input channels for CNN
output_channels = 1    # output channels for CNN
lam_itr = 100          # Lambda for initial SENSE recon
lam_init = 100
cgIter_itr = 10
cgIter_init = 50
err_tol = 1e-4
nFETarget = 500
learning_rate_lipshitz = 0.5e0
clip_wt = 0.8e-6
eps = 0.01
alpha = 0.01

T = torch.tensor(0.98)
e = torch.tensor(1e-3)


outdir = '/home/zcemksu'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


#@title Define forward and adjoint operators \[ $\mathbf A(\cdot)$ and $\mathbf A^H(\cdot)$ \]
class cg_block(nn.Module):
    def __init__(self, cgIter, cgTol):
        super(cg_block, self).__init__()
        self.cgIter = cgIter
        self.cgTol = cgTol

    def forward(self, lhs, rhs, x0):
        fn=lambda a,b: torch.abs(torch.sum(torch.conj(a)*b,axis=[-1,-2,-3,-4]))
        x = x0
        r = rhs-lhs(x0)
        p = r
        rTr = fn(r,r)
        eps=torch.tensor(1e-10)
        for i in range(self.cgIter):
            Ap = lhs(p)
            alpha=rTr/(fn(p,Ap)+eps)
            x = x +  alpha[:,None,None,None,None] * p
            r = r -  alpha[:,None,None,None,None] * Ap
            rTrNew = fn(r,r)
            if torch.sum(torch.sqrt(rTrNew+eps)) < self.cgTol:
                break
            beta = rTrNew / (rTr+eps)
            p = r + beta[:,None,None,None,None] * p
            rTr=rTrNew

        return x


class sense(nn.Module):
    def __init__(self, cgIter):
        super().__init__()

        self.cgIter = cgIter
        self.cg = cg_block(self.cgIter, 1e-9)

# PF: added fft ops for clarity and to reduce copy errors
    def fftn(self, img):
      ksp = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(img, dim=[-1, -2, -3]), dim=[-1, -2, -3], norm="ortho"), dim=[-1, -2, -3])
      return ksp

    def ifftn(self, ksp):
      img = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(ksp, dim=[-1, -2, -3]), dim=[-1, -2, -3], norm="ortho"), dim=[-1, -2, -3])
      return img

# PF: Renamed "magic_square" to "dipole_kernel" for clarity
    def forward(self, img, dipole_kernel, mask):
        mimg = img * mask
        cimg = self.fftn(mimg)
        mcksp = cimg * dipole_kernel
        usksp  = self.ifftn(mcksp)
        usksp = usksp
        return usksp

    def adjoint(self, usksp, dipole_kernel, mask):
        usksp = usksp
        mcksp_adj = self.fftn(usksp)
        cimg_adj = mcksp_adj * (dipole_kernel)
        img = self.ifftn(cimg_adj)
        mimg = img * mask

        return mimg

    def ATA(self, img, dipole_kernel, mask): # PF: For the masked problem this should be F^H D^H F M^H M F^H D F. (masked A is no longer self-adjoint!!)
        usksp = self.forward(img, dipole_kernel, mask)
        out = self.adjoint(usksp, dipole_kernel, mask)

        return out

    def inv(self, x0, rhs, lam, dipole_kernel, mask):

        lhs = lambda x: lam*self.ATA(x, dipole_kernel, mask) + 1.001*x
        out = self.cg(lhs, rhs, x0)

        return out

#@title Construct the DEQ framework using $\mathcal F(\cdot)$ and $\mathcal H_{\theta}(\cdot)$ operators



def complex_to_real(x):
    xr = torch.cat((torch.real(x),torch.imag(x)),dim=1)
    xr = xr.type(torch.float32)
    return xr

def real_to_complex(x):
    re,im = torch.split(x,[1,1],dim=1)
    xc = re + 1j*im
    xc = xc.type(torch.complex64)
    return xc

class convlayer(nn.Module):

    def __init__(self, input_channels, output_channels, last, sn=False):
        super(convlayer, self).__init__()


        self.conv = nn.Conv3d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1,bias=True)
        self.relu = nn.ReLU()
        self.last = last

    def forward(self,x):
        x = self.conv(x)
        if not self.last:
            x = self.relu(x)
        return x

class dwblock(nn.Module):
    def __init__(self, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(dwblock, self).__init__()

        self.num_layers = number_of_layers
        layers = []
        layers.append(convlayer(input_channels*2, features, False, spectral_norm))
        for i in range(1, self.num_layers-1):
            layers.append(convlayer(features, features, False, spectral_norm)) # conv layer
        layers.append(convlayer(features, output_channels*2, True, spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = real_to_complex(self.net(complex_to_real(x)))
        return out



class fwdbwdBlock(nn.Module):
    def __init__(self, A, lam, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(fwdbwdBlock, self).__init__()
        self.dw = dwblock(input_channels, features, output_channels, number_of_layers, spectral_norm)
        self.lam = nn.Parameter(torch.tensor(lam, dtype=torch.float32))
        self.A = A
        self.alpha = torch.tensor(0.1, dtype=torch.float32)

    def forward(self, x, Atb, dipole_kernel, mask):
        z = self.dw(x)
        rhs = (1 - self.alpha) * x + self.alpha * z + self.alpha * self.lam * Atb
        x = self.A.inv(x, rhs, self.alpha * self.lam, dipole_kernel, mask)
        return x


class DEQ(nn.Module):
    def __init__(self,f,K,A_init,lam_init,tol=0.05,verbose=True):
        super().__init__()
        self.K=K
        self.f=f
        self.A_init = A_init
        self.lam_init = lam_init
        self.tol = tol
        self.verbose = verbose

    def forward(self, b, dipole_kernel, mask):
        Atb = self.f.A.adjoint(b, dipole_kernel, mask).to(b.device)
        zero = torch.zeros_like(Atb).to(Atb.device)
        sense_out = self.A_init.inv(zero, self.lam_init * Atb, self.lam_init, dipole_kernel, mask)
        x = sense_out
        with torch.no_grad():
            for blk in range(self.K):
                xold = x
                x = self.f(x, Atb, dipole_kernel, mask)  # Update the forward pass
                errforward = torch.norm(x - xold) / torch.norm(xold)
                if(self.verbose):
                    print(errforward)
                    print("diff", torch.norm(x-xold).cpu().numpy()," xnew ",torch.norm(x).cpu().numpy()," xold ",torch.norm(xold).cpu().numpy())
                if(errforward < self.tol and blk>2):
                    if(self.verbose):
                        print("exiting front prop after ",blk," iterations with ",errforward )
                    break


        z = self.f(x, Atb, dipole_kernel, mask)

        # For computation of Jacobian vector product
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, Atb, dipole_kernel, mask)

       # Backward propagation of gradients
        def backward_hook(grad):
            g = grad
            for i in range(self.K):
                gold = g
                g = autograd.grad(f0,z0,gold,retain_graph=True)[0] + grad
                errback = torch.norm(g-gold)/torch.norm(gold)
                if(errback < self.tol):
                    if(self.verbose):
                        print("exiting back prop after ",blk," iterations with ",errback )
                    break
            g = autograd.grad(f0,z0,gold)[0] + grad
            #g = torch.clamp(g,min=-1,max=1)
            return(g)

        if z.requires_grad == True:
            z.register_hook(backward_hook)


        return z, sense_out, errforward, blk

#@title Estimate Lipschitz of $\mathcal H_{\theta}$
def l2_norm(x):
    return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)


class Lipshitz(nn.Module):
    def __init__(self, u, eps, shap, model,lr=1e8):
        super().__init__()
        self.shap = shap
        self.model = model
        self.lr = lr
        self.eps = eps
        self.u = u
        self.gpu=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.u = self.u.to(self.gpu)
        self.eps = self.eps.to(self.gpu)
        self.v = torch.complex(torch.rand(self.shap,dtype=torch.float32),torch.rand(self.shap,dtype=torch.float32))
        self.v = self.v.to(self.gpu)
        self.v = self.u + self.eps*self.v


        self.v = self.v.requires_grad_(True)


    def compute_ratio(self):
        u_out = self.model(self.u)
        v_out = self.model(self.v)
        loss = l2_norm(u_out - v_out)
        loss = loss/l2_norm(self.u - self.v)
        return loss

    def adverserial_update(self, iters=1,reinit=False):

        if(reinit):
            self.v = torch.complex(torch.rand(self.shap,dtype=torch.float32),torch.rand(self.shap,dtype=torch.float32))
            self.v = self.v.to(self.gpu)
            self.v = self.u + self.eps*self.v

        self.v = self.v.requires_grad_(True)

        for i in range(iters):
            loss = self.compute_ratio()
            loss_sum = torch.sum(loss)
            loss_sum.backward()


            v_grad = self.v.grad.detach()
            v_tmp = self.v.data + self.lr * v_grad
            v_tmp = (v_tmp/torch.norm(v_tmp))*torch.norm(self.u)


            self.v.grad.zero_()

            self.v.data = v_tmp

        self.v = self.v.requires_grad_(False)

        loss_sum = self.compute_ratio()
        return loss_sum

#@title Lipschitz training loss using log-barrier method


def contdifflog(x, T, e):

    condition = (x < (T - e))*torch.tensor(1.0)

    y1 = -torch.log((T - x)*condition + (1.0 - condition))

    y2 = -torch.log(e) + (1/e)*(x - T + e)

    return y1*condition + y2*(1.0 - condition)

#@title Instantiate the model, optimizer and loss functions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A_init = sense(cgIter_init).to(device)
A = sense(cgIter_itr).to(device)

optBlock = fwdbwdBlock(A, lam_itr, input_channels, number_of_feature_filters, output_channels, number_of_layers, spectral_norm=False)
optBlock.alpha = torch.tensor(alpha,dtype=torch.float32)

model = DEQ(optBlock, nFETarget, A_init, lam_init, tol=err_tol, verbose=False)
model = model.to(device)

loss_function = nn.MSELoss()

optimizer = optim.Adam([
            {'params': model.f.dw.parameters(), 'lr': learning_rate},
            {'params': model.f.lam, 'lr': learning_rate_lam}
        ])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)


best_training_loss = 1e10
training_loss_epochs, lip_sense_epochs, err_epochs, nFE_epochs = [], [], [], []



# Ensure Drive is mounted
# PF: removed drive mount to make it work for other users of the sheet (i.e. me)
# drive.mount('/content/drive')



# Load the PyTorch dataloader
#data_path = '/content/drive/MyDrive/Archive (1)/data_sim_analytic'
data_path = '/home/zcemksu/data_sim_analytic'

batch_size=10

class QSMLoader(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = list(Path(path).glob('**/*.npz'))
        self.seed = torch.manual_seed(15091402)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file_path = self.files[item]

        phase, chi, mask = self.load_data(file_path)


        mag = chi - chi.min()
        mag = mag / mag.max()
        mag = mag * mask

        scale = torch.pi / torch.max(torch.abs(phase))
        signal = mag * torch.exp(1j * phase * scale)

        # Add variable noise
        snr = torch.randint(80, 120, (1,))
        signal = signal + ((1. / snr) * (torch.randn(signal.shape) + 1j * torch.randn(signal.shape)))

        phase = torch.angle(signal).type(torch.float32) / scale
        phase = phase * mask

        if torch.sum(chi).item() == 0:
            phase = torch.nan_to_num(phase * 0, nan=0, posinf=0, neginf=0)
            chi = torch.nan_to_num(chi * 0, nan=0, posinf=0, neginf=0)


        return phase.float(), chi.float(), mask.float()

    def load_data(self, file_path):
        # print("File path:", file_path)
        numpy_data = np.load(file_path)
        phase = torch.tensor(numpy_data['phase'], dtype=torch.float32)
        phase = phase.unsqueeze(0)
        chi = torch.tensor(numpy_data['chi'], dtype=torch.float32)
        chi = chi.unsqueeze(0)
        mask = torch.tensor(numpy_data['mask'], dtype=torch.float32)
        mask = mask.unsqueeze(0)
        return phase, chi, mask

dataset = QSMLoader(data_path)

# Assuming dataset is your dataset object
dataset_length = len(dataset)

training_set_size = int(dataset_length * 0.2)
validation_set_size = int(dataset_length * 0.006)
rest_size = dataset_length - training_set_size - validation_set_size
training_set, validation_set, rest_set = torch.utils.data.random_split(dataset, [training_set_size, validation_set_size, rest_size])



training_loader  = DataLoader(training_set, batch_size, shuffle=True)
validation_loader  = DataLoader(validation_set, batch_size, shuffle=False)


# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))


dataiter = iter(training_loader)
phases, susceptibilities, masks = next(dataiter)




print('Dataset size is {}'.format(list(phases.shape)))

eps = torch.as_tensor(eps)

n=64
datashape=phases.shape
final_batch_size=datashape[0]


mask_ones = torch.ones((1, 1, n, n, n))


def generate_3d_dipole_kernel(data_shape, voxel_size, B0_dir):

    kx, ky, kz = torch.meshgrid(
        torch.arange(0, data_shape[1]),
        torch.arange(0, data_shape[0]),
        torch.arange(0, data_shape[2])
    )

    kx = (kx-np.ceil(data_shape[1]/2)) / (2 * voxel_size * torch.max(torch.abs(kx)))
    ky = (ky-np.ceil(data_shape[0]/2)) / (2 * voxel_size * torch.max(torch.abs(ky)))
    kz = (kz-np.ceil(data_shape[2]/2)) / (2 * voxel_size * torch.max(torch.abs(kz)))

    k2 = kx**2 + ky**2 + kz**2
    k2[k2 == 0] = torch.finfo(torch.float32).eps

    D = (1 / 3 - ((kx * B0_dir[1] + ky * B0_dir[0] + kz * B0_dir[2])**2 / k2))
    D = D.unsqueeze(0).unsqueeze(0).expand(1, 1, n, n, n)
    return D

# Example usage:
data_shape = (n, n, n)  # Define the shape of the data
voxel_size = 1.0  # Define the voxel size
B0_dir = torch.tensor([0.0, 0.0, 1.0])  # Define the direction of the B0 field

#csm_data = generate_3d_dipole_kernel(data_shape, voxel_size, B0_dir)
#PF: renamed to dipole_kernel for clarity
dipole_kernel = generate_3d_dipole_kernel(data_shape, voxel_size, B0_dir)
dipole_kernel = dipole_kernel.float()
print(dipole_kernel.shape)

num_chunks = int(1)
sub_list = [i for i in range(num_chunks)]

weights_directory = '/home/zcemksu/trained_model_weights/'

# Ensure the directory exists
if not os.path.exists(weights_directory):
    os.makedirs(weights_directory)


# Training loop
for epoch in tqdm(range(1, training_epochs + 1)):
    start_time = time.time()
    model.train()

    epoch_loss = 0
    dipole_kernel = dipole_kernel.to(device)
    mask_ones = mask_ones.to(device)
    for phase, chi, mask in training_loader:
        phase, chi, mask = phase.to(device), chi.to(device), mask.to(device)
#
        # phase_sim = (A.forward(chi,dipole_kernel, mask_ones))
        # mask = mask.bool()
        # mask = ~mask

        # Generate synthetic training phase
        phase_sim = (A.forward(chi,dipole_kernel,mask_ones))
        phase_sim = phase_sim.to(device)

        # Compute model with gradients
        predicted_fully_sampled, sense_out, err, nFE = model(phase_sim, dipole_kernel, mask)

        # Estimate Lipschitz constant
        u_sense = torch.clone(predicted_fully_sampled).detach()
        lipf_sense = Lipshitz(u_sense, eps, u_sense.shape, optBlock.dw, lr=learning_rate_lipshitz)
        lipf_sense_est = lipf_sense.adverserial_update(iters=15)
        lip_sense_epochs.append(lipf_sense_est.sum().detach().cpu().numpy())

        predicted_fully_sampled = torch.real(predicted_fully_sampled)

        prediction_loss = loss_function(predicted_fully_sampled, chi)
#try removing closs and see if it still converges as effectively
# shows that the monotone operator learning aspect is useful
        closs = clip_wt * contdifflog(lipf_sense_est, T, e)
        ploss = prediction_loss

        loss = ploss + closs

        optimizer.zero_grad()
        epoch_loss = epoch_loss + loss

        epoch_loss = epoch_loss.sum()
        loss = loss.sum()


        loss.backward()
        optimizer.step()

        del lipf_sense_est
        torch.cuda.empty_cache()

    # Update learning rate scheduler
    scheduler.step()



    # Track epoch metrics
    training_loss_epochs.append(epoch_loss)
    nFE_epochs.append(nFE)
    err_epochs.append(err.detach().cpu().numpy())

    ploss = ploss.sum().detach().cpu().numpy()
    closs = closs.sum().detach().cpu().numpy()

    # Print epoch information
    print('Epoch:%d, tloss: %.6f, ploss: %.6f, closs: %.6f, lip_sense: %.3f, alpha: %.3f, lam: %.3f, nFE %d, error %.5f' % (epoch, training_loss_epochs[epoch-1], ploss, closs, lip_sense_epochs[epoch-1].sum(), optBlock.alpha.detach().cpu().numpy(), optBlock.lam.detach().cpu().numpy(), nFE_epochs[epoch-1], err_epochs[epoch-1]))


    # Save model weights every 10th epoch
    if epoch % 10 == 0:
        # Define the file path for this epoch's weights
        weights_path = os.path.join(weights_directory, f'epoch_{epoch}_weights.pth')

        # Save the state dictionary of the model
        torch.save(model.state_dict(), weights_path)

        # Print confirmation message
        print(f"Model weights saved for epoch {epoch} at:", weights_path)


    if training_loss_epochs[epoch - 1] > 2 * best_training_loss:
        scheduler.step()




# Move tensors to CPU, detach from computation graph, and convert to NumPy arrays
training_loss_numpy = [tensor.detach().cpu().numpy() for tensor in training_loss_epochs]

fig, ax = plt.subplots(1, 4, figsize=(30, 6))
print(ax.shape)
ax[0].plot(training_loss_numpy)
ax[0].set_title('Training Loss vs Epochs')
ax[1].plot(lip_sense_epochs)
ax[1].set_title('Lipschitz vs Epochs')
ax[2].plot(nFE_epochs)
ax[2].set_title('Number of forward iterations (nFE) vs Epochs')
ax[3].plot(err_epochs)
ax[3].set_title('Error vs Epochs')








# Move tensors to CPU, detach from computation graph, and convert to NumPy arrays
training_loss_numpy = [tensor.detach().cpu().numpy() for tensor in training_loss_epochs]

fig, ax = plt.subplots(1, 4, figsize=(30, 6))
print(ax.shape)
ax[0].plot(training_loss_numpy)
ax[0].set_title('Training Loss vs Epochs')
ax[1].plot(lip_sense_epochs)
ax[1].set_title('Lipschitz vs Epochs')
ax[2].plot(nFE_epochs)
ax[2].set_title('Number of forward iterations (nFE) vs Epochs')
ax[3].plot(err_epochs)
ax[3].set_title('Error vs Epochs')


nii_path = "/home/zcemksu/QSM_Data"



def get_dim_blocks(dim_in, kernel_size, padding=0, stride=1, dilation=1):
    return (dim_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def extract_patches_3d(x, kernel_size, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    x = x.contiguous()

    channels, depth, height, width = x.shape[-4:]
    d_blocks = get_dim_blocks(depth, kernel_size=kernel_size[0], stride=stride[0], dilation=dilation[0])
    h_blocks = get_dim_blocks(height, kernel_size=kernel_size[1], stride=stride[1], dilation=dilation[1])
    w_blocks = get_dim_blocks(width, kernel_size=kernel_size[2], stride=stride[2], dilation=dilation[2])

    shape = (channels, d_blocks, h_blocks, w_blocks, kernel_size[0], kernel_size[1], kernel_size[2])

    strides = (width*height*depth,
               stride[0]*width*height,
               stride[1]*width,
               stride[2],
               dilation[0]*width*height,
               dilation[1]*width,
               dilation[2])

    x = x.as_strided(shape, strides)
    x = x.permute(1,2,3,0,4,5,6)
    return x

def combine_patches_3d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    channels = x.shape[1]
    d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
    d_dim_in = get_dim_blocks(d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[2], padding[2], stride[2], dilation[2])
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    x = x.view(-1, channels, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
    # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

    x = x.contiguous().view(-1, channels * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out), kernel_size=(kernel_size[1], kernel_size[2]), padding=(padding[1], padding[2]), stride=(stride[1], stride[2]), dilation=(dilation[1], dilation[2]))
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(x, output_size=(d_dim_out, h_dim_out * w_dim_out), kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))
    # (B, C, D, H * W)

    x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)
    # (B, C, D, H, W)

    return x



mask = nib.load('/home/zcemksu/QSM_Data/BrainMaskExtracted.nii.gz')
susceptibility =  nib.load('/home/zcemksu/QSM_Data/Chi.nii.gz')
data =  nib.load('/home/zcemksu/QSM_Data/Frequency.nii.gz')

H,W,D = data.shape

data = torch.Tensor(data.get_fdata()).tile(1,1,1,1,1)
mask = torch.Tensor(mask.get_fdata()).tile(1,1,1,1,1)

print(data.shape)

kernel_size = 64
stride = 32

data_patches = extract_patches_3d(data, kernel_size=kernel_size, stride=stride).to(device)
mask_patches = extract_patches_3d(mask, kernel_size=kernel_size, stride=stride).to(device)
out_patches = torch.ones_like(data_patches)
print(data_patches.shape)


for n_patch_h in range(data_patches.shape[0]):
    for n_patch_w in range(data_patches.shape[1]):
        with torch.no_grad():
            out_patches[n_patch_h, n_patch_w, ...] = torch.real(model(
                data_patches[n_patch_h, n_patch_w, ...] , dipole_kernel,
                mask_patches[n_patch_h, n_patch_w, ...] )[0])


out_patches = out_patches.permute(3,0,1,2,4,5,6).unsqueeze(dim=0)
output = combine_patches_3d(out_patches, kernel_size=kernel_size, output_shape=(1,1,H,W,D), stride=stride)
output = output[0,0,...].cpu()
img = nib.Nifti1Image(output, susceptibility.affine, susceptibility.header)
nib.save(img, '/home/zcemksu/mol_output.nii')

