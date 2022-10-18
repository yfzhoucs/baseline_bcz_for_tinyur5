import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
import clip
import contextlib


# courtesy: https://github.com/darkstar112358/fast-neural-style/blob/master/neural_style/transformer_net.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            print('downsampled')
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        print()
        return out


# https://github.com/rosinality/film-pytorch/blob/master/model.py
class ResBlock(nn.Module):
    def __init__(self, filter_size):
        super().__init__()

        self.conv1 = nn.Conv2d(filter_size, filter_size, [1, 1], 1, 1)
        self.conv2 = nn.Conv2d(filter_size, filter_size, [3, 3], 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(filter_size, affine=False)

    def forward(self, input, gamma, beta):
        out = self.conv1(input)
        resid = F.relu(out)
        out = self.conv2(resid)
        out = self.bn(out)

        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        out = gamma * out + beta

        out = F.relu(out)
        out = out + resid

        return out


class ImgEncoder(nn.Module):
    def __init__(self, img_size=224, embedding_size=256, ngf=64, channel_multiplier=4, input_nc=3):
        super(ImgEncoder, self).__init__()
        self.layer1 = nn.Sequential(nn.ReflectionPad2d((3,3,3,3)),
                                    nn.Conv2d(input_nc,ngf,kernel_size=7,stride=1),
                                    nn.InstanceNorm2d(ngf),
                                    nn.ReLU(True))
        
        self.layer2 = nn.Sequential(nn.Conv2d(ngf,ngf*channel_multiplier//2,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*channel_multiplier//2),
                                   nn.ReLU(True))
        
        self.layer3 = nn.Sequential(nn.Conv2d(ngf*channel_multiplier // 2,ngf*channel_multiplier,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*channel_multiplier),
                                   nn.ReLU(True))

        self.layer4 = nn.Sequential(nn.Conv2d(ngf*channel_multiplier,ngf*channel_multiplier,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*channel_multiplier),
                                   nn.ReLU(True))

        self.resblocks = nn.Sequential(ResBlock(ngf*channel_multiplier),
                                    ResBlock(ngf*channel_multiplier),
                                    ResBlock(ngf*channel_multiplier))


        self.film = nn.Linear(embedding_size, ngf*channel_multiplier * 2 * 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, task_embed):

        x = x.permute(0, 3, 1, 2)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)


        film = self.film(task_embed.squeeze()).chunk(3 * 2, 1)

        for i, resblock in enumerate(self.resblocks):
            out = resblock(out, film[i * 2], film[i * 2 + 1])

        out = self.avgpool(out).squeeze()
        return out


class Controller(nn.Module):
    def __init__(self, num_traces=3, num_weight_points=11, embedding_size=128):
        super(Controller, self).__init__()

        self.layer1_1 = nn.Linear(embedding_size, 16 * 16 * 8)
        self.layer2 = nn.Linear(16 * 16 * 8, 16 * 16 * 4)
        self.layer3 = nn.Linear(16 * 16 * 4, 16 * 16 * 1)
        self.layer4 = nn.Linear(16 * 16 * 1, num_traces * num_weight_points)

        self.layer1_bn = nn.BatchNorm1d(16 * 16 * 4)
        self.layer2_bn = nn.BatchNorm1d(16 * 16 * 4)
        self.layer3_bn = nn.BatchNorm1d(16 * 16 * 4)

    def forward(self, goal_embedding):
        x = F.selu(self.layer1_1(goal_embedding))
        x = F.selu(self.layer2(x))
        x = F.selu(self.layer3(x))
        x = self.layer4(x)
        return x


class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


class Backbone(nn.Module):
    def __init__(self, img_size, embedding_size=256, num_traces_in=7, num_traces_out=10, num_weight_points=91, input_nc=3, device=torch.device('cuda')):
        super(Backbone, self).__init__()

        self.device = device
        self.num_traces_in = num_traces_in
        self.num_traces_out = num_traces_out
        self.num_weight_points = num_weight_points
        self.embedding_size = embedding_size

        # Visual Pathway
        self.visual_encoder = ImgEncoder(input_nc=input_nc, embedding_size=embedding_size)

        # Task Pathway
        self.task_id_encoder, _ = clip.load("ViT-B/32", self.device)
        self.task_id_embedding_narrower = nn.Linear(512, embedding_size)

        self.controller_xyz = Controller(num_traces=3, num_weight_points=num_weight_points, embedding_size=embedding_size)
        self.controller_rpy = Controller(num_traces=6, num_weight_points=num_weight_points, embedding_size=embedding_size)
        self.controller_grip = Controller(num_traces=1, num_weight_points=num_weight_points, embedding_size=embedding_size)

    def _img_pathway_(self, img, task_embed):
        # Comprehensive Visual Encoder. img_embedding is the square token list
        img_embedding = self.visual_encoder(img, task_embed)
        return img_embedding

    def _task_id_pathway_(self, lang):
        with torch.no_grad():
            task_embedding = self.task_id_encoder.encode_text(lang)
        task_embedding = task_embedding.float()
        task_embedding = self.task_id_embedding_narrower(task_embedding)
        return task_embedding

    def forward(self, img, sentence, phis):

        # Task Pathway
        task_embed = self._task_id_pathway_(sentence)

        # Image Pathway
        img_embed = self._img_pathway_(img, task_embed)

        # Controller
        dmp_weights_xyz = self.controller_xyz(img_embed)
        dmp_weights_rpy = self.controller_rpy(img_embed)
        dmp_weights_grip = self.controller_grip(img_embed)

        dmp_weights_xyz = dmp_weights_xyz.reshape(img.shape[0], 3, self.num_weight_points)
        dmp_weights_rpy = dmp_weights_rpy.reshape(img.shape[0], 6, self.num_weight_points)
        dmp_weights_grip = dmp_weights_grip.reshape(img.shape[0], 1, self.num_weight_points)

        dmp_weights = torch.cat((dmp_weights_xyz, dmp_weights_rpy, dmp_weights_grip), axis=1)
        
        centers = torch.tensor(np.linspace(0.0, 1.0, self.num_weight_points, dtype=np.float32)).to(self.device)
        centers = centers.unsqueeze(0).unsqueeze(1).repeat(img.shape[0], self.num_traces_out, 1).reshape(img.shape[0] * self.num_traces_out, self.num_weight_points)
        dmp_weights = dmp_weights.reshape(img.shape[0] * self.num_traces_out, self.num_weight_points)
        phis = phis.reshape(img.shape[0] * self.num_traces_out, phis.shape[-1])
        trajectory = Interp1d()(centers, dmp_weights, phis)
        trajectory = trajectory.reshape(img.shape[0], self.num_traces_out, phis.shape[-1])
        dmp_weights = dmp_weights.reshape(img.shape[0], self.num_traces_out, self.num_weight_points)

        return trajectory