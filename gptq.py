import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *  # Importing all functions and classes from quant module

DEBUG = False  # Debug mode flag

torch.backends.cuda.matmul.allow_tf32 = False  # Disabling TF32 for matmul to ensure precision
torch.backends.cudnn.allow_tf32 = False  # Disabling TF32 for cuDNN to ensure precision

class GPTQ:
    def __init__(self, layer):
        self.layer = layer  # Store the layer to be quantized
        self.dev = self.layer.weight.device  # Device where the layer's weights are stored
        W = layer.weight.data.clone()  # Clone the weights of the layer
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)  # Flatten weights for Conv2d layers
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()  # Transpose weights for Conv1D layers from transformers
        self.rows = W.shape[0]  # Number of rows in the weight matrix
        self.columns = W.shape[1]  # Number of columns in the weight matrix
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)  # Initialize H matrix for quantization
        self.nsamples = 0  # Initialize the number of samples added

    def add_batch(self, inp, out):
        if DEBUG:  # Store inputs and outputs in debug mode
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:  # Ensure input has batch dimension
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:  # Reshape input for Linear and Conv1D layers
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )  # Unfold operation for Conv2d layers to transform input
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)  # Update H matrix with new samples
        self.nsamples += tmp  # Update the number of samples
        inp = math.sqrt(2 / self.nsamples) * inp.float()  # Scale input
        self.H += inp.matmul(inp.t())  # Update H matrix with outer product of scaled input

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)  # Flatten weights for Conv2d layers
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()  # Transpose weights for Conv1D layers from transformers
        W = W.float()

        tick = time.time()  # Start timing for quantization process

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)  # Find quantization parameters if not ready

        H = self.H
        del self.H  # Delete H to free memory
        dead = torch.diag(H) == 0  # Identify dead neurons
        H[dead, dead] = 1  # Avoid division by zero for dead neurons
        W[:, dead] = 0  # Zero out weights for dead neurons

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)  # Prepare quantizers for static groups if enabled

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)  # Sort columns by diagonal of H
            W = W[:, perm]
            H = H[perm][:, perm]  # Apply sorting permutation to W and H
            invperm = torch.argsort(perm)  # Inverse permutation for later use

        Losses = torch.zeros_like(W)  # Initialize loss matrix
        Q = torch.zeros_like(W)  # Initialize quantized weight matrix

        damp = percdamp * torch.mean(torch.diag(H))  # Calculate damping factor
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp  # Apply damping to diagonal of H
        H = torch.linalg.cholesky(H)  # Cholesky decomposition of H
        H = torch.cholesky_inverse(H)  # Inverse of Cholesky decomposition
        H = torch.linalg.cholesky(H, upper=True)  # Cholesky decomposition of inverse H
        Hinv = H  # Inverse H for weight update

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)  # Calculate block size
            count = i2 - i1

            W1 = W[:, i1:i2].clone()  # Clone a block of weights
            Q1 = torch.zeros_like(W1)  # Initialize quantized block
            Err1 = torch.zeros_like(W1)  # Initialize error matrix for block
            Losses1 = torch.zeros_like(W1)  # Initialize losses for block
            Hinv1 = Hinv[i1:i2, i1:i2]  # Extract block from inverse H

            for i in range(count):
                w = W1[:, i]  # Select a column of weights
                d = Hinv1[i, i]  # Diagonal element of inverse H for the column

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)  # Find quantization parameters for a group
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]  # Adjust index for sorted order
                        self.quantizer = groups[idx // groupsize]  # Select quantizer for static group

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()  # Quantize the column
                Q1[:, i] = q  # Store quantized column
                Losses1[:, i] = (w - q) ** 2 / d ** 2  # Calculate loss for the column

                err1 = (w - q) / d  # Calculate error for the column
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))  # Adjust the weights for the current and subsequent columns within the block based on the quantization error and inverse H matrix
                Err1[:, i] = err1  # Store the per-column quantization error

            Q[:, i1:i2] = Q1  # Update the quantized weight matrix for the current block
            Losses[:, i1:i2] = Losses1 / 2  # Update the loss matrix for the current block

            # Propagate the adjustments due to quantization error to the remaining weights outside the current block
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:  # If in debug mode, evaluate the impact of quantization on the layer's output error and log the total quantization loss
                self.layer.weight.data[:, :i2] = Q[:, :i2]  # Apply the quantized weights to the layer up to the current block for debugging
                self.layer.weight.data[:, i2:] = W[:, i2:]  # Update the layer's weights with the adjusted values for the rest
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))  # Print the sum of squared differences between the actual output and expected output
                print(torch.sum(Losses))  # Print the accumulated quantization loss

        torch.cuda.synchronize()  # Ensure all CUDA operations are completed before proceeding
        print('time %.2f' % (time.time() - tick))  # Print the total time taken for the quantization process
        print('error', torch.sum(Losses).item())  # Print the total quantization error

        if actorder:  # If the weights were reordered for quantization, apply the inverse permutation to restore the original order
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()  # Transpose the quantized weights back for Conv1D layers
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)  # Update the layer with the quantized weights, maintaining the original shape and data type
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))  # In debug mode, print the sum of squared differences after quantization

    def free(self):
        # Clean-up method to free memory by deleting stored inputs, outputs, and intermediate matrices
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()  # Clear CUDA memory cache to free up GPU memory

