import time  # Importing the time module for benchmarking and timing

import torch  # PyTorch library for tensor computation and neural networks
import torch.nn as nn  # Importing neural network module from PyTorch

# Importing custom modules, assuming they provide specific functionalities for this code
from gptq import *
from modelutils import *
from quant import *


def get_opt(model):
    # Skips certain initializations to potentially speed up or alter model loading
    def skip(*args, **kwargs):
        pass
    # Overriding PyTorch's default initializations with `skip` to prevent them from running
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    # Importing OPTForCausalLM from the transformers library specifically for OPT models
    from transformers import OPTForCausalLM
    # Loading an OPT model with automatic tensor dtype selection for efficiency
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    # Setting the model's sequence length to its maximum position embeddings
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()  # Disables gradient calculation to save memory and computations
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    # Temporarily disables caching mechanism in the model for this operation
    use_cache = model.config.use_cache
    model.config.use_cache = False
    # Accessing the layers of the model for manipulation
    layers = model.model.decoder.layers

    # Moving parts of the model to the specified device (e.g., GPU) for computation
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    # Checking and moving additional components if they exist
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    # Moving the first layer to the device, as a start for sequential processing
    layers[0] = layers[0].to(dev)

    # Preparing tensors for capturing inputs to each layer
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    # Cache for tracking current sample index and attention mask
    cache = {'i': 0, 'attention_mask': None}

    # Custom module to catch inputs to the first layer and interrupt processing
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError  # Used to interrupt the model's forward pass
    # Replacing the first layer with the Catcher to intercept its inputs
    layers[0] = Catcher(layers[0])
    # Running through the data to capture inputs to the first layer
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    # Restoring the original first layer after capturing its inputs
    layers[0] = layers[0].module

    # Moving the modified parts of the model back to CPU to free GPU memory
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()  # Clearing CUDA cache to free up GPU memory

    # Preparing for the main processing with quantization
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        # Function to add batch data for quantization
        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        # Registering forward hooks to capture data for quantization
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        # Processing inputs through each layer for quantization data collection
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        # Removing hooks after data collection
        for h in handles:
            h.remove()

        # Quantizing weights based on collected data
        for name in subset:
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
            )
            # Storing quantizers for later use
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()
        # Final processing of inputs through quantized layers
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        # Cleaning up and preparing for next layer
        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        # Swapping input and output tensors for the next layer
        inps, outs = outs, inps

    # Restoring original caching setting
    model.config.use_cache = use_cache
    
    return quantizers  # Returning quantizers for all processed layers
@torch.no_grad()  # Disables gradient calculation for efficiency
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    # Preparing input data and adjusting its shape for processing
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    # Temporarily disables model caching for evaluation
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    # Moving model components to the specified device
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    # Preparing tensors for capturing inputs to the first layer
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # Catcher module to intercept and store inputs to the first layer
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError  # Interrupts forward pass
    layers[0] = Catcher(layers[0])
    # Process input data to capture inputs to the first layer
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module  # Restore original first layer

    # Move processed components back to CPU to free GPU memory
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()  # Clear GPU cache

    # Preparing tensors for the output from layers
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    # Sequentially process each layer for evaluation
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        # If nearest neighbor quantization is specified
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        # Process inputs through each layer and store outputs
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()  # Move processed layer back to CPU
        del layer  # Free memory
        torch.cuda.empty_cache()  # Clear GPU cache
        inps, outs = outs, inps  # Swap input and output for next layer

    # Final layer processing if exists
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)  # Move model head to device

    testenc = testenc.to(dev)  # Move test encoding to device
    nlls = []  # List to store negative log likelihoods
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()  # Loss function for calculating NLL
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))  # Calculate perplexity
    print(ppl.item())

    model.config.use_cache = use_cache  # Restore original cache setting
# Function to perform packing on the GPU, optimizing the model for inference
def opt_pack3(model, quantizers):
    # Finding all layers in the model that will be quantized
    layers = find_layers(model)
    # Filtering the layers to only include those with quantizers
    layers = {n: layers[n] for n in quantizers}
    # Applying quantization to the model
    make_quant3(model, quantizers, faster=args.faster_kernel)
    # Finding layers that have been converted to Quant3Linear (quantized)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    # For each quantized layer, perform the packing operation
    for name in qlayers:
        print(name)
        # Moving the quantizers to CPU, if not already
        quantizers[name] = quantizers[name].cpu()
        # Packing the quantized layers for efficient execution
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model  # Returning the packed and optimized model

# Function to load a quantized model from a checkpoint file
def load_quant3(model, checkpoint):
    from transformers import OPTConfig, OPTForCausalLM 
    # Loading the model configuration
    config = OPTConfig.from_pretrained(model)
    # Defining a no-operation function to override default weight initialization
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    # Setting default tensor dtype to half precision for efficiency
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False  # Disable default weight initialization
    torch.set_default_dtype(torch.half)
    # Creating the model instance
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)  # Reverting default dtype to float
    model = model.eval()  # Setting the model to evaluation mode
    # Finding layers to exclude from the quantization process
    layers = find_layers(model)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    # Applying quantization to the model
    make_quant3(model, layers, faster=args.faster_kernel)

    print('Loading model ...')
    # Loading the model state from the checkpoint
    model.load_state_dict(torch.load(checkpoint))
    # Setting the maximum sequence length for the model
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model  # Returning the loaded and quantized model

# Function to adapt the model for multi-GPU execution
def opt_multigpu(model, gpus):
    # Assigning model components to specific GPUs
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(gpus[-1])
    import copy
    # Deep copying the model head to the last GPU
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None}  # Cache for storing the attention mask

    # Module to move inputs and outputs across GPUs during execution
    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            return self.module(*inp, **kwargs)

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))  # Number of layers per GPU
    # Assigning layers to GPUs and wrapping them with MoveModule
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus  # Storing GPU information in the model

# Function to benchmark the model's performance on a given input
def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)  # Move input to the correct device
    torch.cuda.synchronize()  # Ensuring all operations have completed on the GPU

    cache = {'past': None}  # Cache for storing past key/values for Transformer layers

    # Function to clear the cache after each forward pass
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    # Registering the cache clearing function on each layer
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()  # Loss function for verification
        tot = 0.  # Total loss for perplexity calculation

    def sync():
        # Synchronize all GPUs used by the model
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():  # Disabling gradient calculation
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)  # Generating a full attention mask
        times = []  # List to store timing for each token
        for i in range(input_ids.numel()):  # Processing each token sequentially
            tick = time.time()  # Start timing
            out = model(
                input_ids[:, i].reshape((1,-1)),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            sync()  # Ensure all GPU operations have completed
            times.append(time.time() - tick)  # Calculate and store processing time
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)  #
            # Update the cache with the current state for the next iteration
            del out  # Free the output to save memory
        sync()  # Ensure all GPUs have finished processing
        import numpy as np
        # Report the median time taken per token
        print('Median:', np.median(times))
        if check:
            # If checking is enabled, calculate and print the perplexity
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())

# The main block where the script starts execution
if __name__ == '__main__':
    import argparse  # Importing the argparse library for command-line argument parsing
    from datautils import *  # Assuming this imports data-related utilities

    # Setting up an argument parser for command-line options
    parser = argparse.ArgumentParser()

    # Adding expected command-line arguments
    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    # Additional optional arguments for configuration
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    # Parsing the arguments provided via the command-line
    args = parser.parse_args()

    # Loading or initializing the model based on provided arguments
    if args.load:
        model = load_quant3(args.model, args.load)
    else:
        model = get_opt(args.model)
        model.eval()  # Set the model to evaluation mode

    # Preparing data loaders for calibration and testing
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # Sequential quantization and evaluation if applicable
    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = opt_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    # If benchmarking is requested, prepare and execute the benchmark
    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)
    if args.load:
        exit()

    # Evaluation across different datasets if new evaluation is not selected
    datasets = ['wikitext2', 'ptb', 'c4']
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV)

    # Save the quantized and optimized model if a save path is provided
    if args.save:
        opt_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save) 
