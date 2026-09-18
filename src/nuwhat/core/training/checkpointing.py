import torch

def load_pretrained(encoder, heads, file_name):
    checkpoint = torch.load(file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])

    ## Load heads as requested
    for name, head in heads.items():
        key = f'{name}_head_state_dict'
        if key in checkpoint:
            head.module.load_state_dict(checkpoint[key])
    return

def load_checkpoint(encoder, heads, optimizer, scheduler, state_file_name):
    checkpoint = torch.load(state_file_name, map_location='cpu')
    encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'].cpu())
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    ## If we have a scheduler, load it
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    ## Load heads as requested
    for	name, head in heads.items():
        key = f'{name}_head_state_dict'
        if key in checkpoint:
            head.module.load_state_dict(checkpoint[key])

    ## Load metrics
    metrics = defaultdict(list, checkpoint.get("metrics", {}))
    
    start_epoch = checkpoint['epoch'] + 1

    return start_epoch, metrics

def save_checkpoint(encoder, heads, optimizer, scheduler, state_file_name, iteration, metrics, args):

    state_dict = {
        'epoch': iteration,
        'encoder_state_dict': encoder.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'metrics': dict(metrics),
        'args':vars(args)
    }

    ## Save a scheduler if we have one
    if scheduler is not None:
        state_dict['scheduler_state_dict'] = scheduler.state_dict()
    
    ## Save heads as needed:
    for name, head in heads.items():
        state_dict[f'{name}_head_state_dict'] = head.module.state_dict()

    torch.save(state_dict, state_file_name)
    
