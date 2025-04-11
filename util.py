import torch
from collections import defaultdict

def save_checkpoint(model, optimizer, epoch, loss, args, filename=None):
    """
    Save model and optimizer state to a checkpoint file.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        args: Program arguments
        filename: Optional specific filename, otherwise uses epoch number
    """
    if filename is None:
        filename = f"{args.train_dir}/checkpoint_epoch{epoch}.pt"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    
    print(f"Checkpoint saved at {filename}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer state from a checkpoint file.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        start_epoch: The epoch to resume from
    """
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 1
        
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
    
    return start_epoch

def data_partition_movie(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_last_indx = {}
    user_valid = {}
    user_test = {}
    Beh = {}
    Beh_w = {}
    # assume user/item index starting from 1
    f = open(fname, 'r')
    for line in f:
        u, i, b = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if b == 'pos':
            last_pos_idx = len(User[u])
            user_last_indx[u] = last_pos_idx
            Beh[(u,i)] = [1,0,0,0]
            Beh_w[(u,i)] = 0.9

        elif b == 'neutral':
            Beh[(u,i)] = [0,1,0,0]
            Beh_w[(u,i)] = 0.1

        elif b == 'neg':
            Beh[(u,i)] = [0,0,1,0]
            Beh_w[(u,i)] = 0.0

        #elif b == 'tip':
        #    Beh[(u,i)] = [0,0,0,1]
        #    Beh_w[(u,i)] = 0.2
        User[u].append(i)

    for user in User:
        Beh[(user,0)] = [0,0,0,0]
        Beh_w[(user,0)] = 0
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            last_item_indx = user_last_indx[user]
            last_item = User[user][last_item_indx]
            items_list = User[user]
            del items_list[last_item_indx]

            user_train[user] = items_list
            #user_train[user] = [value for value in items_list if value != last_item]
            user_valid[user] = []
            user_valid[user].append(last_item)
            user_test[user] = []
            user_test[user].append(last_item)
    return [user_train, user_valid, user_test, Beh, Beh_w, usernum, itemnum]