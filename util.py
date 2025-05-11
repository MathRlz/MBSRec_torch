import torch
from collections import defaultdict
import csv

def save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, args, best_ndcg, epochs_no_improve):
    """
    Save model and optimizer state to a checkpoint file.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        scheduler: The scheduler to save
        epoch: Current epoch number
        avg_loss: Current average loss value
        args: Program arguments
        best_ndcg: Current best NDCG value
        epochs_no_improve: Number of epochs with no improvement
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'avg_loss': avg_loss,
        'best_ndcg': best_ndcg,
        'epochs_no_improve': epochs_no_improve,
    }
    torch.save(checkpoint, f"{args.train_dir}/checkpoint.pt")
    
    print(f"Checkpoint saved at {args.train_dir}/checkpoint.pt")

def load_checkpoint(model, optimizer, scheduler, resume_path):
    """
    Load model and optimizer state from a checkpoint file.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into
        resume_path: Path to the checkpoint file
        
    Returns:
        start_epoch: The epoch to resume from
    """
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch'] + 1
    best_ndcg = checkpoint.get('best_ndcg', -float('inf'))
    epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
    return epoch, best_ndcg, epochs_no_improve

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

def data_partition_tmall(fname):
    usernum = 0
    itemnum = 0
    interactions = 0
    User = defaultdict(list)
    user_train = {}
    user_last_indx = {}
    user_valid = {}
    user_test = {}
    Beh = {}
    Beh_w = {}
    # assume user/item index starting from 1
    f = open(fname, 'r')
    next(f)
    for line in f:
        interactions+=1
        u, i, b = line.rstrip().split(' ')
        #print( 'data type of user ....', u, '  ',i,'    ', b)
        #print(abc)
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        if b == 'buy':
            last_pos_idx = len(User[u])
            user_last_indx[u] = last_pos_idx
            Beh[(u,i)] = [1,0,0,0]
            Beh_w[(u,i)] = 0.7

        elif b == 'cart':
            Beh[(u,i)] = [0,0,1,0]
            Beh_w[(u,i)] = 0.1

        elif b == 'fav':
            Beh[(u,i)] = [0,0,0,1]
            Beh_w[(u,i)] = 0.1
            
        elif b == 'pv':
            Beh[(u,i)] = [0,1,0,0]
            Beh_w[(u,i)] = 0.1
            
        User[u].append(i)
    print('Total Number of interactions is .....', interactions)
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

            #user_train[user] = items_list
            #user_train[user] = [value for value in items_list if value != last_item]
            truncated_item_list = items_list[:last_item_indx]       
            user_train[user] = [value for value in truncated_item_list if value != last_item]
            user_valid[user] = []
            user_valid[user].append(last_item)
            user_test[user] = []
            user_test[user].append(last_item)
    return [user_train, user_valid, user_test, Beh, Beh_w, usernum, itemnum]

def data_partition_yelp(fname):
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
            Beh_w[(u,i)] = 0.3   # was 0.5
            
        elif b == 'tip':
            Beh[(u,i)] = [0,0,0,1]
            Beh_w[(u,i)] = 0.3    #was 0.2
            
        elif b == 'neutral':
            Beh[(u,i)] = [0,1,0,0]
            Beh_w[(u,i)] = 0.2    # was 0.2

        elif b == 'neg':
            Beh[(u,i)] = [0,0,1,0]
            Beh_w[(u,i)] = 0.2     #was 0.1

        
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

def data_partition_anime(fname):
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
    anime_reader = csv.reader(f, delimiter=';')
    next(anime_reader) # Skip the header row
    for row in anime_reader:
        u, i, b = row 
        u = int(u)
        i = int(i)
        b = int(b)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)

        Beh[(u,i)] = [0]*11
        Beh[(u,i)][b] = 1
        Beh_w[(u,i)] = (float)(b)/10.0

        # Rating over 7 is considered positive
        if b > 7:
            last_pos_idx = len(User[u])
            user_last_indx[u] = last_pos_idx

        User[u].append(i)

    for user in User:
        Beh[(user,0)] = [0] * 11
        Beh_w[(user,0)] = 0
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            # Check if user has any positive interactions
            if user in user_last_indx:
                last_item_indx = user_last_indx[user]
                last_item = User[user][last_item_indx]
                items_list = User[user]
                del items_list[last_item_indx]

                user_train[user] = items_list
                user_valid[user] = []
                user_valid[user].append(last_item)
                user_test[user] = []
                user_test[user].append(last_item)
            else:
                # Handle users with no positive interactions
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
    return [user_train, user_valid, user_test, Beh, Beh_w, usernum, itemnum]
