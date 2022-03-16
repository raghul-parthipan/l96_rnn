import numpy as np

def create_batch(input_list, batch_s=32):
    
    batch_list = []
    shape_label = input_list[0].shape[0]
    batch_idx_la = np.random.choice(list(range(shape_label)), batch_s)
    for i in input_list: 
        batch_item = (i[batch_idx_la,])
        batch_list.append(batch_item)
    
    del batch_idx_la
            
    return batch_list


def progress_bar(iteration, total, size=30):
    """Progress bar for training"""
    running = iteration < total
    c = ">" if running else "="
    p = (size - 1) * iteration // total
    fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
    params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
    return fmt.format(*params)

def print_status_bar(iteration, total, loss, metrics=None, size=30):
    """Status bar for training"""
    metrics = " - ".join(["Loss for batch: {:.4f}".format(loss)])
    end = "" if iteration < total else "\n"
    print("\r{} - {}".format(progress_bar(iteration, total), metrics), end=end)
    
def print_status_bar_epoch(iteration, total, training_loss_for_epoch,val_loss, metrics=None, size=30):
    """Status bar for training (end of epoch)"""
    metrics = " - ".join(
        ["trainLoss: {:.4f}  Val_loss: {:.4f} ".format(
            training_loss_for_epoch,val_loss)]
    )
    end = "" if iteration < total else "\n"
    print("\r{} - {}".format(progress_bar(iteration, total), metrics), end=end)
    
    
def list_average(list_of_loss):
    return sum(list_of_loss)/len(list_of_loss)
