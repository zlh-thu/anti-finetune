import copy
from .rec_img import rec_img
from .anti_finetune import anti_finetune_dlg

def gradient_inv(args, attack_item, client_idx, attack_ids, attack_repeat, global_model, epoch, w, device, grad, train_dataset_list, target_label=None):
    if args.attack == 'anti_finetune' and args.leakage_attack:
        assert args.mood == 'avg_loss'
        anti_finetune_dlg(model=copy.deepcopy(global_model), w_client=copy.deepcopy(w),
           ground_truth_item=attack_item, args=args, global_ep=epoch, device=device, target_label=target_label)
    else:
        raise NotImplementedError

    return attack_ids, attack_repeat