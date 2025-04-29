import os
import torch

import inversefed
from utils import save_img
from .utils import get_gradient

def anti_finetune_dlg(model, w_client, ground_truth_item, args, global_ep, device, target_label):
    print(' reconstructing imgs with dlg...')
    dm = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_mean'), dtype=torch.float, device=device)[
         :, None, None]
    ds = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_std'), dtype=torch.float, device=device)[:,
         None, None]

    image_path = './save/' + str(args.exp_id) + '/global_ep' + str(global_ep) + '/'
    os.makedirs(image_path, exist_ok=True)

    for i in range(len(ground_truth_item['index'])):
        gt_denormalized = torch.clamp(ground_truth_item['img'][i].to(device) * ds + dm, 0, 1)
        if i == ground_truth_item['max_id_in_batch']:
            gt_filename = ('gt_{}_label_{}_max_loss.png'.format(str(i), str(int(ground_truth_item['target'][i].cpu().numpy()))))
        else:
            gt_filename = ('gt_{}_label_{}.png'.format(str(i), str(int(ground_truth_item['target'][i].cpu().numpy()))))
        save_img(gt_denormalized.cpu(), os.path.join(image_path, gt_filename))


    if len(ground_truth_item['img'].shape) == 4:
        # batch_size , C, H, W
        img_shape = ground_truth_item['img'][0].shape
        num_images = args.batch_size

    elif len(ground_truth_item['img'].shape) == 3:
        img_shape = ground_truth_item['img'].shape
        num_images = 1
        assert args.batch_size == 1


    w_global = model.state_dict(keep_vars=True)
    input_gradient = get_gradient(w_global, w_client, args.lr)

    model.zero_grad()
    model.eval()
    input_gradient = [grad.detach() for grad in input_gradient]
    full_norm = torch.stack([g.norm() for g in input_gradient]).mean()

    if args.use_target_label:
        assert target_label is not None
        labels = torch.zeros([args.batch_size]).to(device) + target_label
        labels = labels.long()
    else:
        labels = None

    print(f'Full gradient norm is {full_norm:e}.')

    config = dict(signed=args.signed,
                      boxed=args.boxed,
                      cost_fn=args.cost_fn,
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim=args.rec_optimizer,
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')

    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
    output, _ = rec_machine.reconstruct(input_gradient, labels=labels, image_path=image_path, dm=dm, ds=ds, img_shape=img_shape, dryrun=args.dryrun)

    if ground_truth_item['img'].dim() == 3:
        ground_truth_item['img'] = torch.unsqueeze(ground_truth_item['img'], dim=0)

    # Save the resulting image
    output_denormalized = torch.clamp(output * ds + dm, 0, 1)
    for i in range(num_images):
        rec_filename = ('rec_{}.png'.format(str(i)))
        save_img(output_denormalized[i].cpu(), os.path.join(image_path, rec_filename))
    print('done')

    return