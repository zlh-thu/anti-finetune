import os
import torch

import inversefed
from utils import save_img
from .utils import get_gradient


def rec_img(model, w_client, ground_truth_item, args, global_ep, device, grad=None):
    image_path = './save/' + str(args.exp_id) + '/global_ep' + str(global_ep) + '/' + str(
        int(ground_truth_item['index'].cpu().numpy())) + '/'
    os.makedirs(image_path, exist_ok=True)

    print('gt', int(ground_truth_item['target'].cpu().numpy()))

    img_shape = ground_truth_item['img'].shape
    w_global = model.state_dict(keep_vars=True)
    if grad is not None:
        input_gradient = grad
    else:
        input_gradient = get_gradient(w_global, w_client, args.lr)

    input_gradient = [grad.detach() for grad in input_gradient]
    full_norm = torch.stack([g.norm() for g in input_gradient]).mean()

    dm = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_mean'), dtype=torch.float, device=device)[:, None, None]
    ds = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_std'), dtype=torch.float, device=device)[:, None, None]

    print(f'Full gradient norm is {full_norm:e}.')

    if ground_truth_item['img'].dim() == 3:
        ground_truth_item['img'] = torch.unsqueeze(ground_truth_item['img'], dim=0)

    gt_denormalized = torch.clamp(ground_truth_item['img'].to(device) * ds + dm, 0, 1)
    gt_filename = ('gt.png')
    save_img(gt_denormalized[0].cpu(), os.path.join(image_path, gt_filename))


    if args.dataset == 'cifar10':
        config = dict(signed=True,
                          boxed=True,
                          cost_fn='sim',
                          indices='def',
                          weights='equal',
                          lr=0.1,
                          optim='adam',
                          restarts=1,
                          # max_iterations=24_000,
                          max_iterations=args.max_iterations,
                          total_variation=1e-6,
                          init='randn',
                          filter='none',
                          lr_decay=True,
                          scoring_choice='loss')
    elif args.dataset == 'imagenet':
        config = dict(signed=True,
                      boxed=True,
                      cost_fn='sim',
                      indices='top10',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=args.max_iterations,
                      total_variation=1e-6,
                      init='randn',
                      filter='median',
                      lr_decay=True,
                      scoring_choice='loss')
    else:
        NotImplementedError


    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
    output, _ = rec_machine.reconstruct(input_gradient, labels=None, image_path=image_path, dm=dm, ds=ds, img_shape=img_shape, dryrun=args.dryrun)


    # Save the resulting image
    output_denormalized = torch.clamp(output * ds + dm, 0, 1)
    rec_filename = ('rec.png')
    save_img(output_denormalized[0].cpu(), os.path.join(image_path, rec_filename))

    return