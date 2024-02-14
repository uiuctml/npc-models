#!/usr/bin/env python3

import argument
import dataset
import header
import logger
import model
import os
import torch
import torchvision
import tqdm
import utility

def attack_pgd(model_decomposed, x, y, attribute, num_steps = 20, step_size=1e-2, step_norm="inf", eps=1/5, eps_norm="inf", clamp=(0, 1), y_target=None):
    x_adv = x.clone().detach().requires_grad_(True)
    targeted = y_target is not None
    num_channels = x.shape[1]
    batch_size = x.shape[0]

    loss_fn = torch.nn.CrossEntropyLoss()
    model_decomposed = model_decomposed.train()

    for _ in range(num_steps):
        (prediction, _) = model_decomposed(x_adv)

        loss = loss_fn(prediction[attribute], y_target if targeted else y[:, attribute])
        loss.backward()

        with torch.no_grad():
            if step_norm == 'inf':
                gradients = x_adv.grad.sign() * step_size
            else:
                gradients = x_adv.grad * step_size / x_adv.grad.view(batch_size, -1).norm(step_norm, dim=-1).view(-1, num_channels, 1, 1)

            if targeted:
                x_adv -= gradients
            else:
                x_adv += gradients

        if eps_norm == 'inf':
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        else:
            delta = x_adv - x
            mask = delta.view(batch_size, -1).norm(eps_norm, dim=1) <= eps
            scaling_factor = delta.view(batch_size, -1).norm(eps_norm, dim=1)
            scaling_factor[mask] = eps
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)
            x_adv = x + delta

        x_adv = x_adv.clamp(*clamp)
        x_adv = x_adv.detach().requires_grad_(True)

    x_adv = list(x_adv.detach())

    return x_adv

def main():
    argument.processArgumentsAttack()

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if not os.path.exists(header.dir_dataset_test_adversarial):
        os.makedirs(header.dir_dataset_test_adversarial, exist_ok = True)

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    config_dataset = dataset_test.config

    if header.attack_targeted_attribute >= len(config_dataset["attributes"]):
        logger.log_fatal("Invalid targeted attribute. Quit.")
        exit(-1)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    attribute_name = config_dataset["attributes"][header.attack_targeted_attribute]["name"]
    dir_attack = header.config_decomposed["model"] + "_" + "pgd" + "_" + attribute_name
    dir_dataset_test_adversarial_attack = os.path.join(header.dir_dataset_test_adversarial, dir_attack)

    if not os.path.exists(dir_dataset_test_adversarial_attack):
        os.makedirs(dir_dataset_test_adversarial_attack, exist_ok = True)
    else:
        logger.log_info("Directory \"" + dir_dataset_test_adversarial_attack + "\" exists. Quit")
        exit(0)

    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)

    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)

    progress_bar = tqdm.tqdm(total = len(data_loader_test), position = 0, leave = False)
    progress_bar.set_description_str("[INFO]: Attacking \"" + attribute_name + "\" attribute")

    for (batch_index, (input, labels, _, input_file_paths)) in enumerate(data_loader_test):
        input = input.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        input_attacked_list = attack_pgd(model_decomposed, input, labels, header.attack_targeted_attribute)

        progress_bar_save = tqdm.tqdm(total = len(input_attacked_list), position = 1, leave = False)
        progress_bar_save.set_description_str("[INFO]: Saving attacks")

        for (input_index, (input_attacked, input_file_path)) in enumerate(zip(input_attacked_list, input_file_paths)):
            file_name_input = os.path.basename(input_file_path)
            dir_class = os.path.basename(os.path.dirname(input_file_path))
            dir_dataset_test_adversarial_attack_class = os.path.join(dir_dataset_test_adversarial_attack, dir_class)
            input_attacked_file_path = os.path.join(dir_dataset_test_adversarial_attack_class, file_name_input)

            if not os.path.exists(dir_dataset_test_adversarial_attack_class):
                os.makedirs(dir_dataset_test_adversarial_attack_class, exist_ok = True)

            torchvision.utils.save_image(input_attacked, input_attacked_file_path)

            progress_bar_save.n = input_index + 1
            progress_bar_save.refresh()

        progress_bar_save.close()

        progress_bar.n = batch_index + 1
        progress_bar.refresh()

    progress_bar.close()

    return

if __name__ == "__main__":
    main()
