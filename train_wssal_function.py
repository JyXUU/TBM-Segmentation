import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_wssal(cfg, epoch, sup_loader, unsup_loader, full_model, teacher_model, optimizer,
                ema_decay, lambda_unsup, writer_dict):
    full_model.train()
    teacher_model.eval()

    sup_iter = iter(sup_loader)
    unsup_iter = iter(unsup_loader)
    num_batches = min(len(sup_loader), len(unsup_loader))

    total_loss, total_sup, total_unsup = 0.0, 0.0, 0.0

    for i in tqdm(range(num_batches), desc=f"Epoch {epoch}"):
        try:
            sup_images, sup_labels, sup_edges, _, _ = next(sup_iter)
        except StopIteration:
            sup_iter = iter(sup_loader)
            sup_images, sup_labels, sup_edges, _, _ = next(sup_iter)


        try:
            unsup_images, _, _, _ = next(unsup_iter)
        except StopIteration:
            unsup_iter = iter(unsup_loader)
            unsup_images, _, _, _ = next(unsup_iter)

        sup_images = sup_images.cuda()
        sup_labels = sup_labels.cuda()
        unsup_images = unsup_images.cuda()

        # === Supervised loss ===
        sup_loss = full_model(sup_images, sup_labels)

        # === Unsupervised loss ===
        with torch.no_grad():
            teacher_logits = teacher_model(unsup_images)
            pseudo_probs = torch.sigmoid(teacher_logits)
            pseudo_labels = (pseudo_probs > 0.97).float()

        unsup_loss = full_model(unsup_images, pseudo_labels)

        # === Combined loss ===
        loss = sup_loss + lambda_unsup * unsup_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === EMA update ===
        for t_param, s_param in zip(teacher_model.parameters(), full_model.model.parameters()):
            t_param.data = ema_decay * t_param.data + (1.0 - ema_decay) * s_param.data

        total_loss += loss.item()
        total_sup += sup_loss.item()
        total_unsup += unsup_loss.item()

        writer = writer_dict['writer']
        global_step = writer_dict['train_global_steps']
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        writer.add_scalar('Train/Supervised_Loss', sup_loss.item(), global_step)
        writer.add_scalar('Train/Unsupervised_Loss', unsup_loss.item(), global_step)
        writer_dict['train_global_steps'] += 1

    print(f"[Epoch {epoch}] Total Loss: {total_loss/num_batches:.4f} | Supervised: {total_sup/num_batches:.4f} | Unsupervised: {total_unsup/num_batches:.4f}")
