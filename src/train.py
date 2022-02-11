import torch
from torch.utils.tensorboard import SummaryWriter
from yolo_v3 import YOLOv3
from loss import YOLOLoss
from utils import (
    mean_average_precision,
    # cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    # load_checkpoint,
    # get_loaders,
    # plot_couple_examples
)
import config
from tqdm import tqdm
from torch_dataloader import make_train_val_loaders


def train(train_loader, model, scaled_anchors, loss_fn, opt, scaler, epoch):

    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (image, targets) in enumerate(loop):
        # Port over to CUDA
        image = image.float()
        x = image.to(config.DEVICE)
        y0, y1, y2 = (
            targets[0].to(config.DEVICE),
            targets[1].to(config.DEVICE),
            targets[2].to(config.DEVICE)
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(predictions=out[0], target=y0, anchors=scaled_anchors[0])
                + loss_fn(predictions=out[1], target=y1, anchors=scaled_anchors[1])
                + loss_fn(predictions=out[2], target=y2, anchors=scaled_anchors[2])
            )

            losses.append(loss.item())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Update prog bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

    writer.add_scalar("Train loss", mean_loss, epoch)
    writer.flush()


def main():
    model = YOLOv3().to(config.DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    loss_func = YOLOLoss()

    train_loader, val_loader = make_train_val_loaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)

    ).to(config.DEVICE)

    opt = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(config.NUM_EPOCHS):
        train(train_loader, model, scaled_anchors, loss_fn=loss_func, opt=opt, scaler=scaler, epoch=epoch)

        if epoch % 5 == 0 and epoch > 0:
            save_checkpoint(model, opt)

            pred_boxes, true_boxes = get_evaluation_bboxes(
                val_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD
            )

            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=0
            )

            writer.add_scalar("Validation mAP", mapval, epoch)
            writer.flush()

            model.train()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    writer = SummaryWriter()
    main()
