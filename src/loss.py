import torch
import torch.nn as nn
import config

from utils import int_over_union


class YOLOLoss(nn.Module):

    def __init__(self):
        super().__init__()
        # For box preds
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        # self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        # loss weights range 1 -> 10
        # self.lambda_noobj = 10
        # self.lambda_obj = 1
        # self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # No object loss
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No obj loss
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj])
        )
        if no_object_loss == torch.tensor(torch.nan):
            no_object_loss = config.EPS

        # Custom check to see if targets actually have any labels
        obj_presence = len(target[obj])
        if obj_presence != 0:
            # Object loss
            # More nuanced to use iou as a factor during loss calculation
            anchors = anchors.reshape(1, 3, 1, 1, 2)
            box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
            ious = int_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
            object_loss = self.mse(
                self.sigmoid(predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj])
            )

            # Box coordinates loss
            predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
            target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
            box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

            return no_object_loss + object_loss + box_loss

        return no_object_loss
