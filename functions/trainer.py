import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from functions.evaluator import Evaluator
from models.losses.p2m import P2MLoss
from models.p2m import P2MModel
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid


class Trainer(CheckpointRunner):

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        # create ellipsoid
        self.ellipsoid = Ellipsoid()

        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = P2MModel(self.options.model.feat_dim,
                                  self.options.model.hidden,
                                  self.options.model.coord_dim,
                                  self.ellipsoid)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()

        # Setup a joint optimizer for the 2 models
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.options.optim.lr,
            betas=(self.options.optim.adam_beta1, 0.999),
            weight_decay=self.options.optim.wd)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
        )

        # Create loss functions
        self.criterion = P2MLoss(self.ellipsoid).cuda()

        # Create AverageMeters for losses
        self.losses = AverageMeter()

        # Renderer for visualization
        # self.renderer = Renderer(faces=self.smpl.faces.cpu().detach())

        # Evaluators
        self.evaluators = []

    def models_dict(self):
        return {'model': self.model}

    def optimizers_dict(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}

    def train_step(self, input_batch):
        self.model.train()

        # Grab data from the batch
        images, gt_points, gt_normals = input_batch["images"], input_batch["points"], input_batch["normals"]

        # predict with model
        pred_pts_list, pred_feats_list, pred_img = self.model(images)

        # compute loss
        loss, loss_summary = self.criterion(pred_pts_list, pred_feats_list, gt_points)

        self.losses.update(loss.detach().cpu().item())

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments to be used for visualization in a list
        out_args = [pred_pts_list, pred_feats_list, pred_img]
        out_args = [arg.detach() for arg in out_args]
        loss_summary = {k: v.detach() for k, v in loss_summary.items()}
        out_args.append(loss_summary)
        return out_args

    def train(self):
        train_data_loader = DataLoader(self.dataset,
                                       batch_size=self.options.train.batch_size * self.options.num_gpus,
                                       num_workers=self.options.num_workers,
                                       pin_memory=self.options.pin_memory,
                                       shuffle=self.options.train.shuffle)

        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1

            # Reset loss
            self.losses.reset()

            # Iterate over all batches in an epoch
            for step, batch in enumerate(train_data_loader):
                # Run training step
                out = self.train_step(batch)

                self.step_count += 1

                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch, *out)

                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint()

                # Run validation every test_steps steps
                if self.step_count % self.options.train.test_steps == 0:
                    self.test()

            # save checkpoint after each epoch
            self.dump_checkpoint()

            # lr scheduler step
            self.lr_scheduler.step()

    def train_summaries(self, input_batch,
                        pred_vertices, pred_vertices_smpl, pred_camera,
                        pred_keypoints_2d, pred_keypoints_2d_smpl, loss_summary):
        # Do visualization for the first 4 images of the batch
        rend_imgs, rend_imgs_smpl = visualize_batch_recon_and_keypoints(4, self.renderer, input_batch["img_orig"],
                                                                        pred_vertices, pred_vertices_smpl,
                                                                        pred_camera, pred_keypoints_2d,
                                                                        pred_keypoints_2d_smpl,
                                                                        input_batch['keypoints'])

        # Save results in Tensorboard
        self.summary_writer.add_image('imgs', rend_imgs, self.step_count)
        self.summary_writer.add_image('imgs_smpl', rend_imgs_smpl, self.step_count)
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v, self.step_count)

        # Save results to log
        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.9f (%.9f)" % (
            self.epoch_count, self.step_count,
            self.options.train.num_epochs * len(self.dataset) // (
                        self.options.train.batch_size * self.options.num_gpus),
            timedelta(seconds=time.time() - self.time_start), self.losses.val, self.losses.avg))

    def test(self):
        for evaluator in self.evaluators:
            evaluator.evaluate()
