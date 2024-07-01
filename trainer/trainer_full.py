import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer
from plot.plot import show_tensor_images, plot_grayscale_image
from utils.util import visualize_flow_torch, create_color_image

class DefaultTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None):
        super(DefaultTrainer, self).__init__(config, model, loss, metrics, optimizer, lr_scheduler, resume,
                                             train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, pred, gt):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(pred, gt)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        # set the model to train mode
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        # start training
        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            # get data and send them to GPU
            E = sample['E'].to(self.device)
            B = sample['B'].to(self.device)
            Bi = sample['Bi'].to(self.device)

            F_gt = sample['F'].to(self.device)
            Bi_clean_gt = sample['Bi_clean'].to(self.device)
            S_gt = sample['S'].to(self.device)

            # get network output
            Bi_clean_pred, log_diff, S_pred, code = self.model(B, Bi, E)
            S_pred = torch.clamp(S_pred, min=0, max=1)

            # visualization
            with torch.no_grad():
                if batch_idx % 100 == 0:
                    # save images to tensorboardX
                    self.writer.add_image('Bi_clean_pred', make_grid(Bi_clean_pred))
                    self.writer.add_image('log_diff', make_grid(log_diff))
                    self.writer.add_image('S_pred', make_grid(S_pred))
                    self.writer.add_image('S_gt', make_grid(S_gt))
                    self.writer.add_image('Blurred', make_grid(B))

            # train model
            self.optimizer.zero_grad()
            model_loss = self.loss(Bi_clean_pred, Bi_clean_gt, S_pred, S_gt, code)
            model_loss.backward()
            self.optimizer.step()

            # calculate total loss/metrics and add scalar to tensorboard
            self.writer.add_scalar('loss', model_loss.item())
            total_loss += model_loss.item()
            total_metrics += self._eval_metrics(S_pred, S_gt)

            # show current training step info
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss.item()  # it's a tensor, so we call .item() method
                    )
                )
                self.logger.info('+' * 30)  # 添加分割线

        # get batch average loss/metrics as log and do validation
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}
            val_loss = val_log['val_loss']
            self.lr_scheduler.step(val_loss)  # 传递验证损失
        else:
            self.lr_scheduler.step(total_loss / len(self.data_loader))  # 没有验证集时，使用训练损失

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        # set the model to validation mode
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        # start validating
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

                # get data and send them to GPU
                E = sample['E'].to(self.device)
                B = sample['B'].to(self.device)
                Bi = sample['Bi'].to(self.device)

                F_gt = sample['F'].to(self.device)
                Bi_clean_gt = sample['Bi_clean'].to(self.device)
                S_gt = sample['S'].to(self.device)

                # infer and calculate the loss
                Bi_clean_pred, log_diff, S_pred, code = self.model(B, Bi, E)
                S_pred = torch.clamp(S_pred, min=0, max=1)

                with torch.no_grad():
                    if batch_idx % 100 == 0:
                        # save images to tensorboardX
                        self.writer.add_image('Bi_clean_pred', make_grid(Bi_clean_pred))
                        
                        color_images = create_color_image(log_diff)
                        grid = make_grid(color_images)
                        self.writer.add_image('log_diff', grid)
                        
                        self.writer.add_image('S_pred', make_grid(S_pred))
                        self.writer.add_image('S_gt', make_grid(S_gt))
                        self.writer.add_image('Blurred', make_grid(B))

                loss = self.loss(Bi_clean_pred, Bi_clean_gt, S_pred, S_gt, code)

                # calculate total loss/metrics and add scalar to tensorboardX
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(S_pred, S_gt)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }