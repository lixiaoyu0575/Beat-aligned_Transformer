import torch.nn as nn
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn.functional as F
from data_loader.util import *

class Trainer_beat_aligned_data(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, _25classes=False,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, do_test=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = do_test
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # test
        self.num_classes = config['arch']['args']['num_classes']
        self.lead_number = config['data_loader']['args']['lead_number']
        self.split_index = config['data_loader']['args']['split_index']
        self.label_dir = config['data_loader']['args']['label_dir']
        self.resample_Fs = config["data_loader"]['args']["resample_Fs"]
        self.window_size = config["data_loader"]['args']["window_size"]
        self.n_segment = config["evaluater"]["n_segment"]
        self.save_dir = config["data_loader"]["args"]["save_dir"]
        self.seg_with_r = config["data_loader"]["args"]["seg_with_r"]

        split_idx = loadmat(self.split_index)
        train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
        self.test_index = test_index.reshape((test_index.shape[1],))
        # self.train_info = np.array(np.load(
        #     os.path.join(self.save_dir, 'train_info' + str(self.window_size) + '_' + str(self.resample_Fs) + '_' + str(self.seg_with_r) + '.npy')))
        if self.lead_number == 2:
            # two leads
            self.leads_index = [1, 10]
        elif self.lead_number == 3:
            # three leads
            self.leads_index = [0, 1, 7]
        elif self.lead_number == 6:
            # six leads
            self.leads_index = [0, 1, 2, 3, 4, 5]
        elif self.lead_number == 8:
            # eight leads
            self.leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        else:
            self.leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        if self.do_validation:
            keys_val = ['val_' + k for k in self.keys]
            for key in keys_val:
                self.log[key] = []

        if self.do_test:
            keys_test = ['test_' + k for k in self.keys]
            for key in keys_test:
                self.log[key] = []

        self.weights = torch.tensor(self.data_loader.weights)
        self.weights = torch.mean(self.weights, dim=1)

        self.only_scored_classes = self.config['trainer']['only_scored_class']
        if self.only_scored_classes:
            self.indices = [0, 3, 10, 11, 14, 15, 19, 21, 32, 47, 50, 55, 61, 63, 68, 70, 71, 72, 76, 80, 88, 95, 98, 99]

        self.sigmoid = nn.Sigmoid()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start_time = time.time()
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, ([data, info], target, class_weights) in enumerate(self.data_loader):
            batch_start = time.time()
            data, target, class_weights, info = data.to(device=self.device, dtype=torch.float), target.to(self.device, dtype=torch.float), \
                                                class_weights.to(self.device, dtype=torch.float), info.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()
            output = self.model(data, info)

            if self.only_scored_classes:
                loss = self.criterion(output[:, self.indices], target[:, self.indices]) * class_weights[:, self.indices]
            else:
                loss = self.criterion(output, target) * class_weights

            loss = torch.mean(loss)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            output_logit = self.sigmoid(output)
            output_logit = self._to_np(output_logit)
            target = self._to_np(target)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_logit, target))

            if batch_idx % self.log_step == 0:
                batch_end = time.time()
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    batch_end - batch_start))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        end_time = time.time()
        print("training epoch cost {} seconds".format(end_time-start_time))
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if self.config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
                self.lr_scheduler.step(log['val_loss'])
            elif self.config['lr_scheduler']['type'] == 'GradualWarmupScheduler':
                self.lr_scheduler.step(epoch, log['val_loss'])
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, ([data, info], target, class_weights) in enumerate(self.valid_data_loader):
                data, target, class_weights, info = data.to(self.device), target.to(self.device), class_weights.to(self.device), info.to(self.device)
                # target_coarse = target_coarse.to(device)
                output = self.model(data, info)

                if self.only_scored_classes:
                    loss = self.criterion(output[:, self.indices], target[:, self.indices]) * class_weights[:, self.indices]
                else:
                    loss = self.criterion(output, target) * class_weights

                loss = torch.mean(loss)
                # loss = (loss_coarse + loss) / 2

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                output_logit = self.sigmoid(output)
                output_logit = self._to_np(output_logit)
                target = self._to_np(target)

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output_logit, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _to_np(self, tensor):
        if self.device.type == 'cuda':
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, rule_based_ftns=None, _25classes=False,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.rule_based_ftnsm = rule_based_ftns

        if self.do_validation:
            keys_val = ['val_' + k for k in self.keys]
            for key in keys_val:
                self.log[key] = []
        self.weights = torch.tensor(self.data_loader.weights)
        self.weights = torch.mean(self.weights, dim=1)

        self.only_scored_classes = self.config['trainer']['only_scored_class']
        if self.only_scored_classes:
            self.indices = [0, 3, 10, 11, 14, 15, 19, 21, 32, 47, 50, 55, 61, 63, 68, 70, 71, 72, 76, 80, 88, 95, 98, 99]

        self.sigmoid = nn.Sigmoid()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start_time = time.time()
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, class_weights) in enumerate(self.data_loader):
            batch_start = time.time()
            data, target, class_weights = data.to(device=self.device, dtype=torch.float), target.to(self.device, dtype=torch.float), class_weights.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()
            output = self.model(data)

            if self.only_scored_classes:
                loss = self.criterion(output[:, self.indices], target[:, self.indices]) * class_weights[:, self.indices]
            else:
                loss = self.criterion(output, target) * class_weights

            loss = torch.mean(loss)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            output_logit = self.sigmoid(output)
            output_logit = self._to_np(output_logit)
            target = self._to_np(target)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_logit, target))

            if batch_idx % self.log_step == 0:
                batch_end = time.time()
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    batch_end - batch_start))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        end_time = time.time()
        print("training epoch cost {} seconds".format(end_time-start_time))
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if self.config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
                self.lr_scheduler.step(log['val_loss'])
            elif self.config['lr_scheduler']['type'] == 'GradualWarmupScheduler':
                self.lr_scheduler.step(epoch, log['val_loss'])
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, class_weights) in enumerate(self.valid_data_loader):
                data, target, class_weights = data.to(device=self.device, dtype=torch.float), target.to(self.device, dtype=torch.float), class_weights.to(self.device, dtype=torch.float)
                output = self.model(data)

                if self.only_scored_classes:
                    loss = self.criterion(output[:, self.indices], target[:, self.indices]) * class_weights[:, self.indices]
                else:
                    loss = self.criterion(output, target) * class_weights

                loss = torch.mean(loss)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                output_logit = self.sigmoid(output)
                output_logit = self._to_np(output_logit)
                target = self._to_np(target)

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output_logit, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _to_np(self, tensor):
        if self.device.type == 'cuda':
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()

def get_pred(output, alpha=0.5):
    output = F.sigmoid(output)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i, j] >= alpha:
                output[i, j] = 1
            else:
                output[i, j] = 0
    return output
