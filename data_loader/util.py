import os
import json
import numpy as np
from numpy import inf
from scipy import signal
from scipy.io import loadmat, savemat
import torch
from torch.utils.data import Dataset
import logging
import neurokit2 as nk
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import matplotlib.pyplot as plt
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# Find Challenge files.
def load_label_files(label_directory):
    label_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f) # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            # root, ext = os.path.splitext(f)
            label_files.append(F)
    if label_files:
        return label_files
    else:
        raise IOError('No label or output files found.')

# Load labels from header/label files.
def load_labels(label_files, normal_class, equivalent_classes_collection):
    # The labels_onehot should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        with open(label_files[i], 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    dxs = set(arr.strip() for arr in l.split(': ')[1].split(','))
                    tmp_labels.append(dxs)

    # Identify classes.
    classes = set.union(*map(set, tmp_labels))
    if normal_class not in classes:
        classes.add(normal_class)
        print('- The normal class {} is not one of the label classes, so it has been automatically added, but please check that you chose the correct normal class.'.format(normal_class))
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for labels.
    labels_onehot = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for dx in dxs:
            j = classes.index(dx)
            labels_onehot[i, j] = 1

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The label for the representative class is positive if any of the labels_onehot in the set is positive.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes)>1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            labels_onehot[:, representative_index] = np.any(labels_onehot[:, equivalent_indices], axis=1)
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    labels_onehot = np.delete(labels_onehot, remove_indices, axis=1)

    # If the labels_onehot are negative for all classes, then change the label for the normal class to positive.
    normal_index = classes.index(normal_class)
    for i in range(num_recordings):
        num_positive_classes = np.sum(labels_onehot[i, :])
        if num_positive_classes==0:
            labels_onehot[i, normal_index] = 1

    labels = list()
    for i in range(num_recordings):
        class_list = []
        for j in range(len(classes)):
            if labels_onehot[i][j] == True:
                class_list.append(classes[j])
        class_set = set()
        class_set.update(class_list)
        labels.append(class_set)
    return classes, labels_onehot, labels,

# Load challenge data.
def load_challenge_data(label_file, data_dir):
    file = os.path.basename(label_file)
    name, ext = os.path.splitext(file)
    with open(label_file, 'r') as f:
        header = f.readlines()
    mat_file = file.replace('.hea', '.mat')
    x = loadmat(os.path.join(data_dir, mat_file))
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header, name

# Load weights.
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

# Load_table
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    print(os.getcwd())
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

# Divide ADC_gain and resample
def resample(data, header_data, resample_Fs = 300):
    # get information from header_data
    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs = int(tmp_hea[2])
    sample_len = int(tmp_hea[3])
    gain_lead = np.zeros(num_leads)

    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # divide adc_gain
    for ii in range(num_leads):
        data[ii] /= gain_lead[ii]

    resample_len = int(sample_len * (resample_Fs / sample_Fs))
    resample_data = signal.resample(data, resample_len, axis=1, window=None)

    return resample_data

def ecg_filling(ecg, sampling_rate, length):
    # try:
    ecg_single_lead = ecg[1]
        # processed_ecg = nk.ecg_process(ecg_II, sampling_rate)
    cleaned = nk.ecg_clean(ecg_single_lead, sampling_rate=sampling_rate)
    processed_ecg = nk.ecg_findpeaks(cleaned, sampling_rate=sampling_rate, method='neurokit')
    # except:
    #     ecg_single_lead = ecg[1]
    #     # processed_ecg = nk.ecg_process(ecg_II, sampling_rate)
    #     cleaned = nk.ecg_clean(ecg_single_lead, sampling_rate=sampling_rate)
    #     processed_ecg = nk.ecg_findpeaks(cleaned, sampling_rate=sampling_rate)
    rpeaks = processed_ecg['ECG_R_Peaks']

    beats = nk.ecg_segment(cleaned, rpeaks, sampling_rate, True)
    ecg_filled = np.zeros((ecg.shape[0], length))
    sta = rpeaks[-1]
    ecg_filled[:, :sta] = ecg[:, :sta]
    seg = ecg[:, rpeaks[0]:rpeaks[-1]]
    len = seg.shape[1]
    while True:
        if (sta + len) >= length:
            ecg_filled[:, sta: length] = seg[:, : length - sta]
            break
        else:
            ecg_filled[:, sta: sta + len] = seg[:, :]
            sta = sta + len
    return ecg_filled

def ecg_filling2(ecg, length):
    len = ecg.shape[1]
    ecg_filled = np.zeros((ecg.shape[0], length))
    ecg_filled[:, :len] = ecg
    sta = len
    while length - sta > len:
        ecg_filled[:, sta : sta + len] = ecg
        sta += len
    ecg_filled[:, sta:length] = ecg[:, :length-sta]

    return ecg_filled

import scipy
# def slide_and_cut_beat_aligned(data, n_segment=1, window_size=3000, sampling_rate=300):
#
#     print("********************************************************************")
#
#     print("data_shape:", data.shape)
#
#     channel_num, length = data.shape
#     ecg_single_lead = data[1]
#     # processed_ecg = nk.ecg_process(ecg_II, sampling_rate)
#     ecg2save = np.zeros((10, channel_num, 400))
#     info2save = np.zeros((10,))
#     # ecg2save = []
#     # info2save = []
#     cleaned = nk.ecg_clean(ecg_single_lead, sampling_rate=sampling_rate)
#     try:
#         processed_ecg = nk.ecg_findpeaks(cleaned, sampling_rate=sampling_rate, method='neurokit')
#         rpeaks = processed_ecg['ECG_R_Peaks']
#         # ecg_segments = nk.ecg_segment(cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)
#     except:
#         return None, None
#     for i in range(len(rpeaks) - 1):
#         # key = str(i+1)
#         # seg_values = ecg_segments[key].values
#         # indexes = seg_values[:, 1]
#         # start_index = indexes[0] if indexes[0] > 0 else 0
#         # end_index = indexes[-1]
#         start_index = rpeaks[i]
#         end_index = rpeaks[i+1]
#
#         print("start_index", start_index)
#         print("end_index", end_index)
#
#         beat = data[:, start_index:end_index]
#         resample_ratio = beat.shape[1] / 400
#         resampled_beat = scipy.signal.resample(beat, 400, axis=1) # Resample x to num samples using Fourier method along the given axis.
#         ecg2save[i] = resampled_beat
#         info2save[i] = resample_ratio
#         # ecg2save.append(resampled_beat)
#         # info2save.append(resample_ratio)
#         if i >= 9:
#             break
#     # ecg2save = np.array(ecg2save)
#     # info2save = np.array(info2save)
#     return ecg2save, info2save

def slide_and_cut_beat_aligned(data, n_segment=1, window_size=3000, sampling_rate=300, seg_with_r=False, beat_length=400):

    # print("********************************************************************")
    #
    # print("data_shape:", data.shape)

    channel_num, length = data.shape
    ecg_single_lead = data[1]
    # processed_ecg = nk.ecg_process(ecg_II, sampling_rate)

    # ecg2save = []
    # info2save = []
    cleaned = nk.ecg_clean(ecg_single_lead, sampling_rate=sampling_rate)

    try:

        if not seg_with_r:
            ecg_segments = nk.ecg_segment(cleaned, sampling_rate=sampling_rate)
        else:
            processed_ecg = nk.ecg_findpeaks(cleaned, sampling_rate=sampling_rate, method='neurokit')
            rpeaks = processed_ecg['ECG_R_Peaks']
    except:
        return None, None

    ecgs2save = []
    infos2save = []

    offset = int((len(rpeaks) - 10 * n_segment) / (n_segment + 1))
    if offset >= 0:
        start = 0 + offset
    else:
        start = 0

    for j in range(n_segment):

        # print("segment:", j)

        ecg2save = np.zeros((10, channel_num, beat_length))
        info2save = np.zeros((10,))

        start_ind = start + j * (10 + offset)

        for i in range(start_ind, len(rpeaks) - 1):
            if not seg_with_r:
                key = str(i+1)
                seg_values = ecg_segments[key].values
                indexes = seg_values[:, 1]
                start_index = indexes[0] if indexes[0] > 0 else 0
                end_index = indexes[-1]
            else:
                start_index = rpeaks[i]
                end_index = rpeaks[i+1]

            # print("start_index", start_index)
            # print("end_index", end_index)

            beat = data[:, start_index:end_index]
            resample_ratio = beat.shape[1] / beat_length
            resampled_beat = scipy.signal.resample(beat, beat_length, axis=1) # Resample x to num samples using Fourier method along the given axis.
            ecg2save[i-start_ind] = resampled_beat
            info2save[i-start_ind] = resample_ratio
            # ecg2save.append(resampled_beat)
            # info2save.append(resample_ratio)
            if i-start_ind >= 9:
                break
        ecgs2save.append(ecg2save)
        infos2save.append(info2save)

    ecgs2save = np.array(ecgs2save)
    infos2save = np.array(infos2save)
    return ecgs2save, infos2save

# def slide_and_cut(data, n_segment=1, window_size=3000, sampling_rate=300):
#     channel_num, length = data.shape
#     ecg_single_lead = data[1]
#     # processed_ecg = nk.ecg_process(ecg_II, sampling_rate)
#     ecg2save = np.zeros((10, channel_num, 400))
#     info2save = np.zeros((10,))
#     # ecg2save = []
#     # info2save = []
#     cleaned = nk.ecg_clean(ecg_single_lead, sampling_rate=sampling_rate)
#     try:
#         processed_ecg = nk.ecg_findpeaks(cleaned, sampling_rate=sampling_rate, method='neurokit')
#         rpeaks = processed_ecg['ECG_R_Peaks']
#         ecg_segments = nk.ecg_segment(cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)
#     except:
#         return None, None
#     for i in range(len(ecg_segments) - 1):
#         key = str(i+1)
#         seg_values = ecg_segments[key].values
#         indexes = seg_values[:, 1]
#         start_index = indexes[0] if indexes[0] > 0 else 0
#         end_index = indexes[-1]
#         beat = data[:, start_index:end_index]
#         resample_ratio = beat.shape[1] / 400
#         resampled_beat = scipy.signal.resample(beat, 400, axis=1) # Resample x to num samples using Fourier method along the given axis.
#         ecg2save[i] = resampled_beat
#         info2save[i] = resample_ratio
#         # ecg2save.append(resampled_beat)
#         # info2save.append(resample_ratio)
#         if i >= 9:
#             break
#     # ecg2save = np.array(ecg2save)
#     # info2save = np.array(info2save)
#     return ecg2save, info2save


# split into training and validation

def slide_and_cut(data, n_segment=1, window_size=3000, sampling_rate=300):
    length = data.shape[1]
    print("length:", length)
    if length < window_size:
        segments = []
        ecg_filled = ecg_filling2(data, window_size)
        # try:
        #     ecg_filled = ecg_filling(data, sampling_rate, window_size)
        # except:
        #     ecg_filled = ecg_filling2(data, window_size)
        segments.append(ecg_filled)
        segments = np.array(segments)
    else:
        offset = (length - window_size * n_segment) / (n_segment + 1)
        if offset >= 0:
            start = 0 + offset
        else:
            offset = (length - window_size * n_segment) / (n_segment - 1)
            start = 0
        segments = []
        for j in range(n_segment):
            ind = int(start + j * (window_size + offset))
            segment = data[:, ind:ind + window_size]
            segments.append(segment)
        segments = np.array(segments)

    return segments

def stratification(label_dir):
    print('Stratification...')

    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    # Find the label files.
    label_files = load_label_files(label_dir)

    # Load the labels and classes.
    label_classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

    temp = [[] for _ in range(len(labels_onehot))]
    indexes, values = np.where(np.array(labels_onehot).astype(int) == 1)
    for k, v in zip(indexes, values):
       temp[k].append(v)
    labels_int = temp

    X = np.zeros(len(labels_onehot))
    y = labels_onehot

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    for train_index, val_index in msss.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        print('Saving split index...')
        datasets_distribution(labels_int, [train_index, val_index])
        savemat('model_training/split.mat', {'train_index': train_index, 'val_index': val_index})

    print('Stratification done.')

def datasets_distribution(labels_int, indexs):
   num_of_bins = 108
   fig, axs = plt.subplots(len(indexs), 1, sharey=True, figsize=(50, 50))
   for i in range(len(indexs)):
      subdataset = list()
      for j in indexs[i]:
         for k in labels_int[j]:
            subdataset.append(k)
      subdataset = np.array(subdataset)
      axs[i].hist(subdataset, bins=num_of_bins)
   plt.show()
import time
# Training
def make_dirs(base_dir):

    checkpoint_dir = base_dir + '/checkpoints'
    log_dir = base_dir + '/log_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    tb_dir = base_dir + '/tb_log'
    result_dir = base_dir + '/results'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    return result_dir, log_dir, checkpoint_dir, tb_dir

def init_obj(hype_space, name, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    """
    module_name = hype_space[name]['type']
    module_args = dict(hype_space[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def to_np(tensor, device):
    if device.type == 'cuda':
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

def get_mnt_mode(trainer):
    monitor = trainer.get('monitor', 'off')
    if monitor == 'off':
        mnt_mode = 'off'
        mnt_best = 0
        early_stop = 0
        mnt_metric_name = None
    else:
        mnt_mode, mnt_metric_name = monitor.split()
        assert mnt_mode in ['min', 'max']
        mnt_best = inf if mnt_mode == 'min' else -inf
        early_stop = trainer.get('early_stop', inf)

    return mnt_metric_name, mnt_mode, mnt_best, early_stop

def save_checkpoint(model, epoch, mnt_best, checkpoint_dir, file_name, classes, save_best=True):
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'monitor_best': mnt_best,
        'classes': classes
    }

    # save_path = checkpoint_dir + '/model_' + str(epoch) + '.pth'
    # torch.save(state, save_path)

    if save_best:
        best_path = checkpoint_dir + '/' + file_name
        torch.save(state, best_path)
        print("Saving current best: model_best.pth ...")

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def progress(data_loader, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(data_loader, 'n_samples'):
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
    else:
        current = batch_idx
        total = len(data_loader)
    return base.format(current, total, 100.0 * current / total)

def load_checkpoint(model, resume_path, logger):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    epoch = checkpoint['epoch']
    mnt_best = checkpoint['monitor_best']

    # load architecture params from checkpoint.
    model.load_state_dict(checkpoint['state_dict'])

    logger.info("Checkpoint loaded from epoch {}".format(epoch))

    return model

# Customed TensorDataset
class CustomTensorDataset_BeatAligned(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *tensors, transform=None, p=0.5):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.p = p

    def __getitem__(self, index):
        x = self.tensors[0][0][index]
        x2 = self.tensors[0][1][index]
        torch.randn(1)

        if self.transform:
            if torch.rand(1) >= self.p:
                x = self.transform(x)

        y = self.tensors[1][index]
        w = self.tensors[2][index]

        return [x, x2], y, w

    def __len__(self):
        return self.tensors[0][0].size(0)

# Customed TensorDataset
# class CustomTensorDataset(Dataset):
#     """TensorDataset with support of transforms.
#     """
#     def __init__(self, *tensors, transform=None, p=0.5):
#         # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
#         self.tensors = tensors
#         self.transform = transform
#         self.p = p
#
#     def __getitem__(self, index):
#         x = self.tensors[0][index]
#         torch.randn(1)
#
#         if self.transform:
#             if torch.rand(1) >= self.p:
#                 x = self.transform(x)
#
#         y = self.tensors[1][0][index]
#         y2 = self.tensors[1][1][index]
#         w = self.tensors[2][index]
#
#         return x, [y, y2], w
#
#     def __len__(self):
#         return self.tensors[0].size(0)

# Customed TensorDataset
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *tensors, transform=None, p=0.5):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.p = p

    def __getitem__(self, index):
        x = self.tensors[0][index]
        torch.randn(1)

        if self.transform:
            if torch.rand(1) >= self.p:
                x = self.transform(x)

        y = self.tensors[1][index]
        w = self.tensors[2][index]

        return x, y, w

    def __len__(self):
        return self.tensors[0].size(0)
