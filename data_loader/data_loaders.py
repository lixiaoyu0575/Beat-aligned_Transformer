from torchvision import transforms
from torch.utils.data import TensorDataset
from base import BaseDataLoader
from data_loader.util import *
import augmentation.augmentation as module_augmentation


class ChallengeDataLoader_beat_aligned_data(BaseDataLoader):
    """
    challenge2020 data loading
    """

    def __init__(self, label_dir, split_index, batch_size, shuffle=True, num_workers=0, resample_Fs=300,
                 window_size=3000, n_segment=1, normalization=False, augmentations=None, p=0.5, _25classes=False,
                 lead_number=12, save_data=False, load_saved_data=True, save_dir=None, seg_with_r=False, beat_length=400):
        self.label_dir = label_dir
        print('Loading data...')

        weights_file = 'weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(label_dir)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class,
                                                                           equivalent_classes)
        self.classes = classes

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)
        self.weights = weights

        # Classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        indices_unscored = ~indices

        # Get number of samples for each category
        self.indices = indices

        split_idx = loadmat(split_index)
        train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
        train_index = train_index.reshape((train_index.shape[1],))
        val_index = val_index.reshape((val_index.shape[1],))

        num_files = len(label_files)
        train_recordings = list()
        train_class_weights = list()
        train_info = list()
        train_labels_onehot = list()

        val_recordings = list()
        val_class_weights = list()
        val_info = list()
        val_labels_onehot = list()

        file_names = list()

        ### class for dataset
        CPSC_classes = ['270492004', '164889003', '164909002', '284470004', '426783006',
                        '713427006']  # "59118001" = "713427006"
        CPSC_class_weight = np.zeros((108,))
        for cla in CPSC_classes:
            CPSC_class_weight[classes.index(cla)] = 1
        # CPSC_extra
        CPSC_extra_excluded_classes = ['445118002', '39732003', '251146004', '698252002', '10370003', '164947007',
                                       '111975006', '164917005', '47665007', '427393009', '426783006', '59931005']
        CPSC_extra_class_weight = np.ones((108,))
        for cla in CPSC_extra_excluded_classes:
            CPSC_extra_class_weight[classes.index(cla)] = 0
        # PTB-XL
        PTB_XL_excluded_classes = ['426627000', '427172004']  # , '17338001'
        PTB_XL_class_weight = np.ones((108,))
        for cla in PTB_XL_excluded_classes:
            PTB_XL_class_weight[classes.index(cla)] = 0
        # G12ECG
        G12ECG_excluded_classes = ['10370003', '164947007']
        G12ECG_class_weight = np.ones((108,))
        for cla in G12ECG_excluded_classes:
            G12ECG_class_weight[classes.index(cla)] = 0

        recordings_saved = np.load(os.path.join('/data/ecg/challenge2020/data/recordings_' + str(5000) + '_' + str(
            500) + '_' + str(False) + '.npy'))
        ratio_saved = np.load(os.path.join('/data/ecg/challenge2020/data/info_' + str(5000) + '_' + str(
            500) + '_' + str(False) + '.npy'))

        if load_saved_data is False:

            ### preprocess data and label
            for i in range(num_files):
                print('{}/{}'.format(i+1, num_files))
                recording, header, name = load_challenge_data(label_files[i], label_dir)
                if name[0] == 'S' or name[0] == 'I': # PTB or St.P dataset
                    continue
                elif name[0] == 'A': # CPSC
                    class_weight = CPSC_class_weight
                elif name[0] == 'Q': # CPSC-extra
                    class_weight = CPSC_extra_class_weight
                elif name[0] == 'H': # PTB-XL
                    class_weight = PTB_XL_class_weight
                elif name[0] == 'E': # G12ECG
                    class_weight = G12ECG_class_weight
                else:
                    print('warning! not from one of the datasets')
                    print(name)

                ### coarse label

                recording[np.isnan(recording)] = 0
                recording = [recordings_saved[i]]
                info2save = ratio_saved[i]
                if info2save.sum() <= 0:
                    continue

                if recording is None:
                    print('escape 1 recording')
                    continue
                file_names.append(name)

                if i in train_index:
                    for j in range(1):
                        train_recordings.append(recording[j])
                        train_labels_onehot.append(labels_onehot[i])
                        train_class_weights.append(class_weight)
                        train_info.append(info2save[j])
                elif i in val_index:
                    for j in range(1):
                        val_recordings.append(recording[j])
                        val_labels_onehot.append(labels_onehot[i])
                        val_class_weights.append(class_weight)
                        val_info.append(info2save[j])
                else:
                    pass

            print(np.isnan(train_recordings).any())
            print(np.isnan(val_recordings).any())

            train_recordings = np.array(train_recordings)
            train_class_weights = np.array(train_class_weights)
            train_labels_onehot = np.array(train_labels_onehot)
            train_info = np.array(train_info)

            val_recordings = np.array(val_recordings)
            val_class_weights = np.array(val_class_weights)
            val_info = np.array(val_info)
            val_labels_onehot = np.array(val_labels_onehot)

        else:
            ### get saved data and label
            train_recordings = np.load(
                os.path.join(save_dir, 'train_recordings_' + str(window_size) + '_' + str(
                    resample_Fs) + '_' + str(seg_with_r) + '.npy'))
            train_class_weights = np.load(
                os.path.join(save_dir, 'train_class_weights_' + str(window_size) + '_' + str(
                    resample_Fs) + '_' + str(seg_with_r) + '.npy'))
            train_labels_onehot = np.load(
                os.path.join(save_dir, 'train_labels_onehot_' + str(window_size) + '_' + str(
                    resample_Fs) + '_' + str(seg_with_r) + '.npy'))
            train_info = np.load(os.path.join(save_dir, 'train_info' + str(window_size) + '_' + str(
                resample_Fs) + '_' + str(seg_with_r) + '.npy'))

            val_info = np.load(os.path.join(save_dir, 'val_info_' + str(window_size) + '_' + str(
                resample_Fs) + '_' + str(seg_with_r) + '.npy'))
            val_recordings = np.load(
                os.path.join(save_dir, 'val_recordings_' + str(window_size) + '_' + str(
                    resample_Fs) + '_' + str(seg_with_r) + '.npy'))
            val_class_weights = np.load(
                os.path.join(save_dir, 'val_class_weights_' + str(window_size) + '_' + str(
                    resample_Fs) + '_' + str(seg_with_r) + '.npy'))
            val_labels_onehot = np.load(
                os.path.join(save_dir, 'val_labels_onehot_' + str(window_size) + '_' + str(
                    resample_Fs) + '_' + str(seg_with_r) + '.npy'))

            print('got data and label!!!')

        if save_data:
            np.save(os.path.join(save_dir, 'train_recordings_' + str(window_size) + '_' + str(
                resample_Fs) + '_' + str(seg_with_r) + '.npy'), train_recordings)
            np.save(os.path.join(save_dir, 'train_class_weights_' + str(window_size) + '_' + str(
                resample_Fs) + '_' + str(seg_with_r) + '.npy'), train_class_weights)
            np.save(os.path.join(save_dir, 'train_info' + str(window_size) + '_' + str(
                resample_Fs) + '_' + str(seg_with_r) + '.npy'), train_info)
            np.save(os.path.join(save_dir, 'train_labels_onehot_' + str(window_size) + '_' + str(
                resample_Fs) + '_' + str(seg_with_r) + '.npy'), train_labels_onehot)

            np.save(os.path.join(save_dir, 'val_recordings_' + str(window_size) + '_' + str(
                resample_Fs) + '_' + str(seg_with_r) + '.npy'), val_recordings)
            np.save(os.path.join(save_dir, 'val_class_weights_' + str(window_size) + '_' + str(
                resample_Fs) + '_' + str(seg_with_r) + '.npy'), val_class_weights)
            np.save(os.path.join(save_dir, 'val_info_' + str(window_size) + '_' + str(
                resample_Fs)  + '_' + str(seg_with_r) + '.npy'), val_info)
            np.save(os.path.join(save_dir, 'val_labels_onehot_' + str(window_size) + '_' + str(
                resample_Fs)  + '_' + str(seg_with_r) + '.npy'), val_labels_onehot)

            print('data saved!!!')

        # Normalization
        if normalization:
            train_recordings = self.normalization(train_recordings)
            val_recordings = self.normalization(val_recordings)

        X_train = torch.from_numpy(train_recordings).float()
        X_train_class_weight = torch.from_numpy(train_class_weights).float()
        train_info = np.array(train_info)

        X_train_info = torch.from_numpy(train_info).float()
        Y_train = torch.from_numpy(train_labels_onehot).float()

        X_val = torch.from_numpy(val_recordings).float()
        X_val_class_weight = torch.from_numpy(val_class_weights).float()
        val_info = np.array(val_info)

        X_val_info = torch.from_numpy(val_info).float()
        Y_val = torch.from_numpy(val_labels_onehot).float()

        leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        if lead_number == 2:
            # two leads
            leads_index = [1, 10]
        elif lead_number == 3:
            # three leads
            leads_index = [0, 1, 7]
        elif lead_number == 6:
            # six leads
            leads_index = [0, 1, 2, 3, 4, 5]
        else:
            # eight leads
            leads_index = [0, 1, 6, 7, 8, 9, 10, 11]

        ### different leads in the same shape
        X_train_tmp = X_train[:, leads_index, :, :]
        X_val_tmp = X_val[:, leads_index, :, :]

        X_train = X_train_tmp
        X_val = X_val_tmp

        if augmentations:
            transformers = list()

            for key, value in augmentations.items():
                module_args = dict(value['args'])
                transformers.append(getattr(module_augmentation, key)(**module_args))

            train_transform = transforms.Compose(transformers)
            self.train_dataset = CustomTensorDataset_BeatAligned([X_train, X_train_info], Y_train,
                                                                 X_train_class_weight, transform=train_transform, p=p)
        else:
            self.train_dataset = CustomTensorDataset_BeatAligned([X_train, X_train_info], Y_train,
                                                                 X_train_class_weight)
        self.val_dataset = CustomTensorDataset_BeatAligned([X_val, X_val_info], Y_val,
                                                           X_val_class_weight)

        super().__init__(self.train_dataset, self.val_dataset, None, batch_size, shuffle, num_workers)

        self.valid_data_loader.file_names = file_names
        self.valid_data_loader.idx = val_index

    def normalization(self, X):
        return X


class ChallengeDataLoader(BaseDataLoader):
    """
    challenge2020 data loading
    """

    def __init__(self, label_dir, split_index, batch_size, shuffle=True, num_workers=2, resample_Fs=300,
                 window_size=3000, n_segment=1, normalization=False, training_size=None, augmentations=None, p=0.5,
                 lead_number=12, save_data=False, load_saved_data=True, save_dir=None):
        self.label_dir = label_dir
        print('Loading data...')

        weights_file = 'weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(label_dir)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class,
                                                                           equivalent_classes)
        self.classes = classes

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)
        self.weights = weights

        # Classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        indices_unscored = ~indices

        # Get number of samples for each category
        self.indices = indices

        split_idx = loadmat(split_index)
        train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
        train_index = train_index.reshape((train_index.shape[1],))
        val_index = val_index.reshape((val_index.shape[1],))
        test_index = test_index.reshape((test_index.shape[1],))

        num_files = len(label_files)
        train_recordings = list()
        train_class_weights = list()
        train_labels_onehot = list()

        val_recordings = list()
        val_class_weights = list()
        val_labels_onehot = list()

        file_names = list()

        CPSC_classes = ['270492004', '164889003', '164909002', '284470004', '426783006',
                        '713427006']  # "59118001" = "713427006"
        CPSC_class_weight = np.zeros((108,))
        for cla in CPSC_classes:
            CPSC_class_weight[classes.index(cla)] = 1
        # CPSC_extra
        CPSC_extra_excluded_classes = ['445118002', '39732003', '251146004', '698252002', '10370003', '164947007',
                                       '111975006', '164917005', '47665007', '427393009', '426783006', '59931005']
        CPSC_extra_class_weight = np.ones((108,))
        for cla in CPSC_extra_excluded_classes:
            CPSC_extra_class_weight[classes.index(cla)] = 0
        # PTB-XL
        PTB_XL_excluded_classes = ['426627000', '427172004']  # , '17338001'
        PTB_XL_class_weight = np.ones((108,))
        for cla in PTB_XL_excluded_classes:
            PTB_XL_class_weight[classes.index(cla)] = 0
        # G12ECG
        G12ECG_excluded_classes = ['10370003', '164947007']
        G12ECG_class_weight = np.ones((108,))
        for cla in G12ECG_excluded_classes:
            G12ECG_class_weight[classes.index(cla)] = 0

        if load_saved_data == False:
            ### preprocess data and label
            for i in range(num_files):
                print('{}/{}'.format(i + 1, num_files))
                recording, header, name = load_challenge_data(label_files[i], label_dir)

                if name[0] == 'S' or name[0] == 'I':  # PTB or St.P dataset
                    continue
                elif name[0] == 'A':  # CPSC
                    class_weight = CPSC_class_weight
                elif name[0] == 'Q':  # CPSC-extra
                    class_weight = CPSC_extra_class_weight
                elif name[0] == 'H':  # PTB-XL
                    class_weight = PTB_XL_class_weight
                elif name[0] == 'E':  # G12ECG
                    class_weight = G12ECG_class_weight
                else:
                    print('warning! not from one of the datasets')
                    print(name)

                ### coarse label

                recording[np.isnan(recording)] = 0

                # divide ADC_gain and resample
                recording = resample(recording, header, resample_Fs)

                # slide and cut
                recording = slide_and_cut(recording, n_segment, window_size, resample_Fs)
                # np.save('/data/ecg/preprocessed_data/challenge2020/recordings_'+str(window_size)+'_'+str(resample_Fs)+'.npy', recording)
                # recording = np.load('/data/ecg/preprocessed_data/challenge2020/recordings_'+str(window_size)+'_'+str(resample_Fs)+'.npy')
                file_names.append(name)
                if i in train_index:
                    for j in range(recording.shape[0]):
                        train_recordings.append(recording[j])
                        train_labels_onehot.append(labels_onehot[i])
                        train_class_weights.append(class_weight)
                elif i in val_index:
                    for j in range(recording.shape[0]):
                        val_recordings.append(recording[j])
                        val_labels_onehot.append(labels_onehot[i])
                        val_class_weights.append(class_weight)
                else:
                    pass

            train_recordings = np.array(train_recordings)
            train_class_weights = np.array(train_class_weights)
            train_labels_onehot = np.array(train_labels_onehot)

            val_recordings = np.array(val_recordings)
            val_class_weights = np.array(val_class_weights)
            val_labels_onehot = np.array(val_labels_onehot)

        else:
            train_recordings = np.load(os.path.join(save_dir, 'train_recordings_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'))
            train_class_weights = np.load(os.path.join(save_dir, 'train_class_weights_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'))
            train_labels_onehot = np.load(os.path.join(save_dir, 'train_labels_onehot_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'))
            val_recordings = np.load(os.path.join(save_dir, 'val_recordings_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'), )
            val_class_weights = np.load(os.path.join(save_dir, 'val_class_weights_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'), )
            val_labels_onehot = np.load(os.path.join(save_dir, 'val_labels_onehot_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'), )


        if save_data:
            np.save(os.path.join(save_dir, 'train_recordings_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'), train_recordings)
            np.save(os.path.join(save_dir, 'train_class_weights_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'), train_class_weights)
            np.save(os.path.join(save_dir, 'train_labels_onehot_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'), train_labels_onehot)
            np.save(os.path.join(save_dir, 'val_recordings_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'), val_recordings)
            np.save(os.path.join(save_dir, 'val_class_weights_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'), val_class_weights)
            np.save(os.path.join(save_dir, 'val_labels_onehot_' + str(window_size) + '_' + str(
                resample_Fs) + '.npy'), val_labels_onehot)
            print('data saved!!!')

        # Normalization
        if normalization:
            train_recordings = self.normalization(train_recordings)
            val_recordings = self.normalization(val_recordings)

        print(np.isnan(train_recordings).any())
        print(np.isnan(val_recordings).any())
        print(np.isnan(train_class_weights).any())
        print(np.isnan(val_class_weights).any())

        X_train = torch.from_numpy(train_recordings).float()
        X_train_class_weight = torch.from_numpy(train_class_weights).float()
        Y_train = torch.from_numpy(train_labels_onehot).float()

        X_val = torch.from_numpy(val_recordings).float()
        X_val_class_weight = torch.from_numpy(val_class_weights).float()
        Y_val = torch.from_numpy(val_labels_onehot).float()


        if lead_number == 2:
            # two leads
            leads_index = [1, 10]
        elif lead_number == 3:
            # three leads
            leads_index = [0, 1, 7]
        elif lead_number == 6:
            # six leads
            leads_index = [0, 1, 2, 3, 4, 5]
        elif lead_number == 8:
            # eight leads
            leads_index = [0, 1, 6, 7, 8, 9, 10, 11]
        else:
            leads_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        X_train = X_train[:, leads_index, :]
        X_val = X_val[:, leads_index, :]

        #############################################################
        if augmentations:
            transformers = list()

            for key, value in augmentations.items():
                module_args = dict(value['args'])
                transformers.append(getattr(module_augmentation, key)(**module_args))

            train_transform = transforms.Compose(transformers)
            self.train_dataset = CustomTensorDataset(X_train, Y_train, X_train_class_weight, transform=train_transform,
                                                     p=p)
        else:
            self.train_dataset = TensorDataset(X_train, Y_train, X_train_class_weight)

        self.val_dataset = CustomTensorDataset(X_val, Y_val, X_val_class_weight)

        super().__init__(self.train_dataset, self.val_dataset, None, batch_size, shuffle, num_workers)

        self.valid_data_loader.file_names = file_names
        self.valid_data_loader.idx = val_index


    def normalization(self, X):
        return X
