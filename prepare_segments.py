from data_loader.util import *

# Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
weights_file = 'weights.csv'
normal_class = '426783006'
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

input_directory_label = '/data/ecg/raw_data/challenge2020/bak_all_data_2020'
label_dir = '/data/ecg/raw_data/challenge2020/bak_all_data_2020'
# Find the label files.
print('Finding label and output files...')
label_files = load_label_files(input_directory_label)

# # Load the labels and classes.
# print('Loading labels and outputs...')
# label_classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

num_files = len(label_files)
# num_files = 5
recordings2save = []
ratio2save = []
for i in range(num_files):
    print('{}/{}'.format(i + 1, num_files))
    recording, header, name = load_challenge_data(label_files[i], label_dir)
    recording[np.isnan(recording)] = 0

    # divide ADC_gain and resample
    recording = resample(recording, header, 500)

    # slide and cut
    recording, info2save = slide_and_cut_beat_aligned(recording, 1, 5000, 500,
                                                      seg_with_r=False, beat_length=400)
    # print(recording)
    # print(info2save)
    recordings2save.append(recording[0])
    ratio2save.append(info2save)
recordings2save = np.array(recordings2save)
recordings2save = np.transpose(recordings2save, (0, 2, 1, 3))
ratio2save = np.array(ratio2save)

save_dir = '/data/ecg/challenge2020/data'

np.save(os.path.join(save_dir, 'recordings_' + str(5000) + '_' + str(
                500) + '_' + str(False) + '.npy'), recordings2save)
np.save(os.path.join(save_dir, 'info_' + str(5000) + '_' + str(
                500)  + '_' + str(False) + '.npy'), ratio2save)
print('done')
