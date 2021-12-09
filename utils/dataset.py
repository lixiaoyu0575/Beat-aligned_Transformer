import os
import numpy as np

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# Find Challenge files.
def find_challenge_files(label_directory):
    label_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f)  # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            # root, ext = os.path.splitext(f)
            label_files.append(F)
    if label_files:
        return label_files
    else:
        raise IOError('No label or output files found.')

# For each set of equivalent classes, replace each class with the representative class for the set.
def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes:
                classes[j] = multiple_classes[0] # Use the first class as the representative class.
    return classes

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
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
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

# Load weights.
def load_weights(weight_file, equivalent_classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)

    # For each collection of equivalent classes, replace each class with the representative class for the set.
    rows = replace_equivalent_classes(rows, equivalent_classes)

    # Check that equivalent classes have identical weights.
    for j, x in enumerate(rows):
        for k, y in enumerate(rows[j+1:]):
            if x==y:
                assert(np.all(values[j, :]==values[j+1+k, :]))
                assert(np.all(values[:, j]==values[:, j+1+k]))

    # Use representative classes.
    classes = [x for j, x in enumerate(rows) if x not in rows[:j]]
    indices = [rows.index(x) for x in classes]
    weights = values[np.ix_(indices, indices)]

    return classes, weights, indices

# Load labels from header/label files.
def load_labels(label_files, classes, equivalent_classes):
    # The labels should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)
    num_classes = len(classes)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        with open(label_files[i], 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    dxs = [arr.strip() for arr in l.split(': ')[1].split(',')]
                    dxs = replace_equivalent_classes(dxs, equivalent_classes)
                    tmp_labels.append(dxs)

    # Use one-hot encoding for labels.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for j, x in enumerate(classes):
            if x in dxs:
                labels[i, j] = 1

    return labels

# Load outputs from output files.
def load_outputs(output_files, classes, equivalent_classes):
    # The outputs should have the following form:
    #
    # diagnosis_1, diagnosis_2, diagnosis_3
    #           0,           1,           1
    #        0.12,        0.34,        0.56
    #
    num_recordings = len(output_files)
    num_classes = len(classes)

    # Load the outputs. Perform basic error checking for the output format.
    tmp_labels = list()
    tmp_binary_outputs = list()
    tmp_scalar_outputs = list()
    for i in range(num_recordings):
        with open(output_files[i], 'r') as f:
            lines = [l for l in f if l.strip() and not l.strip().startswith('#')]
            lengths = [len(l.split(',')) for l in lines]
            if len(lines)>=3 and len(set(lengths))==1:
                for j, l in enumerate(lines):
                    arrs = [arr.strip() for arr in l.split(',')]
                    if j==0:
                        row = arrs
                        row = replace_equivalent_classes(row, equivalent_classes)
                        tmp_labels.append(row)
                    elif j==1:
                        row = list()
                        for arr in arrs:
                            number = 1 if arr in ('1', 'True', 'true', 'T', 't') else 0
                            row.append(number)
                        tmp_binary_outputs.append(row)
                    elif j==2:
                        row = list()
                        for arr in arrs:
                            number = float(arr) if is_number(arr) else 0
                            row.append(number)
                        tmp_scalar_outputs.append(row)
            else:
                print('- The output file {} has formatting errors, so all outputs are assumed to be negative for this recording.'.format(output_files[i]))
                tmp_labels.append(list())
                tmp_binary_outputs.append(list())
                tmp_scalar_outputs.append(list())

    # Use one-hot encoding for binary outputs and the same order for scalar outputs.
    # If equivalent classes have different binary outputs, then the representative class is positive if any equivalent class is positive.
    # If equivalent classes have different scalar outputs, then the representative class is the mean of the equivalent classes.
    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for j, x in enumerate(classes):
            indices = [k for k, y in enumerate(dxs) if x==y]
            if indices:
                binary_outputs[i, j] = np.any([tmp_binary_outputs[i][k] for k in indices])
                tmp = [tmp_scalar_outputs[i][k] for k in indices]
                if np.any(np.isfinite(tmp)):
                    scalar_outputs[i, j] = np.nanmean(tmp)
                else:
                    scalar_outputs[i, j] = float('nan')

    # If any of the outputs is a NaN, then replace it with a zero.
    binary_outputs[np.isnan(binary_outputs)] = 0
    scalar_outputs[np.isnan(scalar_outputs)] = 0

    return binary_outputs, scalar_outputs