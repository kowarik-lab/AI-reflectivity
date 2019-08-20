import numpy as np
import csv


def load_data(data_file_path):
    data_file = np.loadtxt(data_file_path, delimiter="\t", skiprows=1)
    q_vector = data_file[:, 0]
    reflectivity = data_file[:, 1:].transpose()

    return q_vector, reflectivity


def load_labels(labels_file_path):
    labels = np.loadtxt(labels_file_path, delimiter="\t", skiprows=1)

    return labels


def save_data_as_file(dataset, file_name):
    number_of_samples = dataset.shape[1] - 1
    header = []

    for i in range(number_of_samples):
        sample = i + 1
        header += ["Reflectivity {}".format(sample)]

    header = ["q [1/m]"] + header

    with open(file_name, "w", newline="") as f:

        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerow(header)
        writer.writerows(dataset)


def save_labels_as_file(labels, file_name):
    # TODO: make sure that lables are always passed as correctly shaped
    # ndarray and not as tuple
    if type(labels) == tuple:
        thicknesses = labels[0]
        roghnesses = labels[1]
        SLDs = labels[2]

        labels = np.concatenate((thicknesses, roghnesses, SLDs), axis=1)

    number_of_layers = int(labels.shape[1] / 3)

    thickness_header = []
    roughness_header = []
    SLD_header = []

    for i in range(number_of_layers):
        layer = i + 1
        thickness_header += ["Thickness {} [m]".format(layer)]
        roughness_header += ["Roughness {} [m^2]".format(layer)]
        SLD_header += ["Scattering length density {} [1/m^2]".format(layer)]

    header = thickness_header + roughness_header + SLD_header

    with open(file_name, "w", newline="") as f:

        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerow(header)
        writer.writerows(labels)
