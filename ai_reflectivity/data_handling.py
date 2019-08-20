import numpy as np
from tqdm import tqdm

from . import reflectivity as refl
from .config_loader import config


def restore_labels(labels):
    min_thickness, max_thickness = config().get_thickness()
    min_roughness, max_roughness = config().get_roughness()
    min_scattering_length_density, max_scattering_length_density = (
        config().get_scattering_length_density()
    )

    min_labels = min_thickness + min_roughness + min_scattering_length_density
    max_labels = max_thickness + max_roughness + max_scattering_length_density

    number_of_labels = len(max_labels)

    for label in range(number_of_labels):
        if min_labels[label] == max_labels[label]:
            labels = np.insert(labels, label, max_labels[label], axis=1)
        else:
            labels[:, label] = (
                labels[:, label] * (max_labels[label] - min_labels[label])
                + min_labels[label]
            )

    restored_labels = labels

    return restored_labels


def normalize_labels(labels):
    min_thickness, max_thickness = config().get_thickness()
    min_roughness, max_roughness = config().get_roughness()
    min_scattering_length_density, max_scattering_length_density = (
        config().get_scattering_length_density()
    )

    min_labels = min_thickness + min_roughness + min_scattering_length_density
    max_labels = max_thickness + max_roughness + max_scattering_length_density

    number_of_labels = len(max_labels)

    for label in range(number_of_labels):
        if max_labels[label] != min_labels[label]:
            labels[:, label] = (labels[:, label] - min_labels[label]) / (
                max_labels[label] - min_labels[label]
            )

    normalized_labels = labels

    return normalized_labels


def remove_oxide_thickness(labels):
    labels = np.delete(labels, 1, 1)

    return labels


def insert_oxide_thickness(path_train_data, labels):
    train_data = np.loadtxt(path_train_data, skiprows=1)

    labels = np.insert(labels, 1, train_data[:, 1])

    return labels


def remove_constant_labels(labels):
    min_thickness, max_thickness = config().get_thickness()
    min_roughness, max_roughness = config().get_roughness()
    min_scattering_length_density, max_scattering_length_density = (
        config().get_scattering_length_density()
    )

    min_labels = min_thickness + min_roughness + min_scattering_length_density
    max_labels = max_thickness + max_roughness + max_scattering_length_density

    number_of_labels = labels.shape[1]

    for label in reversed(range(number_of_labels)):
        if min_labels[label] == max_labels[label]:
            labels = np.delete(labels, label, 1)

    return labels


def make_reflectivity_curves(
    q_values, thicknesses, roughnesses, SLDs, number_of_curves
):
    reflectivity_curves = np.zeros([len(q_values), number_of_curves])

    for curve in tqdm(range(number_of_curves)):
        reflectivity = refl.multilayer_reflectivity(
            q_values, thicknesses[curve, :], roughnesses[curve, :], SLDs[curve, :]
        )

        reflectivity_noisy = apply_shot_noise(reflectivity)
        reflectivity_curves[:, curve] = reflectivity_noisy

    return reflectivity_curves


def apply_slit_convolution(q, reflectivity, sigma=0.001):
    conv_reflectivity = np.zeros_like(reflectivity)
    for i in range(len(conv_reflectivity)):
        q_pos = q[i]
        g = gauss(q, sigma, q_pos)
        g_norm = g / sum(g)

        weighted_reflectivity = g_norm * reflectivity
        conv_reflectivity[i] = sum(weighted_reflectivity)
    return conv_reflectivity


def gauss(x, sigma=1, mu=0):
    g = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return g


def apply_shot_noise(reflectivity_curve):
    noise_factor = 0

    noisy_reflectivity = np.clip(
        np.random.normal(
            reflectivity_curve, noise_factor * np.sqrt(reflectivity_curve)
        ),
        1e-8,
        None,
    )

    return noisy_reflectivity
