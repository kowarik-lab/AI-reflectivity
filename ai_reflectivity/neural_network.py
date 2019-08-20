import numpy as np
import keras

from .config_loader import config
from .data_handling import make_reflectivity_curves


def y_absolute_error(ind):
    def abs_err(y_true, y_pred):
        absolute_error = keras.backend.mean(abs(y_true[ind] - y_pred[ind]), axis=0)
        return absolute_error

    return abs_err


def define_model(input_layer_size, output_layer_size, layer_size=400, n_layer=7):

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(400, input_dim=input_layer_size))
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.Dense(800))
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.Dense(400))
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.Dense(300))
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.Dense(200))
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.Dense(output_layer_size))
    model.add(keras.layers.Activation("relu"))

    model.summary()

    return model


################### old genereate TODO: remove comment


# delta_q = q_values * 0


def make_training_data(n_samples):
    test_data_file = config().get_q_values_file()
    test_data = np.loadtxt(test_data_file)  # in 1/m
    q_values = test_data[:, 0] * 1e10

    training_data_input = make_training_input(n_samples)
    [thicknesses, roughnesses, SLDs] = training_data_input

    training_reflectivity = make_reflectivity_curves(
        q_values, thicknesses, roughnesses, SLDs, n_samples
    )

    training_data_output = np.zeros([len(q_values), n_samples + 1])
    training_data_output[:, 0] = q_values
    training_data_output[:, 1:] = training_reflectivity

    return training_data_output, training_data_input


def make_training_input(number_of_sets):

    training_data_file, training_labels_file, validation_file, validation_labels_file = (
        config().get_training_files()
    )

    number_of_training_samples, number_of_validation_samples = (
        config().get_number_of_training_samples()
    )

    min_thickness, max_thickness = config().get_thickness()

    min_roughness, max_roughness = config().get_roughness()

    min_scattering_length_density, max_scattering_length_density = (
        config().get_scattering_length_density()
    )

    randomized_thicknesses = randomize_inputs(
        min_thickness, max_thickness, number_of_sets
    )

    # randomized_roughnesses = randomize_inputs(min_roughness, max_roughness, number_of_sets)
    randomized_roughnesses = np.zeros_like(randomized_thicknesses)

    for sample in range(number_of_sets):
        for layer in range(randomized_thicknesses.shape[1]):
            max_roughness_by_thickness = 0.5 * randomized_thicknesses[sample, layer]

            random_roughness_by_thickness = randomize_inputs(
                [min_roughness[layer]], [max_roughness_by_thickness], 1
            )

            if random_roughness_by_thickness > max_roughness[layer]:
                randomized_roughnesses[sample, layer] = max_roughness[layer]
            elif random_roughness_by_thickness < min_roughness[layer]:
                randomized_roughnesses[sample, layer] = min_roughness[layer]
            else:
                randomized_roughnesses[sample, layer] = random_roughness_by_thickness

    randomized_SLDs = randomize_inputs(
        min_scattering_length_density, max_scattering_length_density, number_of_sets
    )

    return randomized_thicknesses, randomized_roughnesses, randomized_SLDs


def randomize_inputs(min_value, max_value, number_of_samples):
    min_value = np.asarray(min_value)
    max_value = np.asarray(max_value)

    if np.all(np.isreal(min_value)) and np.all(np.isreal(max_value)):

        number_of_layers = len(min_value)

        randomized_inputs = np.zeros([number_of_samples, number_of_layers])

        for layer in range(number_of_layers):
            randomized_inputs[:, layer] = np.random.uniform(
                min_value[layer], max_value[layer], number_of_samples
            )

        return randomized_inputs

    else:
        real_min_value = min_value.real
        real_max_value = max_value.real

        imag_min_value = min_value.imag
        imag_max_value = max_value.imag

        real_randomized_inputs = randomize_inputs(
            real_min_value, real_max_value, number_of_samples
        )
        imag_randomized_inputs = randomize_inputs(
            imag_min_value, imag_max_value, number_of_samples
        )

        complex_randomized_inputs = real_randomized_inputs + 1j * imag_randomized_inputs

        return complex_randomized_inputs
