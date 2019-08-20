import numpy as np
from scipy import stats
import keras
import config_loader
import datetime

config = config_loader.ConfigLoader('organic.config')


def main():
    now = datetime.datetime.now()
    tb_logdir = './Graph/' + now.strftime('%Y-%m-%d-%H%M%S')

    (training_data_file_name, training_labels_file_name, validation_data_file_name,
     validation_labels_file_name) = config.get_training_file_names()

    number_of_epochs = config.get_number_of_epochs()

    q_vector, training_data_sim = load_data(training_data_file_name)

    training_labels_sim = load_labels(training_labels_file_name)

    normalized_training_labels = normalize_labels(training_labels_sim)
    normalized_training_labels = remove_constant_labels(normalized_training_labels)

    number_of_q_values = len(q_vector)
    number_of_labels = normalized_training_labels.shape[1]

    training_input = np.log(training_data_sim)

    # NN architecture

    model = define_model(number_of_q_values, number_of_labels)
    #custom_object_dict = dict([('abs_err_%d' % (i + 1), y_absolute_error(i + 1)) for i in range(4)])
    #custom_object_dict['abs_err'] = y_absolute_error(0)
    #model = keras.models.load_model('my_model.hdf5', custom_objects=custom_object_dict)

    # compile model
    adam_optimizer = keras.optimizers.adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)

    model.compile(loss='mean_squared_error', optimizer=adam_optimizer,
                  metrics=[y_absolute_error(i) for i in range(number_of_labels)])

    # Fit the model

    tb_callback = keras.callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=0, write_graph=True, write_images=True)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=config.get_model_name(), monitor='val_loss', verbose=1,
                                                 save_best_only=True)

    hist = model.fit(training_input, normalized_training_labels, epochs=number_of_epochs, batch_size=128, verbose=2,
                     validation_split=0.20, callbacks=[checkpoint, tb_callback], initial_epoch=0)

    scores = model.evaluate(training_input, normalized_training_labels)
    #scores_eval = model.evaluate(validation_input, normalized_validation_labels)

    results = dict([(model.metrics_names[i], scores[i]) for i in range(number_of_labels + 1)])
    #results_eval = dict([(model.metrics_names[i], scores_eval[i]) for i in range(number_of_labels + 1)])

    print('Train:')
    print(results)
    #print('Test')
    #print(results_eval)


def load_data(data_file_path):
    data_file = np.loadtxt(data_file_path, delimiter='\t', skiprows=1)
    q_vector = data_file[:, 0]
    reflectivity = data_file[:, 1:].transpose()

    return q_vector, reflectivity


def load_labels(labels_file_path):
    labels = np.loadtxt(labels_file_path, delimiter='\t', skiprows=1)

    return labels


def normalize_labels(labels):
    min_thickness, max_thickness = config.get_thickness()
    min_roughness, max_roughness = config.get_roughness()
    min_scattering_length_density, max_scattering_length_density = config.get_scattering_length_density()

    min_labels = min_thickness + min_roughness + min_scattering_length_density
    max_labels = max_thickness + max_roughness + max_scattering_length_density

    number_of_labels = len(max_labels)

    for label in range(number_of_labels):
        if max_labels[label] != min_labels[label]:
            labels[:, label] = (labels[:, label] - min_labels[label]) / (max_labels[label] - min_labels[label])

    normalized_labels = labels

    return normalized_labels


def remove_oxide_thickness(labels):
    labels = np.delete(labels, 1, 1)

    return labels


def remove_constant_labels(labels):
    min_thickness, max_thickness = config.get_thickness()
    min_roughness, max_roughness = config.get_roughness()
    min_scattering_length_density, max_scattering_length_density = config.get_scattering_length_density()

    min_labels = min_thickness + min_roughness + min_scattering_length_density
    max_labels = max_thickness + max_roughness + max_scattering_length_density

    number_of_labels = labels.shape[1]

    for label in reversed(range(number_of_labels)):
        if min_labels[label] == max_labels[label]:
            labels = np.delete(labels, label, 1)

    return labels


def y_absolute_error(ind):
    def abs_err(y_true, y_pred):
        absolute_error = keras.backend.mean(abs(y_true[ind] - y_pred[ind]), axis=0)
        return absolute_error
    return abs_err


def gaussian(sigma, probe):
    return np.exp(-probe**2 / (2 * sigma**2))


def define_model(input_layer_size, output_layer_size, layer_size=400, n_layer=7):

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(400, input_dim=input_layer_size))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(800))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(400))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(300))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(200))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dense(output_layer_size))
    model.add(keras.layers.Activation('relu'))

    model.summary()

    return model


if __name__ == '__main__':
    main()
