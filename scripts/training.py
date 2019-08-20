import numpy as np
import keras
import datetime

# for relative imports of this script add parent directory to path
import sys

sys.path.append("..")

# relative imports (as we do not provide a proper package yet)
from ai_reflectivity import config_loader
from ai_reflectivity.neural_network import define_model, y_absolute_error
from ai_reflectivity.data_io import load_data, load_labels
from ai_reflectivity.data_handling import normalize_labels, remove_constant_labels


def main():
    config = config_loader.config()

    now = datetime.datetime.now()

    tb_logdir = config.get_log_dir() / now.strftime("%Y-%m-%d-%H%M%S")

    (
        training_data_file,
        training_labels_file,
        validation_data_file,
        validation_labels_file,
    ) = config.get_training_files()

    number_of_epochs = config.get_number_of_epochs()

    q_vector, training_data_sim = load_data(training_data_file)

    training_labels_sim = load_labels(training_labels_file)

    normalized_training_labels = normalize_labels(training_labels_sim)
    normalized_training_labels = remove_constant_labels(normalized_training_labels)

    number_of_q_values = len(q_vector)
    number_of_labels = normalized_training_labels.shape[1]

    training_input = np.log(training_data_sim)

    # NN architecture

    model = define_model(number_of_q_values, number_of_labels)
    # custom_object_dict = dict([('abs_err_%d' % (i + 1), y_absolute_error(i + 1)) for i in range(4)])
    # custom_object_dict['abs_err'] = y_absolute_error(0)
    # model = keras.models.load_model('my_model.hdf5', custom_objects=custom_object_dict)

    # compile model
    adam_optimizer = keras.optimizers.adam(
        lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False
    )

    model.compile(
        loss="mean_squared_error",
        optimizer=adam_optimizer,
        metrics=[y_absolute_error(i) for i in range(number_of_labels)],
    )

    # Fit the model

    tb_callback = keras.callbacks.TensorBoard(
        log_dir=tb_logdir, histogram_freq=0, write_graph=True, write_images=True
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(config.get_model_file()),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
    )

    model.fit(
        training_input,
        normalized_training_labels,
        epochs=number_of_epochs,
        batch_size=128,
        verbose=2,
        validation_split=0.20,
        callbacks=[checkpoint, tb_callback],
        initial_epoch=0,
    )

    scores = model.evaluate(training_input, normalized_training_labels)
    # scores_eval = model.evaluate(validation_input, normalized_validation_labels)

    results = dict(
        [(model.metrics_names[i], scores[i]) for i in range(number_of_labels + 1)]
    )
    # results_eval = dict([(model.metrics_names[i], scores_eval[i]) for i in range(number_of_labels + 1)])

    print("Train:")
    print(results)
    # print('Test')
    # print(results_eval)


if __name__ == "__main__":
    main()
