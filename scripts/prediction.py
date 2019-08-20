import numpy as np
import keras
import time

# to use code in ai_reflectivity without installing a package we add
# the root of this git to sys.path
import sys

sys.path.append("..")


from ai_reflectivity import xrrplots, neural_network, config_loader
from ai_reflectivity.data_io import save_labels_as_file, save_data_as_file
from ai_reflectivity.data_handling import restore_labels


def main():
    # load config
    # use given config-file if specified
    if len(sys.argv)>1:
        config = config_loader.config(sys.argv[1])
    else:
        config = config_loader.config()

    # load the model
    model_file = config.get_model_file()

    custom_object_dict = dict(
        [
            ("abs_err_%d" % (i + 1), neural_network.y_absolute_error(i + 1))
            for i in range(4)
        ]
    )
    custom_object_dict["abs_err"] = neural_network.y_absolute_error(0)
    model = keras.models.load_model(str(model_file), custom_objects=custom_object_dict)

    # import your test data
    test_data_file = config.get_experimental_data_file()
    test_data = np.loadtxt(test_data_file, delimiter="\t")

    q_vector = test_data[:, 0] * 1e10
    test_data = test_data[:, 1:]

    test_input = np.transpose(test_data)
    test_scores = np.log(test_input)

    time_before_prediction = time.perf_counter()
    predicted_labels = model.predict(test_scores)
    time_after_prediction = time.perf_counter()

    number_of_curves = predicted_labels.shape[0]

    labels = restore_labels(predicted_labels)

    number_of_layers = int(labels.shape[1] / 3)

    thicknesses = labels[:, 0:number_of_layers]
    roughnesses = labels[:, number_of_layers : 2 * number_of_layers]
    SLDs = labels[:, 2 * number_of_layers : 3 * number_of_layers]

    predicted_reflectivity = np.zeros([len(q_vector), number_of_curves + 1])
    predicted_reflectivity[:, 0] = q_vector

    predicted_reflectivity = neural_network.make_reflectivity_curves(
        q_vector, thicknesses, roughnesses, SLDs, number_of_curves
    )

    predicted_reflectivity = np.concatenate(
        (np.reshape(q_vector, (q_vector.shape[0], 1)), predicted_reflectivity), axis=1
    )

    prediction_files = config.get_prediction_files()

    save_labels_as_file(labels, prediction_files[1])
    save_data_as_file(predicted_reflectivity, prediction_files[0])

    total_prediction_time = time_after_prediction - time_before_prediction
    time_per_curve = total_prediction_time / number_of_curves
    print(
        f"Pure prediction time for {number_of_curves} curves: {total_prediction_time*1000:.2f} ms.\n"
    )
    print(f"Prediction time per curve: {time_per_curve * 1000:.2f} ms.\n")

    xrrplots.plot_reflectivity_gallery(
        q_vector, test_data, predicted_reflectivity[:, 1:], 10, output="show"
    )

    xrrplots.plot_thickness_vs_time(
        range(number_of_curves), thicknesses[:, 0], output="show"
    )

    xrrplots.plot_roughness_vs_time(
        range(number_of_curves), roughnesses[:, 0], output="show"
    )

    xrrplots.plot_SLD_vs_time(range(number_of_curves), SLDs[:, 0], output="show")


if __name__ == "__main__":
    main()
