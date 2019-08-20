import numpy as np


# for relative imports of this script add parent directory to path
import sys

sys.path.append("..")

# relative imports (as we do not provide a proper package yet)
from ai_reflectivity import config_loader

from ai_reflectivity.neural_network import make_training_data
from ai_reflectivity.data_io import save_data_as_file, save_labels_as_file


def main():
    
    # use given config-file if specified
    if len(sys.argv)>1:
        config = config_loader.config(sys.argv[1])
    else:
        config = config_loader.config()
    number_of_training_samples, number_of_validation_samples = (
        config.get_number_of_training_samples()
    )
    training_data_file, training_labels_file, validation_data_file, validation_labels_file = (
        config.get_training_files()
    )
    np.random.seed(config.get_random_seed())

    [training_data, training_labels] = make_training_data(number_of_training_samples)
    [validation_data, validation_labels] = make_training_data(
        number_of_validation_samples
    )

    print(f"training data path: \t {training_data_file}")
    save_data_as_file(training_data, training_data_file)
    print(f"validation data path: \t {validation_data_file}")
    save_data_as_file(validation_data, validation_data_file)

    save_labels_as_file(training_labels, training_labels_file)
    save_labels_as_file(validation_labels, validation_labels_file)


if __name__ == "__main__":
    main()
