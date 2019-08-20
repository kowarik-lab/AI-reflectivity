import configobj
from pathlib import Path, PurePath

"""
This config loader is intended to be used as Singleton since
it is accessed from different places within the project.

Please try to use pathlib wherever possible to ensure compatibility with 
Win/Mac/Linux
"""


def config(config_file_name=None):
    if _ConfigLoader._instance is None:
        if config_file_name is not None:
            _ConfigLoader._instance = _ConfigLoader(str(Path(config_file_name)))
        else:
            # fallback: intitialize with default
            _ConfigLoader._instance = _ConfigLoader(
                str(Path("../example_data/example.config"))
            )

    return _ConfigLoader._instance


# Singleton config class
class _ConfigLoader:
    _instance = None

    def __init__(self, config_file_name):
        self.config_object = configobj.ConfigObj(config_file_name)

    @property
    def config_file_path(self):
        return PurePath(self.config_object.filename).parent

    def get_q_values_file(self):
        q_values_file = self.config_object["training"]["q_values_file"]

        return self.config_file_path.joinpath(q_values_file)

    def get_log_dir(self):
        log_dir = self.config_object["training"]["log_dir"]

        return self.config_file_path.joinpath(log_dir)

    def get_training_files(self):
        training_data_file = self.config_file_path.joinpath(
            self.config_object["training"]["training_data_file"]
        )
        training_labels_file = self.config_file_path.joinpath(
            self.config_object["training"]["training_labels_file"]
        )

        validation_data_file = self.config_file_path.joinpath(
            self.config_object["training"]["validation_data_file"]
        )
        validation_labels_file = self.config_file_path.joinpath(
            self.config_object["training"]["validation_labels_file"]
        )

        return (
            training_data_file,
            training_labels_file,
            validation_data_file,
            validation_labels_file,
        )

    def get_random_seed(self):
        random_seed = int(self.config_object["training"]["random_seed"])

        return random_seed

    def get_number_of_training_samples(self):
        number_of_training_samples = int(
            self.config_object["training"]["number_of_training_samples"]
        )
        number_of_validation_samples = int(
            self.config_object["training"]["number_of_validation_samples"]
        )

        return number_of_training_samples, number_of_validation_samples

    def get_model_file(self):
        model_file = self.config_object["training"]["model_file"]

        return self.config_file_path.joinpath(model_file)

    def get_number_of_epochs(self):
        number_of_epochs = int(self.config_object["training"]["number_of_epochs"])

        return number_of_epochs

    def get_thickness(self):
        min_thickness = self._convert_to_list_of_floats(
            self.config_object["film-properties"]["min_thickness"]
        )
        max_thickness = self._convert_to_list_of_floats(
            self.config_object["film-properties"]["max_thickness"]
        )

        return min_thickness, max_thickness

    def get_roughness(self):
        min_roughness = self._convert_to_list_of_floats(
            self.config_object["film-properties"]["min_roughness"]
        )
        max_roughness = self._convert_to_list_of_floats(
            self.config_object["film-properties"]["max_roughness"]
        )

        return min_roughness, max_roughness

    def get_scattering_length_density(self):
        min_scattering_length_density = self._convert_to_list_of_floats(
            self.config_object["film-properties"]["min_scattering_length_density"]
        )
        max_scattering_length_density = self._convert_to_list_of_floats(
            self.config_object["film-properties"]["max_scattering_length_density"]
        )

        return min_scattering_length_density, max_scattering_length_density

    @staticmethod
    def _convert_to_list_of_floats(list_of_strings):
        if type(list_of_strings) == str:
            return [float(list_of_strings)]
        else:
            list_of_floats = [float(item) for item in list_of_strings]

        return list_of_floats

    def get_prediction_files(self):
        prediction_data_file = self.config_file_path.joinpath(
            self.config_object["prediction"]["prediction_data_file"]
        )
        prediction_labels_file = self.config_file_path.joinpath(
            self.config_object["prediction"]["prediction_labels_file"]
        )

        return (prediction_data_file, prediction_labels_file)

    def get_experimental_data_file(self):
        experimental_data_file = self.config_object["prediction"][
            "experimental_data_file"
        ]

        return self.config_file_path.joinpath(experimental_data_file)
