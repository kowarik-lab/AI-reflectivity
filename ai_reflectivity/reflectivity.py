import numpy as np


def multilayer_reflectivity(q, thickness_layer, roughness_interface, SLD_layer):
    if len(thickness_layer) != len(SLD_layer):
        raise ValueError("number of thicknesses must be equal to number of SLDs")

    if len(roughness_interface) == len(SLD_layer):
        number_of_interfaces = len(roughness_interface)
    else:
        raise ValueError("number of roughnesses must be equal to number of SLDs")

    k_z0 = q / 2
    SLD_air = 0.0
    thickness_air = 1.0

    for interface in range(number_of_interfaces):

        prev_layer = interface - 1
        next_layer = interface

        if interface == 0:
            thickness_prev_layer = thickness_air
            k_z_previous_layer = _get_k_z(k_z0, SLD_air)
        else:
            thickness_prev_layer = thickness_layer[prev_layer] * np.ones_like(q, "d")
            k_z_previous_layer = _get_k_z(k_z0, SLD_layer[prev_layer])

        k_z_next_layer = _get_k_z(k_z0, SLD_layer[next_layer])
        current_roughness = roughness_interface[interface] * np.ones_like(q, "d")

        R = _make_reflection_matrix(
            k_z_previous_layer, k_z_next_layer, current_roughness
        )

        if interface == 0:
            M = R
        else:
            T = _make_translation_matrix(k_z_previous_layer, thickness_prev_layer)

            for n in range(len(q)):
                M[:, :, n] = np.matmul(M[:, :, n], T[:, :, n])
                M[:, :, n] = np.matmul(M[:, :, n], R[:, :, n])

    r = np.zeros_like(q, "D")

    for n in range(len(r)):
        r[n] = M[0, 1, n] / M[1, 1, n]

    reflectivity = abs(r) ** 2
    reflectivity.reshape(len(reflectivity), 1)

    return reflectivity


def _get_k_z(k_z0, scattering_length_density):
    return np.sqrt(np.clip(k_z0 ** 2 - 4 * np.pi * scattering_length_density, 0, None))


def _make_reflection_matrix(k_z_previous_layer, k_z_next_layer, interface_roughness):
    p = _safe_div(
        abs(k_z_previous_layer + k_z_next_layer), (2 * k_z_previous_layer)
    ) * np.exp(
        -(k_z_previous_layer - k_z_next_layer) ** 2 * 0.5 * interface_roughness ** 2
    )

    m = _safe_div(
        abs(k_z_previous_layer - k_z_next_layer), (2 * k_z_previous_layer)
    ) * np.exp(
        -(k_z_previous_layer + k_z_next_layer) ** 2 * 0.5 * interface_roughness ** 2
    )

    R = np.array([[p, m], [m, p]])

    R = R + 1j * 0

    return R


def _make_translation_matrix(k_z, thickness):
    return np.array(
        [
            [np.exp(-1j * k_z * thickness), np.zeros_like(k_z)],
            [np.zeros_like(k_z), np.exp(1j * k_z * thickness)],
        ]
    )


def _safe_div(numerator, denominator):
    result = np.zeros_like(numerator, "D")
    length = len(numerator)
    for i in range(length):

        if numerator[i] == denominator[i]:
            result[i] = 1
        elif denominator[i] == 0:
            result[i] = numerator[i] / 1e-20
        else:
            result[i] = numerator[i] / denominator[i]

    return result
