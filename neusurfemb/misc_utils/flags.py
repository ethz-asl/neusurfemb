import argparse
import copy
import os
import yaml

from typing import List


class ModelFlagsParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        self.add_argument('--workspace', type=str, default='workspace')

        self.add_argument(
            '--crop-res-train-dataset',
            type=int,
            help=(
                "If specified, the training dataset is cropped and resized to "
                "this size, and coordinate maps are rendered for each image, "
                "after NeuS2 training."))
        self.add_argument(
            '--one-uom-scene-to-m',
            type=float,
            required=True,
            help=("Conversion factor from one unit in the scale of the NeRF "
                  "model to one meter. NOTE: This depends on the dataset used "
                  "to train the NeuS model."))


class DatasetFlagsParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        self.add_argument(
            '--obj-id',
            type=int,
            help=("If given, ID of the object on which to train/test. This is "
                  "required for datasets in which more than one object is "
                  "contained, and has associated poses, in the scene (e.g, "
                  "YCB-V)."))


def yaml_to_param_list(yaml_file_path: str) -> List:
    param_list = []
    with open(yaml_file_path, "rb") as f:
        flags = yaml.load(f, Loader=yaml.SafeLoader)
    for param in flags:
        if (isinstance(param, dict)):
            assert (len(param) == 1)
            for key, value in param.items():
                param_list.append(f"--{key}")
                param_list.append(f"{value}")
        else:
            param_list.append(f"--{param}")

    return param_list


def save_experiment_flags(parsed_training_arguments: argparse.Namespace,
                          one_uom_scene_to_m: float,
                          output_yaml_file_path: str):
    model_flags_parser = ModelFlagsParser()

    assert (isinstance(parsed_training_arguments, argparse.Namespace))
    options_current_training = parsed_training_arguments.__dict__
    options_curr_training_as_list = []

    # Only consider flags that are flags also of `ModelFlagsParser`.
    options_current_training_in_flags_parser = {
        b: v
        for b, v in options_current_training.items()
        if b in [a.dest for a in model_flags_parser._actions]
    }

    for key, value in options_current_training_in_flags_parser.items():
        # - Find associated action.
        action = [a for a in model_flags_parser._actions if a.dest == key]
        assert (len(action) == 1)
        action = action[0]
        if (isinstance(action, argparse._StoreAction)):
            if (value is not None):
                options_curr_training_as_list.append(action.option_strings[0])
                options_curr_training_as_list.append(f"{value}")
        elif (isinstance(action, argparse._StoreFalseAction)):
            if (value == False):
                options_curr_training_as_list.append(action.option_strings[0])
        elif (isinstance(action, argparse._StoreTrueAction)):
            if (value == True):
                options_curr_training_as_list.append(action.option_strings[0])
        else:
            raise ValueError(
                "Currently, only `store`, `store_false` and `store_true` "
                "arguments are supported for conversion to command sequence.")

    required_options_flags_parser_not_in_current_training = [
        a for a in model_flags_parser._actions
        if (a.dest not in options_current_training and a.required and
            not isinstance(a, argparse._HelpAction))
    ]
    assert (len(required_options_flags_parser_not_in_current_training) == 1 and
            required_options_flags_parser_not_in_current_training[0].dest
            == "one_uom_scene_to_m")
    options_curr_training_as_list += [
        '--one-uom-scene-to-m', f"{one_uom_scene_to_m}"
    ]

    # Check that the arguments can be parsed.
    model_flags_parser.parse_args(options_curr_training_as_list)

    # Save the retrieved options as a YAML file, so that they can later be
    # loaded again (e.g., during pose estimation).
    if (param_list_to_yaml(param_list=options_curr_training_as_list,
                           output_yaml_file_path=output_yaml_file_path)):
        print("\033[94mSaved experiment config file to "
              f"'{output_yaml_file_path}'.\033[0m")


def param_list_to_yaml(param_list: List, output_yaml_file_path: str) -> None:
    # Very hacky. TODO(fmilano): Use `jsonargparse` instead.
    # Convert list to desired output YAML dumpable format.
    yaml_list = []
    prev_param_key_word = None
    prev_param_value_words = []
    for param_word in param_list:
        if (param_word[0] == "-"):
            if (len(param_word) == 2):
                assert (param_word[1] != "-")
                new_param_key_word = param_word[1:]
            else:
                if (param_word[1] != "-"):
                    new_param_key_word = param_word[1:]
                else:
                    assert (param_word[2] != "-")
                    new_param_key_word = param_word[2:]
            if (prev_param_key_word is not None):
                # Add the previous argument to the list.
                if (len(prev_param_value_words) == 0):
                    yaml_list.append(prev_param_key_word)
                elif (len(prev_param_value_words) == 1):
                    yaml_list.append(
                        {prev_param_key_word: prev_param_value_words[0]})
                else:
                    raise ValueError("Only single- or no-parameter arguments "
                                     "are currently supported.")
            # Start forming the new argument.
            prev_param_key_word = copy.copy(new_param_key_word)
            prev_param_value_words = []
        else:
            prev_param_value_words.append(param_word)

    if (prev_param_key_word is not None):
        # Add the last argument to the list.
        if (len(prev_param_value_words) == 0):
            yaml_list.append(prev_param_key_word)
        elif (len(prev_param_value_words) == 1):
            yaml_list.append({prev_param_key_word: prev_param_value_words[0]})
        else:
            raise ValueError("Only single- or no-parameter arguments are "
                             "currently supported.")

    if (os.path.exists(output_yaml_file_path)):
        # If the file already exists, check that the content is the same that
        # would be written.
        with open(output_yaml_file_path, "r") as f:
            existing_yaml_list = yaml.load(f, Loader=yaml.SafeLoader)

        assert (existing_yaml_list == yaml_list), (
            f"Found existing YAML file at '{output_yaml_file_path}', but its "
            "content does not match the configuration that would have been "
            f"written.\nExisting content:\n{existing_yaml_list}.\n"
            f"Configuration that would have been written:\n{yaml_list}.")
        print(f"\033[94mFound existing YAML file '{output_yaml_file_path}' "
              "with the same content as the configuration that would have been "
              "written. No further action is performed.\033[0m")
        return False
    else:
        # Save to YAML file.
        with open(output_yaml_file_path, "w") as f:
            yaml.dump(yaml_list, f)
        return True
