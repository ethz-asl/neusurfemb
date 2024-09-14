def string_to_bool(string):
    if (string.lower() == "true"):
        return True
    elif (string.lower() == "false"):
        return False
    else:
        raise ValueError("Please use either `true` or `false` (case does not "
                         "matter).")
