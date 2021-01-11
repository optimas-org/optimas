def determine_fidelity_type_and_length(mf_parameters):
    """
    Determine the type of the fidelity (i.e. float, int, str...) and, if it
    is a string, also its length.
    """
    # Check that all fidelities in 'range' are of the same type.
    fidel_types = [type(z) for z in mf_parameters['range']]
    if fidel_types.count(fidel_types[0]) != len(fidel_types):
        raise ValueError("The fidelities in 'range' are of different types.")
    fidel_type = fidel_types[0]
    fidel_len = None
    # If fidelities are strings, determine the lenght of the longest one
    # so that it can be fully stored in a numpy array.
    if fidel_type == str:
        str_lengths = [len(z) for z in mf_parameters['range']]
        fidel_len = max(str_lengths)
    return fidel_type, fidel_len
