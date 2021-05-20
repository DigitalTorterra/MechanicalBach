def num_powers_of_two(n: int):
    if n % 2 == 1:
        return 0
    return 1 + num_powers_of_two(int(n/2))

def get_starting_size(num_hidden_conv_layers: int, n_vocab: int, input_length: int):
    """
    "Image" size: (input_length, n_vocab, 1)
    """
    d1_pows = num_powers_of_two(input_length)
    d2_pows = num_powers_of_two(n_vocab)

    num_upsizings = min(d1_pows, d2_pows, num_hidden_conv_layers)

    d1_base = int(input_length / 2**num_upsizings)
    d2_base = int(n_vocab / 2**num_upsizings)

    return (d1_base, d2_base), num_upsizings
