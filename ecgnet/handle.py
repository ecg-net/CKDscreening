import datetime
import sys

print(sys.argv)


def train_wrapper(class_name, model_type, is_2d, lr, binary, conv_width, drop_prob, batch_norm, first_layer_out_channels,
                   first_layer_kernel_size, train_delta, train_signal_length, valid_signal_length, normalize_x, normalize_y, first_n=None):

    if batch_norm == "infer":
        batch_norm = lr >= 1e-3

    if normalize_y == "infer":
        normalize_y = lr >= 1e-2

    try:
        from train import train, name_model
        model_name=name_model(class_name, model_type, is_2d, lr, False, conv_width, drop_prob, batch_norm, first_layer_out_channels,
                      first_layer_kernel_size, train_delta, train_signal_length, valid_signal_length, normalize_x, normalize_y)
    except Exception as err:
        with open('../../output/ekg/reports/errors.txt', 'a') as f:
            f.write('import : {} {} \n'.format(datetime.datetime.now(), sys.argv))
        raise err

    try:
        train(
            save_path='../../output/ekg',
            class_name=class_name,
            model_type=model_type,
            is_2d=is_2d,
            lr=lr,
            binary=binary,
            conv_width=conv_width,
            drop_prob=drop_prob,
            batch_norm=batch_norm,
            first_layer_out_channels=first_layer_out_channels,
            first_layer_kernel_size=first_layer_kernel_size,
            train_delta=train_delta,
            train_signal_length=train_signal_length,
            valid_signal_length=valid_signal_length,
            normalize_x=normalize_x,
            normalize_y=normalize_y,
            first_n=first_n
        )


    except Exception as err:
        with open('../../output/ekg/reports/errors.txt', 'a') as f:
            f.write('runtime: {} {} \n'.format(datetime.datetime.now(), sys.argv))
            f.write(str(err))
            f.write('\n')
        with open('../../output/ekg/reports/{}.txt'.format(model_name), 'a') as f:
            f.write('runtime: {} {} \n'.format(datetime.datetime.now(), sys.argv))
            f.write(str(err))
        raise err
