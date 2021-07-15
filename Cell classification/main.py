import os
from training import TrainClassifierModel
from parse_arguments import get_parsed_arguments

if __name__ == '__main__':
    args = get_parsed_arguments()

    input_dir = args.input_dir
    output_dir = args.output_dir
    
    params = dict(
        output_dir=output_dir,
        training_data_dir=os.path.join(input_dir, 'train'),
        val_data_dir=os.path.join(input_dir, 'val'),
        epochs=300,
        patch_size=(28, 28, 3),
    )
    
    learning_rates = [1e-4, 1e-3]  # list of lr
    backends = ['vgg']  # ['vgg', 'inception']
    depths = [3, 4]  # list of depth
    bases = [32, 16]  # this the number of filters in the first hidden layer
    batch_sizes = [32, 48]  # list of patch sizes
    
    for lr in learning_rates:
        for depth in depths:
            for base in bases:
                for batch_size in batch_sizes:
                    for backend in backends:
                        params['backend'] = backend
                        params['learning_rate'] = lr
                        params['base'] = base
                        params['batch_size'] = batch_size
                        params['depth'] = depth
                        params['mapping_func'] = None
                        params['ratio_weight'] = None
                        obj = TrainClassifierModel(**params)
                        obj.run(all_params=params)
