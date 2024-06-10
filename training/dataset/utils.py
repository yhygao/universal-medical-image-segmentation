

def get_dataset(args, mode, **kwargs):
    
    if args.dimension == '3d':
        if args.dataset == 'universal':
            from .dim3.dataset_universal import UniversalDataset

            return UniversalDataset(args, dataset_list=kwargs['dataset_name_list'], mode=mode)

        else:
            raise NameError(f"No {args.dataset} dataset")
