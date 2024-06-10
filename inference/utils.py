


def get_inference(args):
    if args.dimension == '2d':
        if args.sliding_window:
            from .inference2d import inference_sliding_window
            return inference_sliding_window
        else:
            from .inference2d import inference_whole_image
            return inference_whole_image

    elif args.dimension == '3d':
        if args.sliding_window:
            from .inference3d import inference_sliding_window
            return inference_sliding_window

        else:
            from .inference3d import inference_whole_image
            return inference_whole_image
        
    

    else:
        raise ValueError('Error in image dimension')


