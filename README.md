This folder includes the code for our monotone mean-field DEQ implementation.

-- Required package: torch, numpy, matplotlib, tqdm

-- Structure of the code: 
	-- proxsoftmax.py includes the implementation for our prox_f^alpha for alpha between 0 and 1. Notice that one cannot set alpha=1 here.
	-- multitier_conv.py includes the multi-tier structure of the DEQ, in the form of convolutions (notice that even dense linear layers can be re-written as convolutions)
		The most important arguments are 

		sizes = [(num_classes, 28, 28, 1),
         (40, 14, 14, 10,),
         (80, 7, 7, 20),
         (10, 1)]

		kernels = np.array([[3, 0, 0, 0],
		                    [3, 3, 0, 0],
		                    [3, 3, 3, 0],
		                    [0, 0, 1, 1]])

		Here 'sizes' specifies the (input_channel, input_height, input_width, input_groups) for convolutions, or (input_size, input_groups) for linear layers. 'kernels' specifies the connection and kernel sizes.

	-- deq_model.py includes the implementation for the forward-backward splitting. The most high-level class is 'ConvDeqCrf', which will be our model. 
	-- util.py includes utility functions like data preparation.
	-- train_and_eval.py includes training and evaluation. 
	-- main.py. The main entrance for running this code. One can directly run 'python -i main.py'. The variable 'num_classes' indicates the number of bins we use for discretizing the input intensity. For example, if num_classes=2 and the input 	intensity of MNIST image pixel at position (0,0) is x, then the binned output will be 0 if x<0.5, and 1 otherwise.
		MON_DEFAULTS = {
		    'alpha': 0.125,
		    'tol': 1e-2,
		    'max_iter': 50
		}
		specifies the arguments for the forward-backward splitting. 'alpha' is the alpha used in damped iteration and prox_f^alpha, 'tol' is the relative tolerance at which we stop the splitting, and 'max_iter' is the maximum number of (anderson) iterations.