# Explaining CNN Using Layer-Wise Relevance Propagation

## Jupyter notebooks:

### `accuracy_check_imagenette`:

Checks prediction accuracies of CNN models on [Imagenette dataset](https://github.com/fastai/imagenette)

### `accuracy_check_imagewoof`:

Checks prediction accuracies of CNN models on [Imagewoof dataset](https://github.com/fastai/imagenette)

### `analysis_confidence_score_original`:

Picks out images with highest and lowest confidence scores given that the prediction is correct (applied on pre-trained model)

### `lrp`:

Performs layer-wise relevance propagation on selected images. Images are put into the `input` folder, and the resulting images are put in the `output` folder.

### `vgg19_evaluation_csv`:

Evaluates the model on validation set, and store image paths, true label, predicted label, predicted confidence score, true confidence score, and correctness into a csv file.

### `vgg19_retrain_imagenette`:

Retrain the vgg19 model on [Imagenette dataset](https://github.com/fastai/imagenette) by modifying the final output channel to 10 categories. The resulting model is saved to a pth file.

### `vgg_error_analysis_original`:

Analyse the mistakes made by pre-trained vgg19 model on [Imagenette dataset](https://github.com/fastai/imagenette).

## CSV files:

### `vgg19_results_original`:

Resulting csv file acquired from `vgg19_evaluation_csv.ipynb`. Columns contain image paths, true label, predicted label, predicted confidence score, true confidence score, and correctness. (pre-trained vgg19 model)

### `vgg19_results_retrained`:

Same as above, except that the associated model is the retrained vgg19 model.

## Folder:

### `input`:

Contains images that serves as input to `lrp.ipynb`. Images need to be put into a separate folder named by the category an image belongs to:

\- input

    \- English springer

        \- 1.jpg

        \- 2.jpg

        ...

    - French horn

        - 1.jpg

        - 2.jpg

        ...

    ...

### `output`:

Contains output from `lrp.ipynb`. The output are original images with resulting heatmaps beside them, showing which parts of the images most likely caught the model's attention when it makes a prediction.

### `src`:

Contains source files from [this](https://github.com/kaifishr/PyTorchRelevancePropagation) GitHub repository. These files are essential for carrying out LRP in `lrp.ipynb`.
