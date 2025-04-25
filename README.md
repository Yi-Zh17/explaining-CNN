# Explaining CNN Using Layer-Wise Relevance Propagation

## Jupyter notebooks:

### `analysis_confidence_score_original`:

Loads "vgg19_results.csv" and picks out images with the highest and lowest confidence scores given that the prediction is correct (applied on pre-trained model)

### `lrp`:

Performs layer-wise relevance propagation on selected images. Images are put into the `input` folder, and the resulting images are put in the `output` folder. Some code is adpated from [this](https://github.com/kaifishr/PyTorchRelevancePropagation) GitHub repository.

### `vgg19_evaluation_csv_imagenette`:

Evaluates the model on validation set in [Imagenette](https://github.com/fastai/imagenette), and store image paths, true label, predicted label, predicted confidence score, true confidence score, and correctness into a csv file.

### `data_exploration`:

Makes a bar plot of number of images contained in the training and validation set in Imagenette.

### `vgg19_evaluate_modified`:

Re-evaluates the confidence scores of modified images.



## CSV files:

### `vgg19_results.csv`:

Resulting csv file acquired from `vgg19_evaluation_csv_imagenette.ipynb`. Columns contain image paths, true label, predicted label, predicted confidence score, true confidence score, and correctness. (pre-trained vgg19 model)



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

### `selected_images`:

Contains images with high ($\geq$ 0.99) and low ($\leq$ 0.20) confidence scores. The heatmaps of those images are also stored in a separate folder.

### `modified_imgs`:

Contains modified images and their heatmaps.

### `further_modified_imgs`:

Contains further experimentation on modified images and other images. The heatmaps are also included.
