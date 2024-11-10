# REIMPLEMENTATION-ONEFORMER

## Download the COCO dataset from the website
Make sure you have `wget` and `unzip` installed. Use the following commands to download the required COCO dataset files.
Once they're installed, you need to create a directory to store the dataset.

```
mkdir -p datasets/coco
cd datasets/coco
```

Then, you should download the datasets from the COCO dataset website.
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
```
`train2017.zip` will contain COCO dataset images for training.

`val2017.zip` will contain COCO dataset images for validation.

`annotations_trainval2017.zip` will contain instance masks for training and validation.

`panoptic_annotations_trainval2017.zip` will contain panoptic masks for training and validation.

---

You can unzip the files using `unzip`
```
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip
```

---
## File Info / Structure
Then you will see all the files downloaded:

`train2017` from `train2017.zip`

`val2017` from `val2017.zip`

`panoptic_train2017` from `panoptic_annotations_trainval2017.zip` (You might have to unzip another folder in this zip file).

`panoptic_val2017` from `panoptic_annotations_trainval2017.zip`

`instances_train2017.json` from `annotations_trainval2017.zip`

`instances_val2017.json` from `annotations_trainval2017.zip`

`panoptic_train2017.json` from `panoptic_annotations_trainval2017.zip`

`panoptic_val2017.json` from `panoptic_annotations_trainval2017.zip`

---

Make sure your directory has a structure like this:

![Screenshot 2024-11-08 at 9 36 51 PM](https://github.com/user-attachments/assets/18990259-cfdf-4f3f-828d-281d00d9673b)

---
## Clone the repository
Now, you can clone the repository:

```
git clone
```

---
## Helper Function Info
You will see the following Python files:

The `backbone.py` file defines a `BackboneWithMultiScaleFeatures` class that extracts multi-scale feature maps from different layers of a pre-trained ResNet-50 model.

The `coco_dataset.py` file defines a `COCOPanopticDataset` class for loading images and panoptic segmentation masks from the COCO dataset, applying optional transformations, and handling missing masks with placeholders if necessary.

The `compute_loss.py` file defines the `SetCriterion` class, which calculates the loss for a model by comparing predicted outputs with ground truth targets.

The `contrastive_loss.py` file implements the `ContrastiveLoss` class, which calculates the contrastive loss between two sets of embeddings by computing a similarity matrix and applying cross-entropy loss.

The `hungarian_matcher.py` file defines the `HungarianMatcher` class, which performs Hungarian matching to optimally match predicted outputs with target labels and masks based on classification, mask, and dice similarity costs.

The `mlp.py` file defines the `TaskMLP` class, which implements a multi-layer perceptron and is used to transform input sequences by flattening, processing through fully connected layers with ReLU activations, and reshaping them.

The `pixeldecoder.py` file defines the `PixelDecoder` class, a module that refines and merges multi-scale feature maps from a backbone network.

The `predict.py` file defines the `MaskClassPredictor` class, which generates both class predictions and mask predictions for each query.

The `query_formulation.py` file defines the `TaskConditionedQueryFormulator` class, which creates task-conditioned query embeddings by combining task embeddings with a set of learned query embeddings.

The `text_mapper.py` file contains the `TextMapper` class, which generates task-specific embeddings for panoptic, instance, and semantic tasks by mapping them into a common feature space.

The `tokenizer.py` file defines the `TaskTokenizer` class, which provides a basic tokenizer and embedding layer for task-specific text.

The `transformer_decoder.py` file defines the `TransformerDecoder` class, which processes task-specific query embeddings through multi-scale attention stages and feed-forward layers.

`prediction_test.ipynb` puts all the helper functions together and outputs results.

---

Now, you can run `prediction_test.ipynb` file and see the results.
