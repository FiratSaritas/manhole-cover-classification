## Description of Data
------------

### `labels.csv` 

This file shows all data when it was labelled.

- `image`: Name of the image in the original image (PK)
- `type`: Labelled type of the image
- `subtype`: Labelled subtype of the image


### `transformation_split.csv`

This is a pre-processed dataframe which originates from `labels.csv`. This file is used for the Train-Validation-Test Splits of the data plus the offline data augmentation steps.

- `image`: Name of the original image
- `transforms`: Applied transformations
- `label`: Concatenation of type and subtype from original dataframe
- `set`: Indicator for Train-Val-Testset
- `filename`: Filename after offline augmentation


