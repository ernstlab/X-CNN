# Running X-CNN

X-CNN (X here is the lowercase Greek letter 'chi') is a method for computationally fine-mapping chromatin interactions, e.g., Hi-C, using ChIP-seq and/or DNase data. The published manuscript can be accessed here: https://doi.org/10.1093/bioinformatics/btz843.

X-CNN requires Python 3 and several easy-to-install packages, including numpy, h5py, pandas, keras, and Integrated Gradients (included in repo). I have written a small database package called chip_db to handle the potentially large amounts of data, though it is not necessary to use if you're handy with Python. Since X-CNN needs to randomly access different parts of the genome and extract potentially on the order of 100 ChIP seq tracks, it can be quite cumbersome to keep all these tracks in memory. Chip_db overcomes this challenge by building a database using an hdf5 backend, allowing random access of regions of the genome. It's easy to use and somewhat flexible in how you handle your data. 

## Preparation

You will first need to either (1) create a database file using ChIP db or (2) create numpy matrices for training.

#### ChIP-seq track files
These are feature tracks to be used for training the model and fine-mapping. The format should be either bedgraph or wig files. To create the database, simply run

`python3 chip_db.py [cell_type] [data_res] [/path/to/hg19.chrom.sizes] [track1 track2 ...]`

The `cell_type` argument is simply the name of the cell or tissue type. The `data_res` argument is the resolution of the data when binning. We recommend using either 50 or 100. You will also need a file with the size of the chromosomes (e.g., http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes). This is simply a two-column text file, where the first column is a chromosome name, and the second column is the length of the chromosome. When running the above script, it may take a while, but once built, you can easily use it with X-CNN or for your own purposes.

The interaction peaks can all be of the same size (as in FitHiC) or different sizes (as in HiCCUPs). X-CNN will extend them all to the same size. Modifying the method to train on different size peaks is to be implemented at a later date.

## Running X-CNN

To run X-CNN, you will need to provide it either (1) a database file, explained above, or (2) numpy matrices representing ChIP / DNase data. Either way, you will need and a chromatin interaction peak file.

#### Database method

Running X-CNN using a database requires passing the `--database flag`. Here is an example:

```
python3 train_X-CNN.py \
GM12878 \  # Name of cell type, used for the database query
100 \  # This is the resolution desired. Cannot be finer resolution than database resolution
/path/to/interaction_file.txt \
--database /path/to/database/ChIP_db.hdf5 \
--two_random 1 \  # number of negative samples to create per positive sample
--chr_size /home/aku/3DP/hg19.chrom.sizes \
--autoencoder 26 \
--filter_len 8 \
--conv_kernel 16 \
--dense_kernel 16 \
--dense_dropout 0.25 \
--regularizer 0 \
--out_dir /home/aku/3DP/current_analyses/GM12878/HiCCUPs/genomic_bgd_25kb \
--intn_len 25000  # Size to extend peaks to
```

If you would like to train on more data, at the cost of not assessing classification performance, add the `--final` flag. This builds a "final" model, and can be used for optimizing fine-mapping performance.

X-CNN includes code specifically designed to handle large genomic data. The difficulty is that when many ChIP-seq datasets are available (say, 10s to 100s), accessing random positions in the genome becomes very difficult for un-indexed files. Instead of having to read in these individual files, we have built code to generate a database, called `chip_db`. It is built on top of HDF5, and implemented in Python. In order to build this database, you will need bedgraph wig files containing the ChIP-seq and DNase data. It builds a rather large file containing a binarized and indexed representation of this data, which can then be used alone using the API or with X-CNN to access all available data at any position in constant time, greatly speeding up computation overall. More information is available at https://github.com/ernstlab/chip_db.

#### Chromatin interaction peak file
This should be a tab-delimited text file with six columns and no header of the form: 
chromosome_A, start_A, end_A, chromosome_B, start_B, end_B

For example:

```
chr1    1580000 1585000 chr1    1645000 1650000
chr1    1710000 1720000 chr1    1830000 1840000
chr1    1890000 1895000 chr1    1965000 1970000
chr1    2130000 2135000 chr1    2315000 2320000
chr1    2345000 2350000 chr1    2480000 2485000
chr1    2350000 2375000 chr1    3325000 3350000
...
```

#### Numpy matrix method

Create two numpy matrices of size `(num_interactions, 2, num_tracks, length)`. The first represents positive interactions, the second is negative interactions. You do not need the same number of interactions in each. The `2` has to do with the two sides of the interaction, the first index being the left side of the interaction, the second being the right. X-CNN will automatically reverse the data as well to increase the size of the training data. We have included generated files for both K562 and GM12878, and are available for download in Google Drive at https://drive.google.com/drive/folders/1SlQHF6SSc-lp-jT7yjeriM7eAjX8NHPb?usp=sharing.

## Fine-mapping

Fine-mapping is straightforward. Simply call the `fine_map.py` script along with a trained model and data. For example:

```
python3 fine_map.py \
--model final_model.hdf5 \
--data GM12878:all:log:100bp.npy
```

This outputs five files:
1. A binary gradient file `gradients.npy` of the same shape as the input data `(num_interactions, 2, num_tracks, length)`
2. A binary importances file `importances.npy` of size `(num_interactions, 2, length)`
3. A tab-delimited importance file `importances.left.txt` with `num interactions` rows and `length` columns. These are the importances of the left interacting regions.
4. A tab-delimited importance file `importances.right.txt` with `num interactions` rows and `length` columns. These are the importances of the right interacting regions.
5. A tab-delimited file of the fine-mapped positions `fine-mapping.txt` for each interacting region. These are the positions with the highest importance scores and X-CNN fine-mapped positions. These are in genomic coordinates.
