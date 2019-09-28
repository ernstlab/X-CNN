# X-CNN
Fine-mapping interaction files for GM12878 and K562 are on peaks called by HiCCUPs (Rao et al. 2014), available at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525, and included in this repo as GSE63525_GM12878_HiCCUPS_looplist.txt and GSE63525_K562_HiCCUPS_looplist.txt. 

The format for both types of files is tab-delimited lines with 6 fields: chrA, startA, endA, chrB, startB, endB.

For example:
```
chr1    1580000 1585000 chr1    1645000 1650000
chr1    1710000 1720000 chr1    1830000 1840000
chr1    1890000 1895000 chr1    1965000 1970000
chr1    2130000 2135000 chr1    2315000 2320000
...
```

The fine-mapping files will have regions of 100bp, the fine-mapping resolution of X-CNN. Additionally, some of the fine-mappings will fall outside of the original HiCCUPs peak. This is expected if a peak is, e.g., incorrectly called at a neighboring peak.
