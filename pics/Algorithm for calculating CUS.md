# Algorithm for calculating CUS values based on gene expression
                                                                               
## INPUT
1. `rawmat`: The counts matrix of scRNA data (genes*cells)
2. `AllGeneSet`: A data.frame of total 33575 genes aligned across the chromosomes
3. `sam.name`: sample names
4. `outpath`: output pathway of results
5. `win.size`: window width of each segment
6. `ngene.chr`: minimum number of genes on a chromosome
7. `LOW.DR`: lower bound of the gene expression ratio
8. `UP.DR`: adjusted lower bound of the gene expression ratio
9. `filter.LOW.DR`: indicator of the second round filtering
10. `filter.UP.DR`: indicator of the third round filtering

**Step 1**: read and filter data
1: filter the cells in rawmat with less than 200 genes
2: retain genes in rawmat whose expression ratio exceeds `LOW.DR`
3: when the number of remaining genes is less than 7,000, adjust the ratio `LOW.DR` to `UP.DR`

**step 2**: annotations gene coordinates
1: annotate the genes
2: second round filter of cells, and retain cells with the number of genes per chromosome > 5

**step 3**: align genes to `AllGeneSet`
1: align genes of different samples to all genes located on chromosomes in order
2: fill 0s as the expression of genes that do not exist in the sample after aligned to the genome

**step 4**: smoothing data with DLM algorithm
1: smooth the gene expressions in a single cell with dynamic linear model (DLM)
2: retain genes whose expression ratio exceeds `UP.DR`
3: apply a third-round filter to cells and retain those cells with more than 5 genes per chromosome

**step 5**: MCMC and segmentation
1: scale the gene expressions of each cell
2: for each cell 
2.1: set the baseline as the vector of length `win.size` with elements initialized by the 25th percentile of all gene expressions in the cell
2.2: set the baseline level as the posterior distributions (posterior2) from the Poisson-Gamma model
2.3: for each segment of the cell
2.3.1: set the segment level as the posterior distributions (posterior1) from the Poisson-Gamma model of all gene expressions in the segment
2.3.2: set the CUS value as the –log10(p-value) of the one side t-test between posterior1 and posterior2
     End for of segments
End for of cells


## OUTPUT 
the CUS matrix (CUS*cells)                                                 





