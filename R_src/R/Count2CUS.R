#' Convert aligned and smooth expression matrix to CUS feature matrix
#'
#' @param rawmat The counts matrix of scRNA data. It could be extract from a Seurat object by "SeuratObject@assays$RNA@counts". The example counts file could be example_raw_count.
#' @param AllGeneSet A data.frame of total 33575 genes aligned across the chromosomes. Default: AllGeneSet
#' @param sam.name
#' @param outpath
#' @param win.size
#' @param n.cores
#' @param distance
#' @param ngene.chr
#' @param LOW.DR
#' @param UP.DR
#' @param filter.LOW.DR
#' @param filter.UP.DR

library(tidyr)
library(dplyr)
library(purrr)
'%not_in%' <- purrr::negate(`%in%`)
library(Matrix)
library(copykat)


Count2CUS <- function(rawmat, all_gene = AllGeneSet, sam.name = "", outpath = "",
                      win.size = 10, distance = "euclidean", ngene.chr = 5,
                      LOW.DR = 0.01, UP.DR = 0.1, filter.LOW.DR = TRUE, filter.UP.DR = TRUE
                      ){

  start_time <- Sys.time()
  print(paste0('Start : ', start_time))
  set.seed(1234)


  print(">>> step1: read and filter data ...")
  print(paste(nrow(rawmat), " genes, ", ncol(rawmat), " cells in raw data", sep=""))
  genes.raw <- apply(rawmat, 2, function(x)(sum(x>0)))
  if(sum(genes.raw> 200)==0) stop("none cells have more than 200 genes")
  if(sum(genes.raw<100)>1){
    rawmat <- rawmat[, -which(genes.raw< 200)]
    print(paste("filtered out ", sum(genes.raw<=200), " cells with less than 200 genes; remaining ", ncol(rawmat), " cells", sep=""))
  }
  der<- apply(rawmat,1,function(x)(sum(x>0)))/ncol(rawmat)
  if(sum(der>LOW.DR)>=1){
    rawmat <- rawmat[which(der > LOW.DR), ]; print(paste(nrow(rawmat)," genes past >LOW.DR filtering", sep=""))
  }
  WNS1 <- "data quality is ok"
  if(nrow(rawmat) < 7000){
    WNS1 <- "low data quality"
    UP.DR<- LOW.DR
    print("WARNING: low data quality; assigned LOW.DR to UP.DR...")
  }


  print(">>> step 2: annotations gene coordinates ...")
  anno.mat <- annotateGenes.hg20(mat = rawmat, ID.type = 'S') #SYMBOL or ENSEMBLE
  anno.mat <- anno.mat[order(as.numeric(anno.mat$abspos), decreasing = FALSE),]
  print(paste(nrow(anno.mat)," genes annotated", sep=""))
  print(paste0("ncol(anno.mat)=", ncol(anno.mat), ": including ", ncol(anno.mat)-7, " cells, ", "7 annotation-head"))

  if(filter.LOW.DR == TRUE){
    ### secondary filtering
    ToRemov2 <- NULL
    for(i in 8:ncol(anno.mat)){
      cell <- cbind(anno.mat$chromosome_name, anno.mat[,i])
      cell <- cell[cell[,2]!=0,]
      if(length(as.numeric(cell))< 5){
        rm <- colnames(anno.mat)[i]
        ToRemov2 <- c(ToRemov2, rm)
      } else if(length(rle(cell[,1])$length)<length(unique(anno.mat$chromosome_name)) | min(rle(cell[,1])$length)< ngene.chr){
        rm <- colnames(anno.mat)[i]
        ToRemov2 <- c(ToRemov2, rm)
      }
      i<- i+1
    }

    if(length(ToRemov2)==(ncol(anno.mat)-7)) stop("all cells are filtered")
    if(length(ToRemov2)>0){
      anno.mat <-anno.mat[, -which(colnames(anno.mat) %in% ToRemov2)]
    }
    print(paste("filtered out ", length(ToRemov2), " cells with less than ",ngene.chr, " genes per chr", " when >LOW.DR", sep=""))
    print(paste0("After filtering>LOW.DR : genes ", dim(anno.mat)[1], " cells ", dim(anno.mat)[2]-7 ))
  } 


  ########  Align genes
  print(">>> step 3: align genes to AllGeneSet ...")
  print(paste('Gene numbers of AllGeneSet : ', length(all_gene$abspos)))
  print(paste('Gene numbers of Sample : ', nrow(anno.mat)))
  share_genes <- intersect(anno.mat$abspos, all_gene$abspos)
  print(paste('Shared Gene numbers of Sample and AllGeneSet : ', length(share_genes)))

  anno.mat_M <- anno.mat[, 8:ncol(anno.mat)]
  rownames(anno.mat_M) <- anno.mat$abspos
  anno.mat_M <- subset(anno.mat_M, rownames(anno.mat_M) %in% share_genes) %>% as.matrix()
  dim(anno.mat_M)

  column_names <- colnames(anno.mat_M)
  row_names <- all_gene$abspos
  M0 <- matrix(0, ncol=length(column_names), nrow = length(row_names), dimnames = list(row_names, column_names))
  M0[rownames(anno.mat_M), colnames(anno.mat_M)] <- anno.mat_M
  anno.mat <- cbind(all_gene, M0)

  ####
  rawmat3 <- data.matrix(anno.mat[, 8:ncol(anno.mat)])
  rownames(rawmat3) <- anno.mat$hgnc_symbol
  norm.mat<- log(sqrt(rawmat3)+sqrt(rawmat3+1))
  norm.mat<- apply(norm.mat,2,function(x)(x <- x-mean(x)))
  colnames(norm.mat) <-  colnames(rawmat3)
  rownames(norm.mat) <-  rownames(rawmat3)
  print(paste("A total of ", ncol(norm.mat), " cells, and ", nrow(norm.mat), " genes after preprocessing", sep=""))



  print(">>> step 4: smoothing data with dlm ...")
  dlm.sm <- function(c){
    model <- dlm::dlmModPoly(order=1, dV=0.16, dW=0.001)
    x <- dlm::dlmSmooth(norm.mat[, c], model)$s
    x<- x[2:length(x)]
    x <- x-mean(x)
  }
  print("smoothing data ...") 
  if (length(grep("linux", R.version$os)) == 1){
    num_cores = parallel::detectCores()
    test.mc <- parallel::mclapply(1:ncol(norm.mat), dlm.sm, mc.cores = num_cores-1)
    } 
  else if(length(grep("mingw32", R.version$os)) == 1){
    num_cores = parallel::detectCores()
    cl <- parallel::makeCluster(getOption("cl.cores", num_cores-1))
    test.mc <- parallel::parLapply(cl, 1:ncol(norm.mat), dlm.sm)
    parallel::stopCluster(cl)
    }
  else{
  	num_cores = parallel::detectCores()
    test.mc <- parallel::mclapply(1:ncol(norm.mat), dlm.sm, mc.cores = num_cores-2)
  }
  norm.mat.smooth <- matrix(unlist(test.mc), ncol = ncol(norm.mat), byrow = FALSE)
  colnames(norm.mat.smooth) <- colnames(norm.mat)
  rownames(norm.mat.smooth) <-  rownames(norm.mat)

  if (filter.UP.DR == TRUE){
    ##### use a smaller set of genes to perform segmentation
    print("use a smaller set of genes to perform MCMC segmentation")
    ## Third filter cells
    DR2 <- apply(rawmat3, 1, function(x)(sum(x>0)))/ncol(rawmat3)
    anno.mat2 <- anno.mat[which(DR2>=UP.DR), ]
    print(paste0(dim(anno.mat2)[1], " genes passed >UP.DR filtering"))

    ToRemov3 <- NULL
    for(i in 8:ncol(anno.mat2)){
      cell <- cbind(anno.mat2$chromosome_name, anno.mat2[,i])
      cell <- cell[cell[,2]!=0,]
      if(length(as.numeric(cell))< 5){
        rm <- colnames(anno.mat2)[i]
        ToRemov3 <- c(ToRemov3, rm)
      } else if(length(rle(cell[,1])$length)<length(unique((anno.mat2$chromosome_name)))|min(rle(cell[,1])$length)< ngene.chr){
        rm <- colnames(anno.mat2)[i]
        ToRemov3 <- c(ToRemov3, rm)
      }
      i<- i+1
    }

    if(length(ToRemov3)==ncol(norm.mat.smooth)) stop ("all cells are filtered")
    if(length(ToRemov3)>0){
      norm.mat.smooth <-norm.mat.smooth[, -which(colnames(norm.mat.smooth) %in% ToRemov3)]
      print(paste("filtered out ", length(ToRemov3), " cells with less than ", ngene.chr, " genes per chr", " when >UP.DR", sep=""))
    }

    print(paste("final for segmentation: ", nrow(norm.mat.smooth), " genes; ", ncol(norm.mat.smooth), " cells", sep=""))
  } 


  print(">>> step 5: MCMC and segmentation ...")
  CUS <- CUS.MCMC(fttmat= norm.mat.smooth, bins = win.size)
  normmat_col <- colnames(norm.mat.smooth)
  rownames(CUS) <- paste0('breaks', 1:nrow(CUS))

  save(CUS, normmat_col, file = paste0(outpath, 'CUS_', sam.name, '_bin', win.size, '.Rdata'))
  end_time <- Sys.time()
  print(paste0('End : ', end_time))
  return(CUS)

}
