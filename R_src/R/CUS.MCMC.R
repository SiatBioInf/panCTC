#' Apply MCMC to aligned and smooth expression matrix
#'
#' @param fttmat aligned and smooth expression matrix using 33575 genes in AllGeneSet.
#' @param bins window size for segment. Default = 10.


CUS.MCMC <- function(fttmat, bins){
  norm.mat.sm <- fttmat
  for (m in 1:ncol(norm.mat.sm)){
    norm.mat.sm[, m] <- 0.01*(norm.mat.sm[, m] - min(norm.mat.sm[, m]))/ (max(norm.mat.sm[, m]) - min(norm.mat.sm[, m]))
  }
  n <- nrow(norm.mat.sm)
  BR <- NULL
  breks <- c(seq(1, as.integer(n/bins-1)*bins, bins), n)
  CON = matrix(0, nrow = length(breks)-1, ncol = ncol(norm.mat.sm))
  CON <- data.frame(CON)
  for(c in 1:ncol(norm.mat.sm)){
    qu1=summary(norm.mat.sm[,c])[2]  # baseline of cell c
    qu1_seq=rep(qu1, bins)
    a2 <-  mean(qu1_seq)
    set.seed(20221010)
    posterior2 <-MCMCpack::MCpoissongamma(qu1_seq, a2, 1, mc=1000)

    bre <- NULL
    for (i in 1:(length(breks)-1)){
      cell=norm.mat.sm[breks[i]:breks[i+1],c]   # window i of cell c
      a1 <-  max(mean(cell), mean(qu1_seq))
      set.seed(20221010)
      posterior1 <-MCMCpack::MCpoissongamma(cell, a1, 1, mc=1000)
      t_test_g=t.test(posterior1, posterior2, alternative = 'g')
      CON[i,c]=round(-log10(t_test_g$p.value),2)
    }#for(i)
  }#for(c)
  colnames(CON) <- colnames(norm.mat.sm)
  print(dim(CON))
  return(CON)
}
