load('./R_src/data/example_raw_count.rda')
load('./R_src/data/AllGeneSet.rda')
source("./R_src/R/CUS.MCMC.R")
source("./R_src/R/Count2CUS.R")

load('./R_src/data/model1_features.rda')    




selectfea.M1 <- function(CUS, smp.name){
  #' Select features for Model1 and create input file and positional encoding information for Model1
  #' @param CUS CUS feature matrix
  #' @param smp.name Sample name of the CUS feature matrix.
  
  fea_M1 <- model1_features$pos1

  # feature position (.csv) for Model1
  write.csv(model1_features, 'immune_cancer_label_position.csv')

  # input csv for Model 1
  selectM1 <- as.data.frame(t(CUS[fea_M1, ]))
  #selectM1$celllabel <- 0  # Initially assuming all cells are immune cells in your data
  per <- sample(1:nrow(selectM1), nrow(selectM1))
  selectM1 <- selectM1[per, ]
  write.csv(selectM1, paste0('Model1_input_', smp.name, '.csv'))
  return(selectM1)
}   

func1 <- function(sp_name){
   
  example_raw_count <- readRDS(paste0('./input/', sp_name, '.rds'))
  testmtx <- as.matrix(example_raw_count)
  cus1 <- Count2CUS(testmtx, sam.name = sp_name, filter.LOW.DR = T, filter.UP.DR = T)

  return(cus1)
}
