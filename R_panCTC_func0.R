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
  per <- sample(1:nrow(selectM1), nrow(selectM1))
  selectM1 <- selectM1[per, ]
  write.csv(selectM1, paste0('Model1_input_', smp.name, '.csv'))
  return(selectM1)
}   



func0 <- function(cus1, sp_name){
  select_M1 <- selectfea.M1(cus1, smp.name = sp_name)
}