

#'
#' @param CUS CUS feature matrix containing all cells in the raw counts.
#' @param pred_file1 File path of output file of Model 1. String. The file should have at least two columns:
#' "X" is cell barcode, and "Predict" is the predicted label by Model 1.
#' The predicted CTCs are labeled by 1, while the predicted immune cells are labeled by 0.
#' @param smp.name Sample name of the data. String.
#' @param can.type The candidate cancer type of the test cells. Only integer.
#' In version 1.0.1, cancer types are composed of cervical cancer (0), non-small cell lung cancer (1), colorectal cancer (2),
#' pancreatic ductal adenocarcinoma (3), nasopharyngeal carcinoma (4), endometrium cancer (5), ovary cancer (6),
#' breast cancer (7), prostate cancer (8), gastric cancer (9), hepatocellular carcinoma (10), and melanoma (11).


fea.M2 <- function(CUS, pred_file1, smp.name){
  # All CUS are used for Model2
  # Subset predicted CTCs
  library(Matrix)
  m1_output <- read.csv(pred_file1, header = T)
  file1_cells <- readRDS(paste0('./input/', smp.name, '.rds'))
  file1_cells <- as.matrix(file1_cells)
  CTC_barcode <- m1_output$X[which(m1_output$Predict == 1)]
  
  if (length(CTC_barcode) > 0){
  	CUS_CTC <- CUS[, CTC_barcode, drop = FALSE]
  	
  	# input csv for Model 2
    M2_input <- as.data.frame(t(CUS_CTC))
    #M2_input$celllabel <- cancer.type
    print(paste0('There are ', dim(M2_input)[1], ' CTCs predicted in the PBMC!'))
    print(paste0('Total remaining cells in PBMC: ', nrow(m1_output)))
    print(paste0('CTC direction rate: ', dim(M2_input)[1], '/', dim(file1_cells)[2], '=', 
          round(100*dim(M2_input)[1]/dim(file1_cells)[2], 2), '%' ))
    print('----------------')
    print('Input file for model 2 is generated!')
    write.csv(M2_input, paste0('Model2_input_', smp.name, '.csv'))
  } else {
  	print('There are 0 CTCs predicted in the PBMC!')
  }
  
  return(CTC_barcode)
}  
  


func2 <- function(cus1, sp_name){
  pred1 = paste0('./predict_label_Model1_', sp_name, '.csv')
  M2_input <- fea.M2(
    CUS = cus1, 
    pred_file1 = pred1,
    smp.name = sp_name)
  return(M2_input)
}
