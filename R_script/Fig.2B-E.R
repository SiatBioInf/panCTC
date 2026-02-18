
library(Seurat)
library(dplyr)
library(patchwork)
library(tidyr)
library(Nebulosa)
library(DoubletFinder)
library(RColorBrewer)
library(purrr)
'%not_in%' <- purrr::negate(`%in%`)
library(ggpubr)
library(reshape2)
library(gridExtra)
library(ggplot2)
library(ggrepel)
library(ggsci)
library(forcats)
library(infercnv)
library(infercnvNGCHM)
library(colorspace)
library(paletteer)


#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#
# Figure 2B
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#



pbmc_obj  <- readRDS("R_data/pbmc_obj_list_HAVE-CTCs.rds")

#### This dataset could be downloaded from 
#### https://doi.org/10.5281/zenodo.18672523
#### 

# ------------------------------------------- #
# Combine all Cancer types
# ------------------------------------------- #
obj <- merge(pbmc_obj[[1]], c(pbmc_obj[[2]], pbmc_obj[[3]], pbmc_obj[[4]],  pbmc_obj[[5]], 
                              pbmc_obj[[6]], pbmc_obj[[7]], pbmc_obj[[8]], pbmc_obj[[9]], pbmc_obj[[10]]))

table(obj@meta.data$CancerType)
table(obj@meta.data$orig.ident)




obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 2000)
obj <- ScaleData(obj)
obj <- RunPCA(obj)
library(harmony)
obj <- RunHarmony(obj, group.by.vars = 'orig.ident')
obj <- FindNeighbors(obj)
obj <- FindClusters(obj, algorithm = 1, resolution = c(0.5))
obj <- RunUMAP(obj, dims = 1:20, reduction = 'pca')

# ------------------------------------------- #
# Draw the dot plot
# ------------------------------------------- #

tm1 <- theme_grey(base_size = 13)+
  theme(panel.background = element_rect(fill = "white", colour = NA), 
        panel.border = element_rect(fill = NA, colour = "black", linewidth = 1), 
        panel.grid = element_line(colour = "grey87"), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(), 
        #axis.ticks = element_line(colour = "black", linewidth = rel(0.5)), 
        axis.ticks = element_blank(),
        axis.text = element_blank(),
        legend.key = element_rect(fill = "white", colour = NA), 
        legend.text = element_text(size = 13),
        plot.title = element_blank(),
        strip.background = element_rect(fill = "grey70", colour = NA), 
        strip.text = element_text(colour = "white", size = rel(0.8), margin = margin(0.8, 0.8, 0.8, 0.8)), 
        complete = TRUE#, legend.position = 'none'
  )
library(paletteer)
col_column <- paletteer_d("ggthemes::Tableau_10")


p5 <- DimPlot(obj, reduction = 'umap',
              group.by = 'CancerType', 
              label = F, raster = T,
              #cols = c(brewer.pal(12,"Paired")[c(10,7,12,3,8,5,4,2,11)], 'black'),
              cols = c(rev(col_column), 'black'),
              pt.size = 0.3, order = (c('CTC', sort(unique(obj@meta.data$CancerType))[-1])), shuffle = F) +
  tm1 ; p5
ggsave('DimPlotMerged_Pred_CTCs_publicPBMC_cancerType-HAVE-CTC-NO-HARMINY.pdf'), path ='R_figure',
       width = 7.14, height = 3.96, units = 'in')
       
       
       
# ------------------------------------------- #
# Rate of predicted CTCs
# ------------------------------------------- #    

df <- read.csv('R_data/Proportion_of_CTCs_All.csv')
dfdf <- data.frame(Smp = unique(df$PBMC), 
                   n_CTC = df$Freq[which(df$Var1 == 'CTC')], 
                   n_Immune = df$Freq[which(df$Var1 == 'Immune')])
dfdf$Rate <- 100*(dfdf$n_CTC / dfdf$n_Immune)
#dfdf$Smp <- stringr::str_remove_all(dfdf$Smp, 'PBMC_')
dfdf <- dfdf[which(dfdf$Smp != 'PBMC_Experiment_Tonsil'), ]

dfdf$Smp[which(dfdf$Smp == 'PBMC_Experiment_Cervix')] <- 'PBMC_CC_(new-seq)'
dfdf$Smp <- stringr::str_replace_all(dfdf$Smp, '_', ' ')
dfdf$Smp <- factor(dfdf$Smp, levels = rev(sort(unique(dfdf$Smp))))
ggplot(dfdf, aes(x = Smp, y = Rate,  fill = Smp)) +
  geom_bar(width = .8, stat = 'identity', color = 'black', linewidth = 0.1#,
           #fill = paletteer_d("ggsci::schwifty_rickandmorty")[9]
           ) +
  geom_text(aes(y = Rate, label = paste0(n_CTC, ' / ', n_Immune, ' (', round(n_CTC/n_Immune, digits = 5)*100, '%)')), 
            position = position_fill(vjust = 1.7), size = 3.3, color = 'black', angle = 90)+
  labs( y = 'CTC detection rate (%)') +
  scale_fill_manual(values = rev(col_column)) +
  #coord_flip()+
  theme_classic(base_size = 13)+
  theme(axis.text.x = element_text(size =13, angle = 90, vjust = 0.5, hjust = 0, color = 'black'),
        axis.text.y = element_text(size =13, color = 'black', angle = 90, vjust = 0.5, hjust = 0.5),
        axis.title.x = element_blank(),
        axis.line = element_line(colour = 'black', linewidth = 0.3),
        legend.position = 'none')
ggsave('Proportion_of_CTC_plot_NEW.pdf', path = 'R_figure',
       width = 3.35, height = 4.5, units = 'in')









#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#
# Figure 2C
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#


# -------------------------------------------- #
# VlnPlot for significant markers
# -------------------------------------------- #

pbmc_obj  <- readRDS("R_data/pbmc_obj_list_HAVE-CTCs.rds")


names(pbmc_obj)
#ns_mks <- list(c('TUBB1', 'SRSF6', 'JUN'), # HCC
#               c(), # PDAC
#               c(), # NSCLC
#               c(), # CRC
#               c('TGFB1', 'ACTN1', 'MMP9'), # Experimental CC
#               c(), # NPC
#               c('ZEB2', 'ACTN1', 'FOS'), # Melanoma
#               c('ACTN1', 'PPBP', 'ZEB2'), # OVC
#               c('CD44', 'CXCR4', 'BTG2'), # GC
#               c('TUBB1', 'TGFB1', 'ERBB3')  # BC
#               )
#
ns_mks <- list(c(), # HCC  # 'TXN', 'SRSF6', 'TUBB1'
               c('EPCAM', 'KRT7', 'SOX4'), ### PDAC
               c(), # NSCLC
               c(), # CRC  # 'CXCR4', 'DUSP2', 'RGS1'
               c('TGFB1', 'ACTN1', 'MMP9'), # Experimental CC
               c(), # NPC  # 'JUN', 'GATA3', 'CXCR6'
               c('ZEB2', 'FOS', 'PPBP'), ### Melanoma
               c('VEGFA', 'ACTN1', 'ZEB2'), ### OVC
               c('CXCR4', 'CD44', 'BTG2'), ### GC
               c('EPCAM', 'KRT8', 'CLDN5')  ### BC
)
names(ns_mks) <- data_name2

tm1 <- theme_classic(base_size = 15)+
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        #axis.text.x = element_text(size = 11, colour = 'black', angle = 45, hjust = 0.7, vjust = 0.9),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 15, colour = 'black'),
        legend.position = 'none'
        )

pm <- list()
for (m in c(2,5,7:10)){
  pm[[m]] <- list()
  if (length(ns_mks[[m]]) > 0){
    obj <- pbmc_obj[[m]]
    obj <- SetIdent(obj, value = 'Predict')
    for (j in 1:length(ns_mks[[m]])){
      pm[[m]][[j]] <- VlnPlot(obj, features = ns_mks[[m]][j], pt.size = 0, #cols = paletteer_d("vapoRwave::mallSoft")[c(2, 6)])
                              cols = paletteer_d("ggsci::schwifty_rickandmorty")[c(12,9)])+
        tm1
    }
  }
  names(pm[[m]]) <- ns_mks[[m]]
}
names(pm) <- data_name2

#sort(data_name2[c(1,5,7,8,9,10)])
data_name2[c(5,2,10,
             7,8,9)]
library(ggpubr)
ggarrange(pm[[5]][[1]], pm[[5]][[2]], pm[[5]][[3]],
          pm[[2]][[1]], pm[[2]][[2]], pm[[2]][[3]],
          pm[[10]][[1]], pm[[10]][[2]], pm[[10]][[3]], 
          pm[[7]][[1]], pm[[7]][[2]], pm[[7]][[3]],
          pm[[8]][[1]], pm[[8]][[2]], pm[[8]][[3]], 
          pm[[9]][[1]], pm[[9]][[2]], pm[[9]][[3]], 
          
          ncol=9, nrow=2, align = "hv",
          widths = c(rep(1,9)), heights = c(1,1))
ggsave('VlnPlot_for_non-significant-markers.pdf', path = 'R_figure/', 
       width = 13.5, height = 3.6, units = 'in')








#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#
# Figure 2D
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#



obj_ctc <- list()
for (m in 1:10){
  obj <- pbmc_obj[[m]]  # "PBMC_GSE174463_BC"
  obj_ctc[[m]] <- subset(obj, celltype2 == 'CTC')
  obj_ctc[[m]][['Dataset']] <- names(pbmc_obj)[m]
  print(names(pbmc_obj)[m])
  print(dim(obj_ctc[[m]]))
}

obj_ctcObj <- merge(obj_ctc[[1]], 
                    c(obj_ctc[[2]], obj_ctc[[3]], obj_ctc[[4]], 
                      obj_ctc[[5]], obj_ctc[[6]], obj_ctc[[7]], 
                      obj_ctc[[8]], obj_ctc[[9]], obj_ctc[[10]]))
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_Experiment_Cervix', 'CTC (CC new-seq)')
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_GSE107747_HCC', 'CTC (GSE107747 HCC)')
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_GSE127465_NSCLC', 'CTC (GSE127465 NSCLC)')
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_GSE155698_PDAC', 'CTC (GSE155698 PDAC)')
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_GSE162025_NPC', 'CTC (GSE162025 NPC)')
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_GSE174463_BC', 'CTC (GSE174463 BC)')
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_GSE178318_CRC', 'CTC (GSE178318 CRC)')
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_GSE189125_Melanoma', 'CTC (GSE189125 Melanoma)')
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_GSE213243_OVC', 'CTC (GSE213243 OVC)')
obj_ctcObj@meta.data$Dataset <- stringr::str_replace_all(obj_ctcObj@meta.data$Dataset, 'PBMC_OMIX001073_GC', 'CTC (OMIX001073 GC)')
table(obj_ctcObj@meta.data$Dataset)

obj_ctcObj <- NormalizeData(obj_ctcObj)
obj_ctcObj <- FindVariableFeatures(obj_ctcObj, selection.method = "vst", nfeatures = 2000)
obj_ctcObj <- ScaleData(obj_ctcObj)
obj_ctcObj <- RunPCA(obj_ctcObj)
library(harmony)
obj_ctcObj <- RunHarmony(obj_ctcObj, group.by.vars = 'Dataset')
obj_ctcObj <- FindNeighbors(obj_ctcObj)
obj_ctcObj <- FindClusters(obj_ctcObj, algorithm = 1, resolution = c(0.1, 0.3, 0.5))
obj_ctcObj <- RunUMAP(obj_ctcObj, reduction = "pca", dims = 1:20)

DimPlot(obj_ctcObj, group.by = c('Dataset'), reduction = 'umap', label = F,
               cols = paletteer_d("ggthemes::Classic_10_Medium")) + 
  theme_bw(base_size = 12) + theme(axis.text = element_blank(),
                                   axis.ticks = element_blank(),
                                   panel.grid = element_blank(),
                                   panel.border = element_rect(linewidth = 1.3),
                                   plot.title = element_blank(),
                                   legend.position = 'right')
ggsave('DimPlot_All_CTCs_group_Legend.pdf', path = 'R_figure/', 
       width = 7.65, height = 2.9, units = 'in')




#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#
# Figure 2E
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#


obj_ctcObj@meta.data$CTC_Group <- 'EPCAM+ CTCs'
obj_ctcObj@meta.data$CTC_Group[which(obj_ctcObj@meta.data$RNA_snn_res.0.1 %in% c(0, 2:8))] <- 'EPCAM- CTCs'
#saveRDS(obj_ctcObj, 'exp_data/results0625/cellchat_output_20240421/CTC_posi_neg.rds')

obj_ctcObj@meta.data$CTC_Group <- factor(obj_ctcObj@meta.data$CTC_Group, levels = c('EPCAM+ CTCs', 'EPCAM- CTCs'))
ppp <- DotPlot(obj_ctcObj, cols = c("lightgrey", '#F15854FF'), group.by = 'CTC_Group',
        features = fts) + RotatedAxis()
ppp + coord_flip()+ theme_bw(base_size = 12) + theme(axis.title = element_blank(),
                         axis.text = element_text(colour = 'black', size = 11),
                         axis.text.x = element_text(angle = 20, hjust = 1, vjust = 1), 
                         #axis.text.y = element_text(angle = 90, hjust = 0.5, vjust = 0), 
                         legend.position = 'right')
ggsave('DotPlot_All_CTCs_group.pdf', path = 'R_figure'), 
       width = 3.5, height = 3.6, units = 'in')



obj_ctcObj[['new_cluster']] <- 'EPCAM+'
obj_ctcObj@meta.data$new_cluster[which(obj_ctcObj@meta.data$RNA_snn_res.0.1 %in% c(0,5))] <- 'EPCAM-c1'
obj_ctcObj@meta.data$new_cluster[which(obj_ctcObj@meta.data$RNA_snn_res.0.1 %in% c(2,4))] <- 'EPCAM-c2'
obj_ctcObj@meta.data$new_cluster[which(obj_ctcObj@meta.data$RNA_snn_res.0.1 %in% c(3))] <- 'EPCAM-c3'
obj_ctcObj@meta.data$new_cluster[which(obj_ctcObj@meta.data$RNA_snn_res.0.1 %in% c(8))] <- 'EPCAM-c4'
obj_ctcObj@meta.data$new_cluster[which(obj_ctcObj@meta.data$RNA_snn_res.0.1 %in% c(6,7))] <- 'EPCAM-c5'
obj_ctcObj <- SetIdent(obj_ctcObj, value = 'new_cluster')
mks <- FindAllMarkers(obj_ctcObj, only.pos = T, logfc.threshold = 1)
top20 <- mks %>% group_by(cluster) %>% top_n(n=20, wt = avg_log2FC)

obj_ctcObj[['new_cluster_MK']] <- 'EPCAM+'
obj_ctcObj@meta.data$new_cluster_MK[which(obj_ctcObj@meta.data$new_cluster == 'EPCAM-c1')] <- 'IFITM1+'
obj_ctcObj@meta.data$new_cluster_MK[which(obj_ctcObj@meta.data$new_cluster == 'EPCAM-c2')] <- 'S100A9+'
obj_ctcObj@meta.data$new_cluster_MK[which(obj_ctcObj@meta.data$new_cluster == 'EPCAM-c3')] <- 'C1QA+'
obj_ctcObj@meta.data$new_cluster_MK[which(obj_ctcObj@meta.data$new_cluster == 'EPCAM-c4')] <- 'PPBP+'
obj_ctcObj@meta.data$new_cluster_MK[which(obj_ctcObj@meta.data$new_cluster == 'EPCAM-c5')] <- 'COL1A2+'


DimPlot(obj_ctcObj, group.by = c('new_cluster_MK'), reduction = 'umap', label = T,
               cols = paletteer_d("ggthemes::few_Medium")[c(2:4, 6:7,5)]) + 
  theme_bw(base_size = 12) + theme(axis.text = element_blank(),
                                   axis.ticks = element_blank(),
                                   panel.grid = element_blank(),
                                   panel.border = element_rect(linewidth = 1.3),
                                   plot.title = element_blank(),
                                   legend.position = 'none')
ggsave('DimPlot_All_CTCs_subcluster_markers.pdf', path = 'R_figure/', 
       width = 7.65, height = 2.9, units = 'in')


