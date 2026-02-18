
library(Seurat)
library(dplyr)
library(patchwork)
library(tidyr)
library(Nebulosa)
#library(DoubletFinder)
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
#library(infercnv)
#library(infercnvNGCHM)
library(plyr)
library(colorspace)
library(ComplexHeatmap)
library(paletteer)

# -------------------------------------------- #
# 1. Upload PBMC object
# -------------------------------------------- #

library(CellChat)
options(stringsAsFactors = FALSE)
library(NMF)
library(ggalluvial)
library(uwot)


out_fd <- 'R_figure/cellchat'
dir.create(out_fd)

cellchat_out <- paste0(out_fd, 'cellchat_output/')
dir.create(cellchat_out)

data_name2 <- c('GSE107747_HCC', 
                'GSE155698_PDAC', 
                'GSE127465_NSCLC', 
                'GSE178318_CRC',
                'Experimental_CC', #
                'GSE162025_NPC',
                'GSE189125_Melanoma',
                'GSE213243_OVC',
                'OMIX001073_GC',
                'GSE174463_BC'#,  #'Experiment_Tonsil'
) 



pbmc_obj <- readRDS('R_data/pbmc_obj_list_HAVE-CTCs.rds')

## ------------------------------------ ##
## 1. Save the cellchat objects         ##
## ------------------------------------ ##

for (m in 1:length(pbmc_obj)){
  obj <- pbmc_obj[[m]]
  ####
  #### Subset n cells for each cell type
  ####
  n <- 500
  vc <- unique(obj@meta.data$celltype2)
  smp_c <- list()
  length(smp_c) <- length(vc)
  names(smp_c) <- vc
  for (i in 1:length(vc)){
    od <- which(obj@meta.data$celltype2 == vc[i])
    if (length(od) > n){
      set.seed(2023)
      od <- sample(od, n)
    } else {
      od <- od
    }
    smp_c[[i]] <- obj@meta.data$cells[od]
  }
  sub_obj <- subset(obj, cells %in% unlist(smp_c))
  table(sub_obj@meta.data$celltype2)
  
  
  ####
  #### cellchat
  ####
  sub_obj <- SetIdent(sub_obj, value = "celltype2")
  data.input <- GetAssayData(sub_obj, assay = "RNA", slot = "data")
  labels <- Idents(sub_obj)
  meta <- data.frame(group = labels, row.names = names(labels))
  
  ## 建立 CellChat 对象
  cellchat <- createCellChat(object = data.input, meta = meta, group.by = "group")
  levels(cellchat@idents)
  
  ## 设置配体受体互作数据库
  CellChatDB <- CellChatDB.human # use CellChatDB.mouse if running on mouse data
  showDatabaseCategory(CellChatDB)
  dplyr::glimpse(CellChatDB$interaction)
  cellchat@DB <- CellChatDB
  
  ## 预处理基因表达数据，用于细胞间互作分析
  cellchat <- subsetData(cellchat) # This step is necessary even if using the whole database
  future::plan("multiprocess", workers = 4) # do parallel
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  # project gene expression data onto PPI network (optional)
  # cellchat <- projectData(cellchat, PPI.human)
  
  ## 计算互作概率，推断细胞通讯网络
  cellchat <- computeCommunProb(cellchat)
  saveRDS(cellchat, paste0(cellchat_out, 'object_cellchat_', data_name2[m], '.rds'))
}


## ------------------------------------ ##
## 2. Overview, draw and save the plots ##
## ------------------------------------ ##

df_out <- NULL  # source CTC, from CTCs to all the other cell types
df_in <- NULL  # target CTC, from all the other cell types to CTCs
for (m in 1:10){
  ct <- readRDS(paste0(cellchat_out, 'object_cellchat_', data_name2[m], '.rds'))
  ct <- filterCommunication(ct, min.cells = 3)  # 过滤掉细胞群中仅有少量细胞的细胞间关系
  ## 提取推算的细胞间通讯网络为数据框架
  #df.net <- subsetCommunication(ct)
  #df.net <- subsetCommunication(ct, sources.use = c(1), targets.use = c(2:10))
  #nrow(df.net)
  #df.net2 <- subsetCommunication(ct, sources.use = c(2:10), targets.use = c(1))
  #nrow(df.net2)
  #df.net <- subsetCommunication(ct, signaling = c("WNT", "TGFb"))
  ## 在信号通路水平上，推算细胞间相互通讯
  ct <- computeCommunProbPathway(ct)
  ## 计算整合的细胞间通讯网络
  ct <- aggregateNet(ct)
  pathways.all <- ct@netP$pathways
  levels(ct@idents)
  lg <- length(levels(ct@idents))
  
  groupSize <- as.numeric(table(ct@idents))
  #netVisual_circle(ct@net$weight, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Interaction weights/strength")
  group.cellType <- c('CTC', rep('Immune', lg-1)) # grouping cell clusters into fibroblast, DC and TC cells
  names(group.cellType) <- levels(ct@idents)
  p0 <- netVisual_chord_cell(ct, signaling = pathways.all, group = group.cellType, title.name = data_name2[m])
  
  pdf(paste0(cellchat_out, 'chord_plot_',  data_name2[m], '.pdf'))
  print(p0)
  dev.off()

  #pairLR.all <- extractEnrichedLR(ct, signaling = pathways.all, geneLR.return = FALSE)
  #contb <- netAnalysis_contribution(ct, signaling = pathways.all)#$data
  #contb <- contb[with(contb, order(-contribution)), ]
  ## 只看前 25% 的配体受体对
  #top0.25pair <- contb$data %>% top_n(n = 0.25*nrow(pairLR.all), wt = contribution)
  
  p_out <- netVisual_bubble(ct, sources.use = c('CTC'), targets.use = setdiff(levels(ct@idents), 'CTC'), 
                            remove.isolate = T, #pairLR.use = data.frame(interaction_name = rownames(contb))
                            title.name = data_name2[m])
  p_out
  ggsave(paste0('Source-from-CTC_dot_plot_', data_name2[m], '.pdf'), plot = p_out, path = paste0(cellchat_out), width = 6.6, height = 6, units = 'in')
  
  p_in <- netVisual_bubble(ct, targets.use = c('CTC'), sources.use = setdiff(levels(ct@idents), 'CTC'), 
                           remove.isolate = T,
                           title.name = data_name2[m])
  p_in
  ggsave(paste0('Target-to-CTC_dot_plot_', data_name2[m], '.pdf'), plot = p_in, path = paste0(cellchat_out), width = 6.6, height = 6, units = 'in')
  
  ## Save the data.frame for drawing plot
  df_out0 <- p_out$data
  df_out0$sample <- data_name2[m]
  df_out <- rbind(df_out, df_out0)
  df_in0 <- p_in$data
  df_in0$sample <- data_name2[m]
  df_in <- rbind(df_in, df_in0)
  
  ct <- netAnalysis_computeCentrality(ct, slot.name = "netP")
  #netAnalysis_signalingRole_network(ct, signaling = pathways.all[2], width = 8, height = 2.5, font.size = 10)
    ## 各类细胞发出和接收的信号通路
  ht1 <- netAnalysis_signalingRole_heatmap(ct, pattern = "outgoing", width = 4, height = 11)
  ht2 <- netAnalysis_signalingRole_heatmap(ct, pattern = "incoming", width = 4, height = 11)
  
  pdf(paste0(cellchat_out, 'heatmap_plot_', data_name2[m], '.pdf'))
  print(ht1 + ht2)
  dev.off()
  
  ## 河流图：发出和接收的通讯模式
  #library(NMF)
  #library(ggalluvial)
  #selectK(ct, pattern = "outgoing")
  #nPatterns = 6
  #ct <- identifyCommunicationPatterns(ct, pattern = "outgoing", k = nPatterns)
  #netAnalysis_river(ct, pattern = "outgoing")
  #netAnalysis_dot(ct, pattern = "outgoing")
  
  #selectK(ct, pattern = "incoming")
  #nPatterns = 4
  #ct <- identifyCommunicationPatterns(ct, pattern = "incoming", k = nPatterns)
  #netAnalysis_river(ct, pattern = "incoming") 
  #netAnalysis_dot(ct, pattern = "incoming")
}
write.csv(df_out, paste0(cellchat_out, 'Source-from-CTC_AllDataSet_dataframe.csv'))
write.csv(df_in, paste0(cellchat_out, 'Target-to-CTC_AllDataSet_dataframe.csv'))












## ------------------------------------ ##
## 3. Integrate the L-R among samples   ##
## ------------------------------------ ##

####
#### Source from CTC
####

df_out <- read.csv(paste0(cellchat_out, 'Source-from-CTC_AllDataSet_dataframe.csv'))
df_in <- read.csv(paste0(cellchat_out, 'Target-to-CTC_AllDataSet_dataframe.csv'))

values <- c(1, 2, 3)
names(values) <- c("p>0.05", "0.01<p<0.05", "p<0.01")
color_use <- rev(RColorBrewer::brewer.pal(11, 'Spectral'))
#color_use <- paletteer_dynamic("cartography::red.pal", 20)

define_plot <- function(tmp){
  px <- ggplot(tmp, aes(x = source.target, y = interaction_name_2,
                        #y = reorder(Description, Sample, FUN = function(x) 0-length(x)),
                        color = prob, size = pval)) + 
    geom_point(pch = 16) + # 
    scale_radius(range = c(min(tmp$pval), max(tmp$pval)), 
                 breaks = sort(unique(tmp$pval)), 
                 labels = names(values)[values %in% sort(unique(tmp$pval))], 
                 name = "p-value") +
    scale_colour_gradientn(colors = colorRampPalette(color_use)(99), 
                           na.value = "white", 
                           limits = c(quantile(tmp$prob, 0, na.rm = T),
                                      quantile(tmp$prob, 1, na.rm = T)),
                           breaks = c(quantile(tmp$prob, 0, na.rm = T), 
                                      quantile(tmp$prob, 1, na.rm = T)),
                           labels = c("min", "max")) + 
    guides(color = guide_colourbar(barwidth = 0.5, title = "Prob.", order = 1)) +
    labs(x=NULL, y=NULL, title = 'XXX') + 
    #geom_vline(xintercept = seq(0.5, length(unique(df0_use$Sample))+0.5, 1), color = 'gray50', linewidth = 0.02)+
    #geom_hline(yintercept = seq(0.5, length(unique(df0_use$Description))+0.5, 1), color = 'gray50', linewidth = 0.02)+
    #coord_flip()+
    theme(plot.title = element_blank(),
          panel.background = element_blank(),
          panel.border = element_rect(linewidth = 1, fill = NA, colour = 'black'),
          panel.grid = element_line(colour = 'gray85', linewidth = 0.2),
          strip.background = element_rect(linewidth = 1, fill = 'gray', colour = 'black'),
          strip.text = element_text(size = 0, color = 'black'),
          #legend.position = "right", 
          legend.title = element_text(size = 9), 
          legend.margin = margin(t = 0),
          axis.ticks = element_blank(), 
          axis.text.x = element_text(size = 9, color = 'black', angle = 90, hjust = 1, vjust = 0.5),
          axis.text.y = element_text(size = 9, color = 'black'))# + 
    #facet_grid(pathway_name~sample, scales = 'free', space = 'free')
  return(px)
}

tmp <- df_out
define_plot(tmp)+
  facet_grid(pathway_name~sample, scales = 'free', space = 'free')
# 14 * 20


## ------------------------------------ ##
## 4. Select L-R, and draw plots        ##
##  Source from CTC
## ------------------------------------ ##

sample_color <- data.frame(sample = sort(data_name2), col = as.character(paletteer_d("ggthemes::Tableau_10")))
anno_color <- data.frame(anno = sort(unique(df_out$annotation)), col = as.character(paletteer_d("vapoRwave::crystalPepsi")[c(1:3)]))

####
#### Specific for each cancer
####

## pathway_name
# CC: TGFb, THBS
# HCC: GALECTIN, RESISTIN, SELL, CXCL
# NSCLC: PECAM1, SELPLG
# PDAC: MK, LAMININ, GRN
# NPC: PARs
# BC: CXCL
# CRC: CD6
# OVC:
# GC: CD6, GZMA

LR_need0 <- unique(c('RETN - CAP1', 'PF4 - CXCR3', 'GZMA - F2R', 'GRN - SORT1'))
tmp0 <- df_out[which(df_out$interaction_name_2 %in% LR_need0 & df_out$target %in% c('T', 'NK', 'MKI67+cells', 'Mono', 'DC', 'Mega')), ]
p0 <- define_plot(tmp0)+
  guides(y = "none", y.sec = "axis")+
  facet_grid(annotation~sample, scales = 'free', space = 'free', switch = "y"); p0
g0 <- ggplot_gtable(ggplot_build(p0))
stripr <- which(grepl('strip-t', g0$layout$name))
fills <- sample_color$col[which(sample_color$sample %in% unique(tmp0$sample))]
k <- 1
for (i in stripr) {
  j <- which(grepl('rect', g0$grobs[[i]]$grobs[[1]]$childrenOrder))
  g0$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills[k]
  k <- k+1
}
stripr2 <- which(grepl('strip-l', g0$layout$name))
fills2 <- anno_color$col[which(anno_color$anno %in% unique(tmp0$annotation))]
k <- 1
for (i in stripr2) {
  j <- which(grepl('rect', g0$grobs[[i]]$grobs[[1]]$childrenOrder))
  g0$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills2[k]
  k <- k+1
}
grid.draw(g0)



path1 <- unique(c('TGFb', 'THBS', 'GALECTIN', 'RESISTIN', 'SELL', 'CXCL', 'PECAM1', 
                  'SELPLG', 'MK', 'LAMININ', 'GRN', 'PARs', 'CXCL', 'CD6', 'GZMA', 'MIF'))
LR_need <- unique(c('HLA-C - KIR2DS4', 'HLA-C - KIR2DL1', 'HLA-A - KIR3DL1', 'HLA-E - KLRK1', 
                    'HLA-E - KLRC1', 'HLA-E - CD94:NKG2A', 
                    'ANXA1 - FPR2', 'ITGB2 - ICAM1'))
df_sub1 <- df_out[which(df_out$pathway_name %in% path1 & df_out$target %in% c('T', 'NK', 'MKI67+cells', 'Mono', 'DC', 'Mega')), ]
df_sub1.add <- df_out[which(df_out$interaction_name_2 %in% LR_need), ]
tmp <- rbind(df_sub1, df_sub1.add)
p1 <- define_plot(tmp)+
  guides(y = "none", y.sec = "axis")+
  facet_grid(annotation~sample, scales = 'free', space = 'free', switch = "y"); p1
g1 <- ggplot_gtable(ggplot_build(p1))
stripr <- which(grepl('strip-t', g1$layout$name))
fills <- sample_color$col[which(sample_color$sample %in% unique(tmp$sample))]
k <- 1
for (i in stripr) {
  j <- which(grepl('rect', g1$grobs[[i]]$grobs[[1]]$childrenOrder))
  g1$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills[k]
  k <- k+1
}
stripr2 <- which(grepl('strip-l', g1$layout$name))
fills2 <- anno_color$col[which(anno_color$anno %in% unique(tmp$annotation))]
k <- 1
for (i in stripr2) {
  j <- which(grepl('rect', g1$grobs[[i]]$grobs[[1]]$childrenOrder))
  g1$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills2[k]
  k <- k+1
}
grid.draw(g1)
# 8.5*4.6


####
#### Shared for each cancer
####

## pathway_name
# ADGRE5, ANNEXIN, APP, CD99, CLEC, ITGB2, MHC-I, MHC-II, MIF

path2 <- unique(c('ADGRE5', 'ANNEXIN', 'APP', 'CD99', 'CLEC', 'ITGB2', 'MHC-I', 'MHC-II'))
LR_rm2 <- unique(c('HLA-DRB5 - CD4', 'HLA-DQB1 - CD4', 'HLA-DQA1 - CD4', 'HLA-DMB - CD4', 'HLA-DMA - CD4',
                   'HLA-E - CD8B', 'HLA-C - CD8B', 'HLA-B - CD8B', 'HLA-A - CD8B',
                   'HLA-C - KIR2DS4', 'HLA-C - KIR2DL1', 'HLA-A - KIR3DL1', 'HLA-E - KLRK1', 
                   'HLA-E - KLRC1', 'HLA-E - CD94:NKG2A', 
                   'ANXA1 - FPR2', 'ITGB2 - ICAM1'
                   ))
df_sub2 <- df_out[which(df_out$pathway_name %in% path2 & 
                          df_out$target %in% c('T', 'NK', 'MKI67+cells', 'Mono', 'DC', 'Mega') &
                          df_out$interaction_name_2 %not_in% LR_rm2), ]
tmp <- df_sub2
p2 <- define_plot(tmp)+
  guides(y = "none", y.sec = "axis")+
  facet_grid(annotation~sample, scales = 'free', space = 'free', switch = "y"); p2
g2 <- ggplot_gtable(ggplot_build(p2))

stripr1 <- which(grepl('strip-t', g2$layout$name))
fills1 <- sample_color$col[which(sample_color$sample %in% unique(tmp$sample))]
k <- 1
for (i in stripr1) {
  j <- which(grepl('rect', g2$grobs[[i]]$grobs[[1]]$childrenOrder))
  g2$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills1[k]
  k <- k+1
}
stripr2 <- which(grepl('strip-l', g2$layout$name))
fills2 <- anno_color$col[which(anno_color$anno %in% unique(tmp$annotation))]
k <- 1
for (i in stripr2) {
  j <- which(grepl('rect', g2$grobs[[i]]$grobs[[1]]$childrenOrder))
  g2$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills2[k]
  k <- k+1
}
grid.draw(g2)
# 8.4*4.4



## ------------------------------------ ##
## 4. Select L-R, and draw plots        ##
##  Target to CTC
## ------------------------------------ ##

####
#### specific interaction_name_2
####

#HLA-DQA2 - CD4, HLA-DOB - CD4, HLA-DOA - CD4,
#ITGB2 - ICAM2, JAM1 - (ITGAL+ITGB2), LCK - (CD8A+CD8B1), 
#HLA-F - CD8B, HLA-E - CD8B, HLA-C - CD8B, HLA-B - CD8B, HLA-A - CD8B, 
#ALCAM - CD6, PECAM1 - PECAM1, CD99 - PILRA, 
#COL4A4 - CD44, THBS1 - CD47,
#FN1 - CD44, COL6A3 - CD44, COL6A2 - CD44, COL6A1 - CD44, COL4A2 - CD44, COL4A1 - CD44, COL1A2 - CD44, COL1A1 - CD44,
#CD70 - CD27, PF4V1 - CXCR3, PF4 - CXCR3, 
#PPBP - CXCR2, ANXA1 - FPR1, TNFSF12 - TNFRSF12A, MDK - NCL, IL16 - CD4

tgt_LR1 <- c('HLA-DQA2 - CD4', 'HLA-DOB - CD4', 'HLA-DOA - CD4',
  'ITGB2 - ICAM2', 'JAM1 - (ITGAL+ITGB2)', 'LCK - (CD8A+CD8B1)', 
 'HLA-F - CD8B', 'HLA-E - CD8B', 'HLA-C - CD8B', 'HLA-B - CD8B', 'HLA-A - CD8B', 
  'ALCAM - CD6', 'PECAM1 - PECAM1', 'CD99 - PILRA', 
  'COL4A4 - CD44', 'THBS1 - CD47',
  'FN1 - CD44', 'COL6A3 - CD44', 'COL6A2 - CD44', 'COL6A1 - CD44', 'COL4A2 - CD44', 'COL4A1 - CD44', 'COL1A2 - CD44', 'COL1A1 - CD44',
  'CD70 - CD27', 'PF4V1 - CXCR3', 'PF4 - CXCR3', 
  'PPBP - CXCR2', 'ANXA1 - FPR1', 'TNFSF12 - TNFRSF12A', 'MDK - NCL', 'IL16 - CD4')

tgt_sub1 <- df_in[which(df_in$interaction_name_2 %in% tgt_LR1 & df_in$source %not_in% c('RBC', 'GATA2+cells', 'Plasma', 'Epithelial')), ]
tmp <- tgt_sub1
tgt_p1 <- define_plot(tmp)+
  guides(y = "none", y.sec = "axis")+
  facet_grid(annotation~sample, scales = 'free', space = 'free', switch = "y"); tgt_p1
g1 <- ggplot_gtable(ggplot_build(tgt_p1))
stripr <- which(grepl('strip-t', g1$layout$name))
fills <- sample_color$col[which(sample_color$sample %in% unique(tmp$sample))]
k <- 1
for (i in stripr) {
  j <- which(grepl('rect', g1$grobs[[i]]$grobs[[1]]$childrenOrder))
  g1$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills[k]
  k <- k+1
}
stripr2 <- which(grepl('strip-l', g1$layout$name))
fills2 <- anno_color$col[which(anno_color$anno %in% unique(tmp$annotation))]
k <- 1
for (i in stripr2) {
  j <- which(grepl('rect', g1$grobs[[i]]$grobs[[1]]$childrenOrder))
  g1$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills2[k]
  k <- k+1
}
grid.draw(g1)
# 8.5*4.6


####
#### specific interaction_name_2
####

#ICAM2 - (ITGAL+ITGB2), ADGRE5 - CD55
#HLA-F - CD8A, HLA-E - CD8A, HLA-C - CD8A, HLA-B - CD8A, HLA-A - CD8A,
#CLEC2D - KLRB1, CLEC2C - KLRB1, CLEC2B - KLRB1,
#CD99 - CD99, SELPLG - SELL,
#HLA-DMA - CD4, CD22 - PTPRC, APP - CD74,
#HLA-DRB1 - CD4, HLA-DRA - CD4, HLA-DPB1 - CD4, HLA-DPA1 - CD4,
#COL9A3 - CD44, MIF - (CD74+CXCR4), BTLA - TNFRSF14,
#RETN - CAP1, MIF - (CD74+CD44), LGALS9 - CD45, LGALS9 - CD44

tgt_LR2 <- c('ICAM2 - (ITGAL+ITGB2)', 'ADGRE5 - CD55',
             'HLA-F - CD8A', 'HLA-E - CD8A', 'HLA-C - CD8A', 'HLA-B - CD8A', 'HLA-A - CD8A',
             'CLEC2D - KLRB1', 'CLEC2C - KLRB1', 'CLEC2B - KLRB1',
             'CD99 - CD99', 'SELPLG - SELL',
             'HLA-DMA - CD4', 'CD22 - PTPRC', 'APP - CD74',
             'HLA-DRB1 - CD4', 'HLA-DRA - CD4', 'HLA-DPB1 - CD4', 'HLA-DPA1 - CD4',
             #'COL9A3 - CD44', 
             'MIF - (CD74+CXCR4)', 'BTLA - TNFRSF14',
             'RETN - CAP1', 'MIF - (CD74+CD44)', 'LGALS9 - CD45', 'LGALS9 - CD44')

tgt_sub2 <- df_in[which(df_in$interaction_name_2 %in% tgt_LR2 & df_in$source %not_in% c('RBC', 'GATA2+cells', 'Plasma')), ]
tmp <- tgt_sub2
tgt_p2 <- define_plot(tmp)+
  guides(y = "none", y.sec = "axis")+
  facet_grid(annotation~sample, scales = 'free', space = 'free', switch = "y"); tgt_p2
g2 <- ggplot_gtable(ggplot_build(tgt_p2))
stripr <- which(grepl('strip-t', g2$layout$name))
fills <- sample_color$col[which(sample_color$sample %in% unique(tmp$sample))]
k <- 1
for (i in stripr) {
  j <- which(grepl('rect', g2$grobs[[i]]$grobs[[1]]$childrenOrder))
  g2$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills[k]
  k <- k+1
}
stripr2 <- which(grepl('strip-l', g2$layout$name))
fills2 <- anno_color$col[which(anno_color$anno %in% unique(tmp$annotation))]
k <- 1
for (i in stripr2) {
  j <- which(grepl('rect', g2$grobs[[i]]$grobs[[1]]$childrenOrder))
  g2$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills2[k]
  k <- k+1
}
grid.draw(g2)
#9*4.6



