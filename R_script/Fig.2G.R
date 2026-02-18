

library(Seurat)
library(dplyr)
library(patchwork)
library(tidyr)
library(Nebulosa)
library(DoubletFinder)
library(purrr)
'%not_in%' <- purrr::negate(`%in%`)
library(paletteer)




## ------------------------------ ##
##  OVC (rc47y6m9mp-1), CTCs
## ------------------------------ ##
inp_fd <- 'R_data/OVC_CTC_prediction/'


ctc_bcd <- NULL
lg <- NULL
smp <- c('ECO1', 'HGSOC1', 'HGSOC2', 'HGSOC3', 'HGSOC4', 'HGSOC6')
for(i in 1:length(smp)) {
  df0 <- read.csv(paste0(inp_fd, 'predict_label_Model1_rc47y6m9mp-1_', smp[i], '.csv'), header = T)
  bcd0 <- df0$X[which(df0$Predict == 1)]
  lg0 <- length(bcd0)
  lg <- c(lg, lg0)
  ctc_bcd <- c(ctc_bcd, bcd0)
  
}
length(ctc_bcd)
d0 <- data.frame(PBMC = smp, CTC_counts = lg)

# -----------------------------
# OVC (rc47y6m9mp-1), PBMCs
# -----------------------------

####
## The raw Seurat object is the  processed data from PRJCA005422
##  which is deposited in Mendeley Data
## https://doi.org/10.17632/rc47y6m9mp.1
####

rc47 <- readRDS('R_data/rc47y6m9mp-1/raw_object.rds')


dim(rc47@meta.data)
head(rc47@meta.data)

obj <- CreateSeuratObject(counts = rc47@assays$RNA@counts)
head(obj@meta.data)
dim(obj)
obj[['Samples']] <- rc47@meta.data$Samples
obj[['Groups']] <- rc47@meta.data$Groups
obj[['Patients']] <- rc47@meta.data$Patients
obj[['Annotation']] <- rc47@meta.data$Annotation
obj[['Group_abb']] <- rc47@meta.data$Group_abb
obj[['maintypes_2']] <- rc47@meta.data$maintypes_2
obj[['maintypes_3']] <- rc47@meta.data$maintypes_3

table(obj@meta.data$Groups)



####
#### PBMC, CTC objects
####

obj_PBMC <- subset(obj, Groups == 'PBMC')
obj_PBMC[['cells']] <- rownames(obj_PBMC@meta.data)




# ---------------------------------------
# OVC (rc47y6m9mp-1), CTCs
# ---------------------------------------

obj_CTCrc47 <- subset(obj_PBMC, cells %in% ctc_bcd)
dim(obj_CTCrc47)

obj_CTCrc47 <- NormalizeData(obj_CTCrc47)
obj_CTCrc47 <- FindVariableFeatures(obj_CTCrc47, selection.method = "vst", nfeatures = 2000)
obj_CTCrc47 <- ScaleData(obj_CTCrc47)
obj_CTCrc47 <- RunPCA(obj_CTCrc47)
library(harmony)
obj_CTCrc47 <- RunHarmony(obj_CTCrc47, group.by.vars = 'Samples')
obj_CTCrc47 <- FindNeighbors(obj_CTCrc47)
obj_CTCrc47 <- FindClusters(obj_CTCrc47, algorithm = 1, resolution = c(0.1, 0.3, 0.5))
obj_CTCrc47 <- RunUMAP(obj_CTCrc47, reduction = "pca", dims = 1:20)
#obj_CTCrc47 <- RunUMAP(obj_CTCrc47, reduction = "harmony", dims = 1:20)
FeaturePlot(obj_CTCrc47, features = c('EPCAM', 'KRT7', 'VIM', 'KRAS'))
DotPlot(obj_CTCrc47, features = c('EPCAM', 'KRT7', 'VIM', 'ZEB2', 'ERBB2', 'ERBB3', 'MUC1', 'MUC16', 'KRAS', 'TP53'), group.by = 'RNA_snn_res.0.5')+RotatedAxis()
DimPlot(obj_CTCrc47, group.by = 'RNA_snn_res.0.5') + DimPlot(obj_CTCrc47, group.by = 'Samples')

ctcShr <- c('VIM', 'KRAS')

ctc_smp <- unique(obj_CTCrc47@meta.data$Samples)
ctc_smp <- stringr::str_remove_all(ctc_smp, '_BC')


obj_CTCrc47 <- SetIdent(obj_CTCrc47, value = 'RNA_snn_res.0.5')
mks_CTC <- FindAllMarkers(obj_CTCrc47, only.pos = T, logfc.threshold = 0.6)


# ---------------------------------------
# OVC (rc47y6m9mp-1), Metastatic Tumor
# ---------------------------------------

obj_MetTumor <- subset(obj, Groups == 'Metastatic Tumor')
table(obj_MetTumor@meta.data$Patients)
table(obj_MetTumor@meta.data$Samples)
table(obj_MetTumor@meta.data$maintypes_3)

Met_smp <- unique(obj_MetTumor@meta.data$Samples)
Met_smp <- stringr::str_remove_all(Met_smp, '_MT')
shr_smp <- intersect(ctc_smp, Met_smp)
Met_smp <- paste0(shr_smp, '_MT')

obj_MetCan <- subset(obj_MetTumor, maintypes_3 == 'Cancer cells' & Samples %in% Met_smp)
obj_MetCan <- NormalizeData(obj_MetCan)
obj_MetCan <- FindVariableFeatures(obj_MetCan, selection.method = "vst", nfeatures = 2000)
obj_MetCan <- ScaleData(obj_MetCan)
obj_MetCan <- RunPCA(obj_MetCan)
library(harmony)
obj_MetCan <- RunHarmony(obj_MetCan, group.by.vars = 'Samples')
obj_MetCan <- FindNeighbors(obj_MetCan)
obj_MetCan <- FindClusters(obj_MetCan, algorithm = 1, resolution = c(0.05, 0.1))
obj_MetCan <- RunUMAP(obj_MetCan, reduction = "pca", dims = 1:20)
DimPlot(obj_MetCan, group.by = 'RNA_snn_res.0.05')

obj_MetCan <- SetIdent(obj_MetCan, value = 'RNA_snn_res.0.05')
mks_MetCan <- FindAllMarkers(obj_MetCan, only.pos = T, logfc.threshold = 1)





# ---------------------------------------
# OVC (rc47y6m9mp-1), Lymph Node
# ---------------------------------------

obj_LN <- subset(obj, Groups == 'Lymph Node')
table(obj_LN@meta.data$Patients)
table(obj_LN@meta.data$Samples)
table(obj_LN@meta.data$maintypes_3)

LN_smp <- unique(obj_LN@meta.data$Samples)
LN_smp <- stringr::str_remove_all(LN_smp, '_LN')
shr_smp <- intersect(ctc_smp, LN_smp)
LN_smp <- paste0(shr_smp, '_LN')

obj_LNCan <- subset(obj_LN, maintypes_3 == 'Cancer cells' &
                     Samples %in% LN_smp)
obj_LNCan <- NormalizeData(obj_LNCan)
obj_LNCan <- FindVariableFeatures(obj_LNCan, selection.method = "vst", nfeatures = 2000)
obj_LNCan <- ScaleData(obj_LNCan)
obj_LNCan <- RunPCA(obj_LNCan)
library(harmony)
obj_LNCan <- RunHarmony(obj_LNCan, group.by.vars = 'Samples')
obj_LNCan <- FindNeighbors(obj_LNCan)
obj_LNCan <- FindClusters(obj_LNCan, algorithm = 1, resolution = c(0.05, 0.1, 0.3))
obj_LNCan <- RunUMAP(obj_LNCan, reduction = "pca", dims = 1:20)
DimPlot(obj_LNCan, group.by = 'RNA_snn_res.0.3')

obj_LNCan <- SetIdent(obj_LNCan, value = 'RNA_snn_res.0.3')
mks_LNCan <- FindAllMarkers(obj_LNCan, only.pos = T, logfc.threshold = 0.6)



# ---------------------------------------
# OVC (rc47y6m9mp-1), Ascites
# ---------------------------------------

obj_Ascites <- subset(obj, Groups == 'Ascites')
table(obj_Ascites@meta.data$Patients)
table(obj_Ascites@meta.data$Samples)
table(obj_Ascites@meta.data$maintypes_3)

Asc_smp <- unique(obj_Ascites@meta.data$Samples)
Asc_smp <- stringr::str_remove_all(Asc_smp, '_AS')
shr_smp <- intersect(ctc_smp, Asc_smp)
Asc_smp <- paste0(shr_smp, '_AS')

obj_AscCan <- subset(obj_Ascites, maintypes_3 == 'Cancer cells' &
                      Samples %in% Asc_smp)
obj_AscCan <- NormalizeData(obj_AscCan)
obj_AscCan <- FindVariableFeatures(obj_AscCan, selection.method = "vst", nfeatures = 2000)
obj_AscCan <- ScaleData(obj_AscCan)
obj_AscCan <- RunPCA(obj_AscCan)
library(harmony)
obj_AscCan <- RunHarmony(obj_AscCan, group.by.vars = 'Samples')
obj_AscCan <- FindNeighbors(obj_AscCan)
obj_AscCan <- FindClusters(obj_AscCan, algorithm = 1, resolution = c(0.1, 0.3, 0.5))
obj_AscCan <- RunUMAP(obj_AscCan, reduction = "pca", dims = 1:20)
DimPlot(obj_AscCan, group.by = 'RNA_snn_res.0.3')

obj_AscCan <- SetIdent(obj_AscCan, value = 'RNA_snn_res.0.3')
mks_AscCan <- FindAllMarkers(obj_AscCan, only.pos = T, logfc.threshold = 0.6)



# ---------------------------------------
# OVC, Shared markers
# ---------------------------------------


nx = 50
top20_CTC <- mks_CTC %>% group_by(cluster) %>% top_n(n = nx, wt = avg_log2FC)

top20_MetCan <- mks_MetCan %>% group_by(cluster) %>% top_n(n = nx, wt = avg_log2FC)
MetShr <- intersect(top20_CTC$gene, top20_MetCan$gene)

top20_LNCan <- mks_LNCan %>% group_by(cluster) %>% top_n(n = nx, wt = avg_log2FC)
LNShr <- intersect(top20_CTC$gene, top20_LNCan$gene)

top20_AscCan <- mks_AscCan %>% group_by(cluster) %>% top_n(n = nx, wt = avg_log2FC)
AscShr <- intersect(top20_CTC$gene, top20_AscCan$gene)

ShrGenes50 <- unique(c(ctcShr,
                     MetShr, LNShr, AscShr))
length(ShrGenes50)


nx = 20
top20_CTC <- mks_CTC %>% group_by(cluster) %>% top_n(n = nx, wt = avg_log2FC)

top20_MetCan <- mks_MetCan %>% group_by(cluster) %>% top_n(n = nx, wt = avg_log2FC)
MetShr <- intersect(top20_CTC$gene, top20_MetCan$gene)

top20_LNCan <- mks_LNCan %>% group_by(cluster) %>% top_n(n = nx, wt = avg_log2FC)
LNShr <- intersect(top20_CTC$gene, top20_LNCan$gene)

top20_AscCan <- mks_AscCan %>% group_by(cluster) %>% top_n(n = nx, wt = avg_log2FC)
AscShr <- intersect(top20_CTC$gene, top20_AscCan$gene)

ShrGenes20 <- unique(c(ctcShr,
                       MetShr, LNShr, AscShr))
length(ShrGenes20)

#### nx = 50
# [1] "VIM"      "KRAS"     "BTG2"     "ITM2C"    "HLA-DQA1" "ATP5G3"   "TMSB4X"   "CD52"     "S100A8"   "CST3"    
# [11] "S100A9"   "SERPINA1" "FOS"      "CFD"      "CEBPD"    "FTH1"     "S100A6"   "S100A4"   "NEAT1"    "LGALS3"  
# [21] "DUSP1"    "IFITM3"   "SSR4"     "UBE2C"    "CDC20"    "HLA-A"    "RARRES3"  "TYROBP"   "FTL"      "S100A11" 
# [31] "PSAP" 

#### nx = 20
# "VIM"    "KRAS"   "ITM2C"  "S100A9" "FTH1"   "S100A6" "TYROBP" "FOS"  



# ---------------------------------------
# OVC, Display selected share markers
# ---------------------------------------


use_mks <- c("EPCAM", "KRT7",
             "VIM", "FTH1", "TMSB4X", "S100A4", 
             "S100A6", "FTL", "SSR4", 'S100A8' #,
             #"CD52", "NEAT1", "LGALS3", "S100A9"
             )  # "ATP5G3", "HLA-A", 





tmU1 <- theme_bw() + theme(axis.title = element_blank(),
                           axis.text.y = element_text(size = 10, colour = 'black'),
                           axis.text.x = element_blank(),
                           axis.ticks.x = element_blank(),
                           legend.position = 'none',
                           panel.border = element_rect(linewidth = 1.3, colour = 'black'))
tmU2 <- theme_bw() + theme(axis.title = element_blank(),
                           axis.text.y = element_text(size = 10, colour = 'black'),
                           axis.text.x = element_text(size = 10, colour = 'black', angle = 45, hjust = 1, vjust = 1),
                           legend.position = 'none',
                           panel.border = element_rect(linewidth = 1.3, colour = 'black'))
colx <- paletteer_d("ggthemes::Tableau_10")[c(3)]


obj_CTCrc47[['Source']] <- 'CTCs\nfrom PBMC'
obj_MetCan[['Source']] <- 'Metastatic\ncancer cells'
obj_LNCan[['Source']] <- 'Lymph node\ncancer cells'
obj_AscCan[['Source']] <- 'Ascites\ncancer cells'
ppp1 <- DotPlot(obj_CTCrc47, features = use_mks, group.by = 'Source', cols = c("lightgrey", colx)) + RotatedAxis() + scale_y_discrete(position = 'right')+ tmU1
ppp2 <- DotPlot(obj_MetCan, features = use_mks, group.by = 'Source', cols = c("lightgrey", colx)) + RotatedAxis() + scale_y_discrete(position = 'right')+ tmU1
ppp3 <- DotPlot(obj_LNCan, features = use_mks, group.by = 'Source', cols = c("lightgrey", colx)) + RotatedAxis() + scale_y_discrete(position = 'right')+ tmU1
ppp4 <- DotPlot(obj_AscCan, features = use_mks, group.by = 'Source', cols = c("lightgrey", colx)) + RotatedAxis() + scale_y_discrete(position = 'right')+ tmU2
#ppp1/ppp2/ppp3/ppp4

dotdf1 <- ppp1$data
dotdf2 <- ppp2$data
dotdf3 <- ppp3$data
dotdf4 <- ppp4$data

dotdf <- rbind(dotdf1, dotdf2, dotdf3, dotdf4)
table(dotdf$id)
ggplot(dotdf, aes(x = features.plot, y = id)) + 
  geom_point(aes(size=pct.exp, color = avg.exp.scaled)) + 
  scale_color_gradientn(colours=c("gray80", colx),
    #colours = rev(paletteer_c("grDevices::Inferno", 30)),
    #trans = "log1p", 
    guide = guide_colorbar(reverse=F, order=1)) +
  scale_size_continuous(range=c(0.05, 5)) + 
  scale_y_discrete(position = 'right')+ 
  tmU2 + 
  theme(#legend.position = 'right',
        plot.margin = margin(l = 20)
        )

ggsave('Shared_Markers_CTCs-Met-LN-Asc_DotPlot.pdf', path = 'R_figure',
       width = 3.2, height = 3, units = 'in')