

library(ggplot2)
library(dplyr)
library(tidyr)
library(reshape2)
library(purrr)
'%not_in%' <- purrr::negate(`%in%`)
library(grid)
library(paletteer)


color_useX <- paletteer_d("ggthemes::Tableau_20")[c(10,4,12)]
color_use_lineX <- paletteer_d("ggthemes::Tableau_20")[c(9,3,11)]


#------------------------------------------------------#
#------------------------------------------------------#
# Model 1 
#------------------------------------------------------#
#------------------------------------------------------#


#-------------------------------------#
# Model 1 Performance
#-------------------------------------#



dtdt <- read.csv("R_data/PERFORMANCE_Model1_df.csv", header = T)

dtdt$variable <- factor(dtdt$variable, 
                          levels = c('Specificity', 'Sensitivity', 'Precision',
                                     'F_0.5', 'F_1', 'F_2'))
dtdt$Group2 <- factor(dtdt$Group2, 
                       levels = unique(dtdt$Group2)[c(2,1,3,4)])
ggplot(dtdt, aes(x= Group2, y = value, 
                 color = variable, fill = variable))+
  #geom_quasirandom(alpha = 0.5, size = 1)+
  geom_boxplot(linewidth = 0.3, alpha = 0.5, #color='black', 
               outlier.shape = 1, outlier.colour = 'gray50', outlier.size = 0.2)+
  scale_y_continuous(limits = c(0.65, 1.02))+
  guides(color = guide_legend(nrow = 2))+ #, shape = guide_legend(nrow = 3)
  labs(title = "Model 1 performance")+
  ylab("Classification accuracy")+
  theme_bw(base_size = 20)+
  theme(#panel.grid  = element_blank(),
        axis.text.x = element_text(angle = 20, hjust = 1, vjust = 1, color = 'black'),
        axis.title.x = element_blank(),
        axis.text.y = element_text(color = 'black'),
        #axis.title.y = element_blank(),
        strip.background = element_rect(fill = 'gray95'),
        axis.ticks.x = element_blank(),
        strip.text.y = element_text(size = 8),
        plot.margin = margin(l = 30, b = 10, r = 10, t = 10),
        legend.position = 'bottom',
        legend.title = element_blank(),
        legend.text = element_text(size = 18),
        #panel.grid = element_line(color = 'gray85', linewidth = 0.1),
        panel.border = element_rect(colour = 'black', fill = NA, linewidth = 1.5)
  )
# 6.8*3.9
ggsave("R_figure/NEW_OUT_model1_performance.pdf", height =6.4, width = 6.4, units = "in")


#-------------------------------------#
# Model 1 vs Other methods, Comparison
#-------------------------------------#

comparison_data_ALL_Met <- read.csv("R_data/NEW_model1_comparison_MetALL.csv", header=T, row.names = 1)
comparison_data_ALL_CTC <- read.csv("R_data/NEW_model1_comparison_CTCALL.csv", header=T, row.names = 1)
comparison_data_ALL <- read.csv("R_data/NEW_model1_comparison_PrimaryALL.csv", header=T, row.names = 1)

comparison_data_ALL_Met$Types <- "Metastatic cancer cell"
comparison_data_ALL$Types <- "Primary cancer cell"
comparison_data_ALL_CTC$Types <- "CTC"
comn_AAALL <- bind_rows(comparison_data_ALL_Met, comparison_data_ALL_CTC, comparison_data_ALL)

# 创建比较图
ggplot(comn_AAALL[which(comn_AAALL$Metr %in% c('Sensitivity', 'Specificity')), ], aes(x = value_other, y = value_panctc, color = Method_other, shape = Types)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  scale_y_continuous(limits = c(0, 1))+
  scale_shape_manual(values=c(1,2,3))+
  guides(color = guide_legend(nrow = 3), shape = guide_legend(nrow = 3))+
  facet_wrap(Metr~., nrow=1)+
  labs(
    #title = paste0(metr000[mm], " Comparison"),
    x = "Value of Other Methods",
    y = "Value of panCTC",
    color = "Method"
  )+
  theme_bw(base_size = 20)+
  theme(#panel.grid  = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, color = 'black'),
    #axis.title.x = element_blank(),
    axis.text.y = element_text(color = 'black'),
    #axis.title.y = element_blank(),
    strip.background = element_rect(fill = 'gray80', linewidth = 1.5),
    axis.ticks.x = element_blank(),
    strip.text = element_text(size = 20),
    plot.margin = margin(l = 5, b = 5, r = 5, t = 5),
    legend.position = 'bottom',
    legend.title = element_blank(),
    legend.text = element_text(size = 18),
    legend.key.size = unit(0.5, "cm"),
    #panel.grid = element_line(color = 'gray85', linewidth = 0.1),
    panel.border = element_rect(colour = 'black', fill = NA, linewidth = 1.5)
  )
ggsave("R_figure/NEW_model1_comparison_ALL.pdf", 
       height = 6.4, width = 8.6, units = "in")




#------------------------------------------------------#
#------------------------------------------------------#
# Model 2
#------------------------------------------------------#
#------------------------------------------------------#


#-------------------------------------#
# Model 2 Performance
#-------------------------------------#


comparison_data_Pri <- read.csv("R_data/NEW_model2_comparison_PrimaryALL.csv", header = T, row.names = 1)
comparison_data_Met <- read.csv("R_data/NEW_model2_comparison_MetALL.csv", header = T, row.names = 1)
comparison_data_CTC <- read.csv("R_data/NEW_model2_comparison_CTCALL.csv", header = T, row.names = 1)

head(comparison_data_Pri)
head(comparison_data_Met)
head(comparison_data_CTC)

#colnames(comparison_data_CTC) <- c("col1", "Method_other", "Acc_other", "Acc_panctc")
comparison_data_Met$Types <- "Metastatic cancer cell"
comparison_data_Pri$Types <- "Primary cancer cell"
comparison_data_CTC$Types <- "CTC"
comn_AAALL_model2 <- rbind(comparison_data_CTC, comparison_data_Met, comparison_data_Pri)
comn_AAALL_model2$Metr <- "Tracing accuracy"
# 创建比较图
ggplot(comn_AAALL_model2, aes(x = Acc_other, y = Acc_panctc, color = Method_other, shape = Types)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  scale_y_continuous(limits = c(0, 1))+
  scale_shape_manual(values=c(1,2,3))+
  guides(color = guide_legend(nrow = 2), shape = guide_legend(nrow = 3))+
  facet_wrap(Metr~., nrow=2)+
  labs(
    #title = paste0(metr000[mm], " Comparison"),
    x = "Value of other Methods",
    y = "Value of panCTC",
    color = "Method"
  )+
  theme_bw(base_size = 20)+
  theme(#panel.grid  = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, color = 'black'),
    #axis.title.x = element_blank(),
    axis.text.y = element_text(color = 'black'),
    #axis.title.y = element_blank(),
    strip.background = element_rect(fill = 'gray80', linewidth = 1.5),
    axis.ticks.x = element_blank(),
    strip.text = element_text(size = 20),
    plot.margin = margin(l = 5, b = 5, r = 5, t = 5),
    legend.position = 'bottom',
    legend.title = element_blank(),
    legend.text = element_text(size = 18),
    legend.key.size = unit(0.5, "cm"),
    #panel.grid = element_line(color = 'gray85', linewidth = 0.1),
    panel.border = element_rect(colour = 'black', fill = NA, linewidth = 1.5)
  )
ggsave("R_figure/NEW_model2_comparison_ALL.pdf", 
       height = 6.4, width = 8.6, units = "in")




#-------------------------------------#
# Model 2 vs Other methods, Comparison
#-------------------------------------#



new2025_com  <- read.csv("R_data/PERFORMANCE_Model2_df.csv", header = T)

new2025_com$Group3 <- factor(new2025_com$Group3, levels = unique(new2025_com$Group3))
ggplot(new2025_com, aes(x= Group3, y = value, fill = variable, color = variable)) + 
  geom_boxplot(linewidth = 0.3, #color = 'black', 
               outlier.shape = 1, outlier.colour = 'gray50', outlier.size = 0.2)+
  #geom_jitter(position = 'identity', size =1)+
  scale_colour_manual(values = color_use_lineX)+
  scale_fill_manual(values = color_useX)+
  #scale_colour_discrete_sequential(palette = 'Sunset')+
  #scale_fill_discrete_sequential(palette = 'Sunset')+
  labs(title='Model 2 performance')+
  ylab('Tracing accuracy')+
  theme_bw(base_size = 20)+
  theme(#panel.grid  = element_blank(),
    axis.text.x = element_text(angle = 20, hjust = 1, vjust = 1, color = 'black'),
    axis.title.x = element_blank(),
    axis.text.y = element_text(color = 'black'),
    #axis.title.y = element_blank(),
    strip.background = element_rect(fill = 'gray95'),
    axis.ticks.x = element_blank(),
    strip.text.y = element_text(size = 8),
    #plot.margin = margin(l = 50, b = 40, r = 10, t = 10),
    legend.position = 'bottom',
    legend.title = element_blank(),
    legend.text = element_text(size = 18),
    #panel.grid = element_line(color = 'gray85', linewidth = 0.1),
    panel.border = element_rect(colour = 'black', fill = NA, linewidth = 1.5)
  )
ggsave("R_figure/NEW_OUT_model2_performance_V10CC.pdf", 
       height = 6.4, width = 6.4, units = "in")
