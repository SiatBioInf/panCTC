



out_rs  <- read.csv("R_data/COMPARISON_CTC_Detection_df.csv", header = T)

out_rs$method <- factor(out_rs$method, levels = c("panCTC", "scATOMIC", "CTCTracer", "iCTC", "scGPT", "scfoundation"))


# 数据处理 - 计算均值和标准差
summary_data <- out_rs %>%
  group_by(method, cancer) %>%
  summarise(
    rate1_mean = mean(rate1_total, na.rm = TRUE),
    rate1_sd = sd(rate1_total, na.rm = TRUE),
    rate2_mean = mean(rate2_overlap_self, na.rm = TRUE),
    rate2_sd = sd(rate2_overlap_self, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  pivot_longer(
    cols = c(rate1_mean, rate2_mean),
    names_to = "rate_type",
    values_to = "mean_value"
  ) %>%
  mutate(
    sd_value = ifelse(rate_type == "rate1_mean", rate1_sd, rate2_sd),
    rate_type = factor(rate_type, 
                       levels = c("rate1_mean", "rate2_mean"),
                       labels = c("Proportion\n(in total PBMCs)", "Proportion\n(in intersected CTCs)"))
  )

summary_data$Shape <- "Other"
summary_data$Shape[which(summary_data$method == 'panCTC')] <- "panCTC"
# 创建优雅的点图
p1 <- ggplot(summary_data, aes(x = cancer, y = mean_value, color = method, shape = Shape)) +
  geom_point(position = position_dodge(width = 0.6), size = 3, alpha=1) +
  geom_errorbar(aes(ymin = mean_value - sd_value, ymax = mean_value + sd_value),
                width = 0.2, position = position_dodge(width = 0.6)) +
  facet_wrap(~ rate_type, scales = "free_y", ncol = 1) +
  scale_shape_manual(values = c(19, 3))+
  scale_color_brewer(palette = "Set1") +
  guides(color = guide_legend(nrow = 6), shape = guide_legend(nrow = 2))+
  labs(
    x = "Cancer Type",
    y = "Rate Value",
    color = "Method",
    #title = "Comparison of Rate1 and Rate2 Across Methods and Cancer Types"
  ) +
  theme_bw(base_size = 18)+
  theme(panel.grid  = element_blank(),
        axis.text.x = element_text(angle = 50, hjust = 1, vjust = 1, color = 'black'),
        axis.text.y = element_text(color = 'black'),
        strip.background = element_rect(fill = 'gray95'),
        strip.text.y = element_text(size = 8),
        legend.position = "bottom",
        legend.title = element_blank()
  )

print(p1)

ggsave("R_figure/Direction_rate_Comparison.pdf", p1, height = 9.2, width=4.2, units = "in")

