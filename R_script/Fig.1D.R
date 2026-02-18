library(ggplot2)

# === Step 1: 读取数据 ===
CUS <- read.csv("Kidney_Cancer.CUS.mat.csv", row.names = 1, check.names = FALSE)
ATAC <- read.csv("Kidney_Cancer.atac_count.csv", row.names = 1, check.names = FALSE)

# === Step 2: 按细胞逐列计算 spearman 相关系数 ===
cors <- sapply(colnames(CUS), function(cell) {
  x <- CUS[, cell]
  y <- ATAC[, cell]
  
  
  # 去掉 ATAC == 0 且 CUS != 0 的点
  valid_idx <- !(y == 0 & x != 0)
  
  if (sum(valid_idx) > 2) {
    cor(x[valid_idx], y[valid_idx], method = "spearman")
  } else {
    NA  # 如果有效点太少，返回 NA
  }
})

# === Step 3: 转换为数据框 ===
df <- data.frame(Cell = names(cors), Correlation = cors)


# === Step 4: 计算统计值 ===
summary_stats <- c(
  mean   = mean(df$Spearman, na.rm = TRUE),
  median = median(df$Spearman, na.rm = TRUE),
  sd     = sd(df$Spearman, na.rm = TRUE),
  min    = min(df$Spearman, na.rm = TRUE),
  max    = max(df$Spearman, na.rm = TRUE),
  n      = sum(!is.na(df$Spearman)),
  na     = sum(is.na(df$Spearman))
)

print(summary_stats)


# === Step 4: 可视化 ===
# 直方图
p1 <- ggplot(df, aes(x = Correlation)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  theme_minimal(base_size = 14) +
  ggtitle("Distribution of Spearman Correlation per Cell\n(ignoring ATAC=0 & CUS≠0 cases)")

# 箱线图
p2 <- ggplot(df, aes(y = Correlation)) +
  geom_boxplot(fill = "tomato", alpha = 0.6) +
  theme_minimal(base_size = 14) +
  ggtitle("Spearman Correlation per Cell\n(ignoring ATAC=0 & CUS≠0 cases)")

# 保存结果
ggsave("Kidney_Cancer1000_Correlation_Histogram_filtered.png", p1, width = 6, height = 4)
ggsave("Kidney_Cancer1000_Correlation_Boxplot_filtered.png", p2, width = 4, height = 6)

# === 可选：导出相关性结果 ===
write.csv(df, "Kidney_Cancer_Cellwise_Correlation_filtered.csv", row.names = FALSE)


summary(cors)