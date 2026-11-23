# ==============================================================
# 作業 01：乳癌良性與惡性分類（完整補充版）
# 基於老師提供的 01_analysis_LDA_QDA_SVM.R 架構
# 模型：LDA、QDA、Linear SVM、RBF SVM
# 使用方式：將本檔與 HW01_Breast_Cancer_Data.csv 放在同一資料夾。
#           修改下方 setwd() 為你的資料夾路徑，再 Source 本檔。
# install.packages(c("MASS", "e1071"))  # 首次執行前安裝
# 僅供教學研究，不可作為臨床診斷工具。
# ==============================================================

# [1] 設定工作目錄並讀取 CSV
# 已設定為你提供的作業資料夾
# Windows 路徑請使用 /，例如 C:/Users/Mark/Desktop/ML_HW01
setwd("C:/Users/User/Desktop/機器學習/HW")

# 直接使用 read.csv() 開啟作業資料
raw <- read.csv("HW01_Breast_Cancer_Data.csv",
                header = TRUE,
                check.names = TRUE,
                stringsAsFactors = FALSE,
                na.strings = c("", "NA", "?"),
                fileEncoding = "UTF-8-BOM")
file_in <- "HW01_Breast_Cancer_Data.csv"

required <- c("MASS", "e1071")
missing_packages <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) stop("請先安裝套件：install.packages(c(", paste(shQuote(missing_packages), collapse=", "), "))")
if (!dir.exists("results")) dir.create("results", recursive = TRUE)

if (!"Class" %in% names(raw)) stop("資料中缺少 Class 欄")
NEGATIVE <- "B"; POSITIVE <- "M"
raw$Class <- factor(as.character(raw$Class), levels=c(NEGATIVE, POSITIVE))
if (anyNA(raw$Class)) stop("Class 含有無法辨識的標籤")
exclude_cols <- c("Class", "ID", "PatientID")
predictor_names <- setdiff(names(raw), exclude_cols)
if (length(predictor_names) != 30L) warning("預測變數數量不是 30，請確認資料內容")
if (!all(vapply(raw[predictor_names], is.numeric, logical(1))))
  stop("本作業預期 30 個預測變數均為數值型")
cat("資料來源：", file_in, "\n")

# [2] EDA：樣本、類別比例、缺失值、兩個重要變數的分布
class_tab <- as.data.frame(table(raw$Class))
names(class_tab) <- c("Class", "Count")
class_tab$Percentage <- round(100 * class_tab$Count / nrow(raw), 2)
write.csv(class_tab, "results/class_distribution.csv", row.names=FALSE, fileEncoding="UTF-8")
missing_tab <- data.frame(Variable=names(raw), Missing=sapply(raw, function(z) sum(is.na(z))))
write.csv(missing_tab, "results/missing_values.csv", row.names=FALSE, fileEncoding="UTF-8")
important_vars <- c("mean_radius", "mean_concavity")
if (!all(important_vars %in% names(raw))) stop("缺少 mean_radius 或 mean_concavity")
eda_rows <- lapply(important_vars, function(v) {
  do.call(rbind, lapply(levels(raw$Class), function(g) {
    z <- raw[raw$Class == g, v]
    data.frame(Variable=v, Class=g, N=sum(!is.na(z)),
               Mean=mean(z, na.rm=TRUE), Median=median(z, na.rm=TRUE),
               SD=sd(z, na.rm=TRUE), Q1=unname(quantile(z,.25,na.rm=TRUE)),
               Q3=unname(quantile(z,.75,na.rm=TRUE)))
  }))
})
eda_summary <- do.call(rbind, eda_rows)
write.csv(eda_summary, "results/eda_two_features.csv", row.names=FALSE, fileEncoding="UTF-8")

png("results/eda_boxplots.png", width=1100, height=550, res=125)
par(mfrow=c(1,2), mar=c(5,4,4,1))
for (v in important_vars) {
  boxplot(raw[[v]] ~ raw$Class, col=c("#35769A", "#C45B47"),
          main=paste(v, "by Class"), xlab="Class (B / M)", ylab=v)
}
dev.off()

# [3] 訓練 / 測試分層切分（固定亂數種子）
set.seed(20261018)
train_idx <- unlist(lapply(split(seq_len(nrow(raw)), raw$Class), function(ii)
  sample(ii, size=floor(length(ii)*0.8))), use.names=FALSE)
test_idx <- setdiff(seq_len(nrow(raw)), train_idx)
train_y <- factor(raw$Class[train_idx], levels=c(NEGATIVE,POSITIVE))
test_y <- factor(raw$Class[test_idx], levels=c(NEGATIVE,POSITIVE))
stopifnot(all(table(train_y)>0), all(table(test_y)>0))

# [4] 數值設計矩陣：排除目標及 ID（30 個數值欄位）
x_df <- raw[predictor_names]
X <- model.matrix(~ . - 1, data=x_df, na.action=na.pass)
X_train <- X[train_idx,,drop=FALSE]
X_test <- X[test_idx,,drop=FALSE]

# [5] 僅用訓練集計算填補中位數，並套用至測試集
for (j in seq_len(ncol(X_train))) {
  med <- median(X_train[,j], na.rm=TRUE)
  if (!is.finite(med)) med <- 0
  X_train[is.na(X_train[,j]),j] <- med
  X_test[is.na(X_test[,j]),j] <- med
}

# [6] 僅依訓練集去除常數特徵
variances <- apply(X_train,2,var)
keep <- which(is.finite(variances) & variances > 1e-10)
if (length(keep)<2) stop("有效特徵不足")
X_train <- X_train[,keep,drop=FALSE]
X_test <- X_test[,keep,drop=FALSE]
if (ncol(X_train)>100) {
  selected <- order(variances[keep], decreasing=TRUE)[seq_len(100)]
  X_train <- X_train[,selected,drop=FALSE]
  X_test <- X_test[,selected,drop=FALSE]
}

# [7] 特徵相關性及 PCA：只在訓練集估計中心、尺度及主成分
corr_mat <- cor(X_train)
upper_corr <- abs(corr_mat[upper.tri(corr_mat)])
high_corr_pairs <- sum(upper_corr > .8, na.rm=TRUE)
write.csv(corr_mat, "results/train_feature_correlations.csv", fileEncoding="UTF-8")
max_pc <- min(5L, ncol(X_train), nrow(X_train)-2L,
              as.integer(min(table(train_y))-2L))
if (max_pc<2) stop("各類別訓練樣本不足，無法穩定配適 QDA")
pca <- prcomp(X_train, center=TRUE, scale.=TRUE)
train_PC <- as.data.frame(pca$x[,seq_len(max_pc),drop=FALSE])
test_PC <- as.data.frame(predict(pca,newdata=X_test)[,seq_len(max_pc),drop=FALSE])
names(train_PC) <- names(test_PC) <- paste0("PC",seq_len(max_pc))
explained <- pca$sdev^2/sum(pca$sdev^2)
pca_tbl <- data.frame(PC=paste0("PC",seq_along(explained)),
                      ExplainedVariance=explained, CumulativeVariance=cumsum(explained))
write.csv(pca_tbl,"results/pca_variance.csv",row.names=FALSE,fileEncoding="UTF-8")
train_dat <- data.frame(Class=train_y,train_PC)
test_dat <- data.frame(Class=test_y,test_PC)

png("results/pca_plot.png",width=950,height=700,res=125)
plot(train_PC$PC1,train_PC$PC2,
     col=ifelse(train_y==POSITIVE,"#C45B47","#35769A"),pch=19,
     xlab="PC1",ylab="PC2",main="Training data: first two principal components")
legend("topright",legend=c("B (benign)","M (malignant)"),
       col=c("#35769A","#C45B47"),pch=19,bty="n")
dev.off()

png("results/pca_cumulative_variance.png",width=900,height=600,res=125)
plot(seq_along(explained),cumsum(explained),type="b",pch=19,
     xlab="Number of principal components",ylab="Cumulative explained variance",
     main="PCA explained variance (training set)",ylim=c(0,1))
abline(v=max_pc,lty=2,col="gray40")
dev.off()

# [8] 配適四個分類器；沿用老師固定超參數，未使用測試集調參
fit_lda <- MASS::lda(Class~.,data=train_dat)
fit_qda <- tryCatch(MASS::qda(Class~.,data=train_dat),
                    error=function(e) {warning("QDA 無法配適：",conditionMessage(e));NULL})
fit_linear <- e1071::svm(Class~.,data=train_dat,kernel="linear",cost=1,scale=TRUE)
fit_rbf <- e1071::svm(Class~.,data=train_dat,kernel="radial",cost=1,
                      gamma=1/max_pc,scale=TRUE)
predictions <- list(
  LDA=predict(fit_lda,newdata=test_dat)$class,
  QDA=if (is.null(fit_qda)) NULL else predict(fit_qda,newdata=test_dat)$class,
  SVM_Linear=predict(fit_linear,newdata=test_dat),
  SVM_RBF=predict(fit_rbf,newdata=test_dat))
predictions <- predictions[!vapply(predictions,is.null,logical(1))]

# [9] 混淆矩陣：以 M（惡性）為陽性；FN=實際 M、預測 B
metrics <- function(actual,predicted,positive="M") {
  a <- as.character(actual); p <- as.character(predicted)
  TP <- sum(a==positive & p==positive)
  FN <- sum(a==positive & p!=positive)
  TN <- sum(a!=positive & p!=positive)
  FP <- sum(a!=positive & p==positive)
  div <- function(x,y) if (y==0) NA_real_ else x/y
  sensitivity <- div(TP,TP+FN)
  specificity <- div(TN,TN+FP)
  precision <- div(TP,TP+FP)
  f1 <- div(2*precision*sensitivity,precision+sensitivity)
  c(TP=TP,FN=FN,TN=TN,FP=FP,
    Accuracy=div(TP+TN,TP+TN+FP+FN),Sensitivity=sensitivity,
    Specificity=specificity,Precision=precision,F1=f1,
    BalancedAccuracy=mean(c(sensitivity,specificity),na.rm=TRUE))
}
score_tbl <- do.call(rbind,lapply(predictions,function(p) metrics(test_y,p,POSITIVE)))
score_tbl <- data.frame(Method=rownames(score_tbl),score_tbl,row.names=NULL,check.names=FALSE)
write.csv(score_tbl,"results/model_metrics.csv",row.names=FALSE,fileEncoding="UTF-8")
pred_csv <- data.frame(RowID=test_idx,Actual=as.character(test_y))
for (nm in names(predictions)) pred_csv[[nm]] <- as.character(predictions[[nm]])
write.csv(pred_csv,"results/test_predictions.csv",row.names=FALSE,fileEncoding="UTF-8")
confusion_rows <- lapply(names(predictions),function(nm) {
  cm <- table(Actual=factor(test_y,levels=c(NEGATIVE,POSITIVE)),
              Predicted=factor(predictions[[nm]],levels=c(NEGATIVE,POSITIVE)))
  z <- as.data.frame(cm)
  data.frame(Method=nm,z)
})
confusion_tbl <- do.call(rbind,confusion_rows)
write.csv(confusion_tbl,"results/confusion_matrices.csv",row.names=FALSE,fileEncoding="UTF-8")

# [10] 圖表：各模型 Accuracy / Sensitivity / Specificity 比較
png("results/metric_comparison.png",width=1050,height=690,res=125)
barplot(t(as.matrix(score_tbl[,c("Accuracy","Sensitivity","Specificity")])),
        beside=TRUE,ylim=c(0,1),names.arg=score_tbl$Method,
        col=c("#35769A","#C45B47","#67A692"),
        main="Test-set model comparison",ylab="Score")
legend("topright",legend=c("Accuracy","Sensitivity","Specificity"),
       fill=c("#35769A","#C45B47","#67A692"),bty="n")
dev.off()

# [11] 匯出文字摘要與完整混淆矩陣
report <- capture.output({
  cat("作業 01：乳癌良性與惡性分類\n")
  cat("資料來源：",file_in,"\n")
  cat("總筆數：",nrow(raw),"；訓練：",length(train_idx),"；測試：",length(test_idx),"\n")
  cat("類別分布：\n");print(class_tab,row.names=FALSE)
  cat("缺失值總數：",sum(missing_tab$Missing),"\n")
  cat("兩個特徵的描述統計：\n");print(eda_summary,row.names=FALSE)
  cat("訓練集絕對相關係數 > 0.8 的特徵配對數：",high_corr_pairs,"\n")
  cat("保留主成分數：",max_pc,"；累積解釋變異比例：",
      round(pca_tbl$CumulativeVariance[max_pc],4),"\n")
  cat("Linear SVM: cost=1；RBF SVM: cost=1, gamma=",1/max_pc,"\n")
  cat("\n測試集模型表現（M 為陽性）\n")
  print(score_tbl,row.names=FALSE,digits=4)
  for (nm in names(predictions)) {
    cat("\n",nm," 混淆矩陣（列：實際；欄：預測）\n",sep="")
    print(table(Actual=factor(test_y,levels=c(NEGATIVE,POSITIVE)),
                Predicted=factor(predictions[[nm]],levels=c(NEGATIVE,POSITIVE))))
  }
  cat("\n注意：FN 為惡性誤判良性，篩檢情境尤其值得注意。\n")
  cat("模型結果只代表此資料集的測試集表現，不能直接作臨床診斷。\n")
  cat("單次切分具有隨機變異；若調整 cost/gamma，須僅在訓練集交叉驗證。\n")
})
writeLines(report,"results/analysis_report.txt",useBytes=TRUE)
cat(paste(report,collapse="\n"),"\n")
cat("\n分析完成：CSV、PNG、TXT 已儲存至 results/\n")
