# ===========================================================================
# ANN 學生作業（二）糖尿病一年後病程指標（連續型迴歸）
# 資料來源：scikit-learn Diabetes dataset (raw unscaled feature option)，n=442
# 重點：Linear Regression versus ANN with linear output; validate architectures
# 注意：僅作統計建模教材，不作臨床預測或醫療判斷。
# ===========================================================================
# 設定工作目錄
setwd("C:\\02_糖尿病病程進展")

# 確認目前工作目錄
getwd()

# 查看資料夾中的檔案
list.files()

# 讀取糖尿病資料
d <- read.csv("diabetes_progression.csv")
if (!requireNamespace("nnet", quietly = TRUE)) {
  stop("尚未安裝 nnet 套件。請先執行 install.packages('nnet')。")
}
set.seed(20261025)
candidates <- c("diabetes_progression.csv",
                "02_學生作業_糖尿病迴歸/diabetes_progression.csv",
                "ANN_雙實例教學專案/02_學生作業_糖尿病迴歸/diabetes_progression.csv",
                "../02_學生作業_糖尿病迴歸/diabetes_progression.csv")
getwd()
list.files(recursive = TRUE, pattern = "diabetes_progression\\.csv$")
hits <- candidates[file.exists(candidates)]
if (!length(hits)) stop("找不到糖尿病 CSV，請將工作目錄設為專案根目錄或案例二資料夾。")
d <- read.csv(hits[1], check.names = FALSE)
features <- c("age", "sex", "bmi", "bp", paste0("s", 1:6))
stopifnot(all(c(features,"progression") %in% names(d)))
stopifnot(!anyNA(d[, c(features,"progression")]))
dir.create("Results_糖尿病迴歸", showWarnings = FALSE)
out <- "Results_糖尿病迴歸"

# [任務一] 描述資料（共 442 筆；因變數為一年後病程進展的數值指標）
cat("資料筆數：", nrow(d), "，預測變數：",length(features),"\n")
print(summary(d[, c("bmi", "bp", "s5", "progression")]))

# [任務二] 固定隨機種子切分 60% train, 20% validation, 20% test
idx <- sample(seq_len(nrow(d)))
n_train <- floor(.70 * nrow(d)); n_valid <- floor(.15 * nrow(d))
id_train <- idx[seq_len(n_train)]
id_valid <- idx[seq.int(n_train + 1, n_train + n_valid)]
id_test  <- idx[seq.int(n_train + n_valid + 1, length(idx))]
train_raw <- d[id_train, , drop = FALSE]
valid_raw <- d[id_valid, , drop = FALSE]
test_raw  <- d[id_test, , drop = FALSE]
cat("train/validation/test:", nrow(train_raw), nrow(valid_raw), nrow(test_raw), "\n")

# [任務三] 僅由 train 估計 X 的縮放參數；y 的縮放也僅看 train
x_mean <- vapply(train_raw[features], mean, numeric(1))
x_sd <- vapply(train_raw[features], sd, numeric(1))
x_sd[x_sd == 0] <- 1
y_mean <- mean(train_raw$progression)
y_sd <- sd(train_raw$progression)
scale_x <- function(raw) {
  z <- sweep(as.matrix(raw[, features, drop = FALSE]), 2, x_mean, "-")
  z <- sweep(z, 2, x_sd, "/")
  as.data.frame(z)
}
train_x <- scale_x(train_raw);valid_x <- scale_x(valid_raw);test_x <- scale_x(test_raw)
train_nn <- data.frame(train_x, y=(train_raw$progression-y_mean)/y_sd)
valid_nn <- data.frame(valid_x)

# [任務四] Linear Regression：在 train 擬合並看 validation RMSE
train_lm <- data.frame(train_x, y=train_raw$progression)
valid_lm <- data.frame(valid_x)
lm_fit <- lm(y ~ ., data=train_lm)
lm_val <- as.numeric(predict(lm_fit, newdata=valid_lm))
rmse <- function(y, p) sqrt(mean((y - p)^2))
mae  <- function(y, p) mean(abs(y - p))
r2_test <- function(y,p) 1-sum((y-p)^2)/sum((y-mean(y))^2)
cat("Linear model validation RMSE:",rmse(valid_raw$progression,lm_val),"\n")

# [任務五] 以 validation 資料選 ANN 架構，避免在 test 上調參
# nnet() 只有單一隱藏層；linout=TRUE 是連續輸出，非 Sigmoid 輸出。
sizes <- c(3, 6, 10)
val_table <- data.frame(Hidden_Units = sizes, Validation_RMSE = NA_real_)
for (i in seq_along(sizes)) {
  set.seed(900+i)
  ann_tmp <- nnet::nnet(y ~ ., data=train_nn, size=sizes[i], linout=TRUE,
                        decay=.01, maxit=900, MaxNWts=1000, trace=FALSE)
  pred_std <- as.numeric(predict(ann_tmp, newdata=valid_nn, type="raw"))
  pred_raw <- pred_std*y_sd+y_mean
  val_table$Validation_RMSE[i] <- rmse(valid_raw$progression,pred_raw)
}
print(val_table)
write.csv(val_table,file.path(out,"01_ann_validation_model_selection.csv"),row.names=FALSE)
best_size <- val_table$Hidden_Units[which.min(val_table$Validation_RMSE)]
cat("Selected hidden units:",best_size,"(selection uses validation only)\n")

# [任務六] 僅在完成架構選擇後，以 train+validation 重新估計兩個模型
# 此處沿用 train-only 的 X / y 縮放參數，避免更動驗證選模設定。
development_raw <- rbind(train_raw, valid_raw)
development_x <- scale_x(development_raw)
development_lm <- data.frame(development_x, y=development_raw$progression)
development_nn <- data.frame(development_x,
                             y=(development_raw$progression-y_mean)/y_sd)
test_lm <- data.frame(test_x)
fit_linear_final <- lm(y ~ ., data=development_lm)
set.seed(20261026)
fit_ann_final <- nnet::nnet(y ~ .,data=development_nn,size=best_size,
                            linout=TRUE,decay=.01,maxit=900,
                            MaxNWts=1000,trace=FALSE)

# [任務七] 鎖定測試集，僅評估一次最終模型
actual <- test_raw$progression
pred_linear <- as.numeric(predict(fit_linear_final,newdata=test_lm))
pred_ann <- as.numeric(predict(fit_ann_final,newdata=test_lm,type="raw"))*y_sd+y_mean
metrics <- data.frame(Model=c("Linear Regression","ANN"),
                      RMSE=c(rmse(actual,pred_linear),rmse(actual,pred_ann)),
                      MAE=c(mae(actual,pred_linear),mae(actual,pred_ann)),
                      Test_R2=c(r2_test(actual,pred_linear),r2_test(actual,pred_ann)))
print(metrics)
write.csv(metrics,file.path(out,"02_heldout_test_metrics.csv"),row.names=FALSE)
write.csv(data.frame(Actual=actual,Linear=pred_linear,ANN=pred_ann),
          file.path(out,"03_test_predictions.csv"),row.names=FALSE)

# [任務八] 預測值與實際值比較（虛線為完美預測 y=x）
png(file.path(out,"04_observed_vs_predicted.png"),width=1100,height=740,res=145)
par(mfrow=c(1,2))
lims <- range(c(actual,pred_linear,pred_ann))
plot(actual,pred_linear,pch=16,cex=.65,col="#3A889E",
     xlab="Observed progression",ylab="Predicted progression",
     xlim=lims,ylim=lims,main="Linear Regression")
abline(0,1,lty=2,col="gray50",lwd=2)
plot(actual,pred_ann,pch=16,cex=.65,col="#D07860",
     xlab="Observed progression",ylab="Predicted progression",
     xlim=lims,ylim=lims,main="ANN")
abline(0,1,lty=2,col="gray50",lwd=2)
invisible(dev.off())

png(file.path(out,"05_residuals.png"),width=1100,height=740,res=145)
par(mfrow=c(1,2))
plot(pred_linear,actual-pred_linear,pch=16,cex=.65,col="#3A889E",
     xlab="Fitted",ylab="Residual",main="Linear residuals")
abline(h=0,lty=2)
plot(pred_ann,actual-pred_ann,pch=16,cex=.65,col="#D07860",
     xlab="Fitted",ylab="Residual",main="ANN residuals")
abline(h=0,lty=2)
invisible(dev.off())

# [任務九] 保存模型與研究摘要；變數 s1-s6 為生理檢測項目，請勿過度臨床解讀
saveRDS(list(Linear=fit_linear_final,ANN=fit_ann_final,hidden=best_size,
             mean_x=x_mean,sd_x=x_sd,mean_y=y_mean,sd_y=y_sd,
             features=features,seed=20261025),file.path(out,"06_models.rds"))
writeLines(c("ANN 迴歸作業參考成果", "Train / valid / test = 60% / 20% / 20%",
             "調整 ANN 架構只使用 validation，未使用 test",
             "驗證集架構比較",capture.output(print(val_table)),
             "最終測試集評估",capture.output(print(metrics)),
             "本研究僅適用統計教學，非臨床預測工具。"),
           file.path(out,"00_結果摘要.txt"))
cat("分析完成。結果在：",normalizePath(out),"\n")
