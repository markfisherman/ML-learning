
##################### Load Packages (For Analysis) ####################

## Libraries Path
.libPaths()
.libPaths("C:\\Users\\USER\\Downloads")
.libPaths()

## Install Packages (Run Once)
# install.packages(c("rpart", "rpart.plot", "caTools",
#                    "ggplot2", "GGally", "gridExtra"))

## Load Packages
library(rpart)
library(rpart.plot)
library(caTools)
library(ggplot2)
library(GGally)
library(gridExtra)


#############################################################################
## Baseball Data (For Regression)
#############################################################################

##################### Load Data #####################

## Set Working Directory
setwd("C:\\Users\\USER\\Downloads")

## Read Data
ball <- read.table("ball.txt",
                   header = TRUE,
                   sep = "",
                   stringsAsFactors = TRUE)

## Remove Id, Name, Salary
ball <- subset(ball, select = -c(Id, Name, Salary))


##################### Descriptive Statistics #####################

## Check Data
dim(ball)
summary(ball)
str(ball)
head(ball, n = 10)

## Check Missing Values
colSums(is.na(ball))

## Remove Missing Values
ball <- na.omit(ball)


##################### Data Visualization #####################

## Boxplot of Numerical Variables
numeric_data <- ball[, sapply(ball, is.numeric)]

boxplot(numeric_data,
        col = rainbow(ncol(numeric_data)),
        main = "Boxplot of Baseball Data",
        las = 2)

## Scatter Plot Matrix
## Select Important Variables
ggpairs(ball,
        columns = c("Bat86", "Hit86", "Hr86",
                    "Run86", "Rb86", "Logsalary")) +
  theme_bw()

## Boxplot of Logsalary by League
p1 <- ggplot(ball, aes(x = League86, y = Logsalary)) +
  geom_boxplot(aes(fill = League86)) +
  labs(title = "Logsalary by League86") +
  theme_bw()

## Boxplot of Logsalary by Division
p2 <- ggplot(ball, aes(x = Div86, y = Logsalary)) +
  geom_boxplot(aes(fill = Div86)) +
  labs(title = "Logsalary by Division") +
  theme_bw()

## Scatter Plot of Hits and Logsalary
p3 <- ggplot(ball, aes(x = Hit86, y = Logsalary)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Hits vs Logsalary") +
  theme_bw()

## Scatter Plot of Home Runs and Logsalary
p4 <- ggplot(ball, aes(x = Hr86, y = Logsalary)) +
  geom_point(color = "red") +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Home Runs vs Logsalary") +
  theme_bw()

## Arrange Plots
grid.arrange(p1, p2, p3, p4,
             ncol = 2, nrow = 2)


#############################################################################
## Training Data and Testing Data
#############################################################################

## Set Random Seed
set.seed(123)

## Split Data (80% Training, 20% Testing)
split <- sample.split(ball$Logsalary, SplitRatio = 0.8)

train_data <- subset(ball, split == TRUE)
test_data <- subset(ball, split == FALSE)

## Check Dimensions
dim(train_data)
dim(test_data)

