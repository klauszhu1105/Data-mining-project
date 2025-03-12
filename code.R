# Load required libraries
library(ROSE)
library(DMwR2)
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(class)

cat("Step 1: Loading necessary packages...\n")
required_packages <- c("caret", "randomForest", "rpart", "ggplot2", "dplyr", "rpart.plot", "MASS")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) install.packages(new_packages)
sapply(required_packages, require, character.only = TRUE)

cat("Step 2: Loading data...\n")
data <- read.csv("project_data.csv")

cat("Step 3: Data exploration...\n")
summary(data)
colSums(is.na(data))

cat("Step 4: Handling missing values...\n")

# Remove columns with more than 50% missing values
missing_threshold <- 0.5
missing_cols <- colnames(data)[colMeans(is.na(data)) > missing_threshold]
data <- data[, !(colnames(data) %in% missing_cols)]

# Fill missing values in numeric variables with the median
data <- data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

cat("Step 5: Detecting and handling outliers...\n")
# Identify and handle outliers
numeric_cols <- sapply(data, is.numeric)
data[numeric_cols] <- lapply(data[numeric_cols], function(x) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = TRUE)
  caps <- quantile(x, probs=c(.05, .95), na.rm = TRUE)
  H <- 1.5 * IQR(x, na.rm = TRUE)
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  return(x)
})

cat("Step 6: Feature engineering...\n")
# Convert character variables to factors
data <- data %>%
  mutate(across(where(is.character), as.factor))

# Remove low-variance columns
remove_low_variance_columns <- function(df) {
  df[, sapply(df, function(col) var(as.numeric(col), na.rm = TRUE) > 0.01)]
}
data_reduced <- remove_low_variance_columns(data)


cat("Step 7: Data normalization...\n")

# Refresh numeric column selection to ensure accurate scaling
numeric_cols <- sapply(data_reduced, is.numeric)
numeric_colnames <- names(data_reduced)[numeric_cols]

# Perform normalization if numeric columns exist
if (length(numeric_colnames) > 0) {
  data_reduced[, numeric_colnames] <- scale(data_reduced[, numeric_colnames])
} else {
  cat("No numeric columns found for normalization.\n")
}

str(data_reduced)

# Export processed data to CSV file
cat("Step 9: Exporting data to CSV...\n")
write.csv(data_reduced, "data_reduced.csv", row.names = FALSE)

cat("Data has been successfully exported to 'data_reduced.csv'!")



# Load necessary libraries
library(ROSE)
library(DMwR2)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(class)
library(openxlsx)


# Read the data
data <- read.csv("data_reduced.csv")  

# Convert the Class variable to a two-level factor
data$Class <- factor(ifelse(data$Class == "Y", 1, 0), levels = c(0, 1))

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$Class, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Export initial training and test data to CSV
write.csv(trainData, "initial_train.csv", row.names = FALSE)
write.csv(testData, "initial_test.csv", row.names = FALSE)

cat("The initial training and test datasets have been saved as 'initial_train.csv' and 'initial_test.csv' respectively.")

# Function to apply variance threshold
apply_variance_threshold <- function(data) {
  nzv <- nearZeroVar(data, saveMetrics = TRUE)
  filtered_data <- data[, !nzv$nzv]
  return(filtered_data)
}

apply_correlation_threshold <- function(data) {
  # Ensure there are numeric columns
  numeric_cols <- sapply(data, is.numeric)
  if (!any(numeric_cols)) {
    cat("No numeric columns found. Returning original data.\n")
    return(data)
  }
  
  # Compute correlation matrix
  corr_matrix <- cor(data[, numeric_cols])
  
  # Identify highly correlated features
  high_corr <- findCorrelation(corr_matrix, cutoff = 0.7)
  
  # Check if high_corr is empty
  if (length(high_corr) == 0) {
    cat("No highly correlated variables found. Returning original data.\n")
    return(data)
  }
  
  # Filter data
  filtered_data <- data[, -high_corr, drop = FALSE]
  
  # Ensure filtered_data retains the Class column
  if (!"Class" %in% colnames(filtered_data)) {
    filtered_data$Class <- data$Class
  }
  
  # Final check
  if (ncol(filtered_data) <= 1) {
    stop("All predictors were removed. Only Class column remains.")
  }
  
  return(filtered_data)
}

# Function to apply RFE
apply_rfe <- function(features, target) {
  # Ensure features are a data frame and target is a factor
  features <- as.data.frame(features)
  target <- as.factor(target)
  
  # Control parameters for RFE
  control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
  
  # Apply RFE to select important features
  rfe_result <- rfe(
    x = features,
    y = target,
    sizes = c(1:3),  # Adjust the sizes based on your data
    rfeControl = control
  )
  
  # Get the names of the selected features
  selected_features <- predictors(rfe_result)
  
  # Create a new data frame with only the selected features and the Class column
  filtered_data <- features[, selected_features, drop = FALSE]
  filtered_data$Class <- target  # Add the Class column back
  
  return(filtered_data)
}

# Function to calculate performance metrics and format the table
calculate_metrics <- function(actual, predicted, probabilities = NULL) {
  # Create confusion matrix
  table_confusion <- table(Predicted = predicted, Actual = actual)
  
  # Extract values from confusion matrix
  tn <- as.numeric(table_confusion["0", "0"] %||% 0) # True Negative
  fp <- as.numeric(table_confusion["1", "0"] %||% 0) # False Positive
  fn <- as.numeric(table_confusion["0", "1"] %||% 0) # False Negative
  tp <- as.numeric(table_confusion["1", "1"] %||% 0) # True Positive
  
  # Total samples
  total_samples <- sum(table_confusion)
  
  # Calculate MCC
  mcc <- if ((tp + fp) > 0 && (tp + fn) > 0 && (tn + fp) > 0 && (tn + fn) > 0) {
    (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  } else {
    NA
  }
  
  # Calculate Kappa
  observed_accuracy <- (tp + tn) / total_samples
  expected_accuracy <- ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / total_samples^2
  kappa <- if (!is.nan(expected_accuracy) && (1 - expected_accuracy) > 0) {
    (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
  } else {
    NA
  }
  
  # Calculate metrics for Class 0
  class_n_tpr <- if ((tn + fp) > 0) tn / (tn + fp) else NA  # TPR (Specificity)
  class_n_fpr <- if ((tn + fp) > 0) fp / (tn + fp) else NA  # FPR
  class_n_precision <- if ((tn + fn) > 0) tn / (tn + fn) else NA  # Precision
  class_n_recall <- class_n_tpr                           # Recall
  class_n_f1 <- if (!is.na(class_n_precision) && !is.na(class_n_recall) &&
                    (class_n_precision + class_n_recall) > 0) {
    2 * class_n_precision * class_n_recall /
      (class_n_precision + class_n_recall)
  } else {
    NA
  }
  
  # Calculate metrics for Class 1
  class_y_tpr <- if ((tp + fn) > 0) tp / (tp + fn) else NA  # TPR (Sensitivity)
  class_y_fpr <- if ((tp + fn) > 0) fn / (tp + fn) else NA  # FPR
  class_y_precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA  # Precision
  class_y_recall <- class_y_tpr                            # Recall
  class_y_f1 <- if (!is.na(class_y_precision) && !is.na(class_y_recall) &&
                    (class_y_precision + class_y_recall) > 0) {
    2 * class_y_precision * class_y_recall /
      (class_y_precision + class_y_recall)
  } else {
    NA
  }
  
  # Calculate weighted averages
  class_n_weight <- (tn + fp) / total_samples
  class_y_weight <- (tp + fn) / total_samples
  
  weighted_avg_metrics <- c(
    TPR = class_n_weight * (class_n_tpr %||% 0) + class_y_weight * (class_y_tpr %||% 0),
    FPR = class_n_weight * (class_n_fpr %||% 0) + class_y_weight * (class_y_fpr %||% 0),
    Precision = class_n_weight * (class_n_precision %||% 0) + class_y_weight * (class_y_precision %||% 0),
    Recall = class_n_weight * (class_n_recall %||% 0) + class_y_weight * (class_y_recall %||% 0),
    F1 = class_n_weight * (class_n_f1 %||% 0) + class_y_weight * (class_y_f1 %||% 0),
    ROC = if (!is.null(probabilities)) {
      roc_curve <- roc(as.numeric(as.character(actual)), probabilities)
      auc(roc_curve)
    } else {
      NA
    }
  )
  
  # Create the performance table in the desired format
  performance_table <- data.frame(
    Class = c("Class N", "Class Y", "Weighted Average"),
    TPR = c(class_n_tpr, class_y_tpr, weighted_avg_metrics["TPR"]),
    FPR = c(class_n_fpr, class_y_fpr, weighted_avg_metrics["FPR"]),
    Precision = c(class_n_precision, class_y_precision, weighted_avg_metrics["Precision"]),
    Recall = c(class_n_recall, class_y_recall, weighted_avg_metrics["Recall"]),
    `F-measure` = c(class_n_f1, class_y_f1, weighted_avg_metrics["F1"]),
    ROC = c(NA, NA, weighted_avg_metrics["ROC"]),
    MCC = c(mcc, mcc, mcc),
    Kappa = c(kappa, kappa, kappa)
  )
  
  return(list(
    confusion_matrix = table_confusion,
    performance_table = performance_table
  ))
}

# Manually creating combinations and saving them into an Excel
wb <- createWorkbook()
addCombinationToSheet <- function(wb, combination_metrics, combination_name) {
  addWorksheet(wb, combination_name)
  writeData(wb, sheet = combination_name, "Confusion Matrix:", startRow = 1, startCol = 1)
  writeData(wb, sheet = combination_name, combination_metrics$confusion_matrix, startRow = 2, startCol = 1)
  writeData(wb, sheet = combination_name, "Performance Table:", startRow = 6, startCol = 1)
  writeData(wb, sheet = combination_name, combination_metrics$performance_table, startRow = 7, startCol = 1)
}


# Combination 1: b1 + f1 + c1
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_variance_threshold(balanced_data)
model_c1 <- glm(Class ~ ., data = selected_data, family = "binomial")
predicted_c1 <- predict(model_c1, testData, type = "response")
predicted_c1_class <- factor(ifelse(predicted_c1 > 0.5, 1, 0), levels = c(0, 1))
metrics_c1 <- calculate_metrics(testData$Class, predicted_c1_class, predicted_c1)
print("Combination 1: b1 + f1 + c1")
print(metrics_c1$confusion_matrix)
print(metrics_c1$performance_table)
addCombinationToSheet(wb, metrics_c1, "Combination 1")

# Combination 2: b1 + f1 + c2
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_variance_threshold(balanced_data)
model_c2 <- train(Class ~ ., data = selected_data, method = "rpart")
predicted_c2 <- predict(model_c2, testData)
predicted_c2 <- factor(predicted_c2, levels = c(0, 1))  # Ensure levels match
metrics_c2 <- calculate_metrics(testData$Class, predicted_c2, NULL)
print("Combination 2: b1 + f1 + c2")
print(metrics_c2$confusion_matrix)
print(metrics_c2$performance_table)
addCombinationToSheet(wb, metrics_c2, "Combination 2")

# Combination 3: b1 + f1 + c3
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_variance_threshold(balanced_data)
model_c3 <- randomForest(Class ~ ., data = selected_data)
predicted_c3 <- predict(model_c3, testData)
predicted_c3 <- factor(predicted_c3, levels = c(0, 1))  # Ensure levels match
metrics_c3 <- calculate_metrics(testData$Class, predicted_c3, NULL)
print("Combination 3: b1 + f1 + c3")
print(metrics_c3$confusion_matrix)
print(metrics_c3$performance_table)
addCombinationToSheet(wb, metrics_c3, "Combination 3")


# Combination 4: b1 + f1 + c4
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_variance_threshold(balanced_data)
model_c4 <- svm(Class ~ ., data = selected_data)
predicted_c4 <- predict(model_c4, testData)
predicted_c4 <- factor(predicted_c4, levels = c(0, 1))  # Ensure levels match
metrics_c4 <- calculate_metrics(testData$Class, predicted_c4, NULL)
print("Combination 4: b1 + f1 + c4")
print(metrics_c4$confusion_matrix)
print(metrics_c4$performance_table)
addCombinationToSheet(wb, metrics_c4, "Combination 4")


# Combination 5: b1 + f1 + c5
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_variance_threshold(balanced_data)
model_c5 <- train(Class ~ ., data = selected_data, method = "knn")
predicted_c5 <- predict(model_c5, testData)
predicted_c5 <- factor(predicted_c5, levels = c(0, 1))  # Ensure levels match
metrics_c5 <- calculate_metrics(testData$Class, predicted_c5, NULL)
print("Combination 5: b1 + f1 + c5")
print(metrics_c5$confusion_matrix)
print(metrics_c5$performance_table)
addCombinationToSheet(wb, metrics_c5, "Combination 5")


# Combination 6: b1 + f1 + c6
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_variance_threshold(balanced_data)
model_c6 <- naiveBayes(Class ~ ., data = selected_data)
predicted_c6 <- predict(model_c6, testData)
probabilities_c6 <- predict(model_c6, testData, type = "raw")[, 2]
predicted_c6 <- factor(predicted_c6, levels = c(0, 1))  # Ensure levels match
metrics_c6 <- calculate_metrics(testData$Class, predicted_c6, probabilities_c6)
print("Combination 6: b1 + f1 + c6")
print(metrics_c6$confusion_matrix)
print(metrics_c6$performance_table)
addCombinationToSheet(wb, metrics_c6, "Combination 6")


# Combination 7: b1 + f2 + c1
# Apply undersampling
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data

# Apply correlation threshold for feature selection
selected_data <- apply_correlation_threshold(balanced_data)

# Train logistic regression model
model_c1 <- glm(Class ~ ., data = selected_data, family = "binomial")

# Make predictions on the test set
predicted_c1 <- predict(model_c1, testData, type = "response")
predicted_c1_class <- factor(ifelse(predicted_c1 > 0.5, 1, 0), levels = c(0, 1))

# Calculate and print performance metrics
metrics_c1 <- calculate_metrics(testData$Class, predicted_c1_class, predicted_c1)
print("Combination 7: b1 + f2 + c1")
print(metrics_c1$confusion_matrix)
print(metrics_c1$performance_table)
addCombinationToSheet(wb, metrics_c1, "Combination 7")


# Combination 8: b1 + f2 + c2
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c2 <- train(Class ~ ., data = selected_data, method = "rpart")
predicted_c2 <- predict(model_c2, testData)
predicted_c2 <- factor(predicted_c2, levels = c(0, 1))
metrics_c2 <- calculate_metrics(testData$Class, predicted_c2, NULL)
print("Combination 8: b1 + f2 + c2")
print(metrics_c2$confusion_matrix)
print(metrics_c2$performance_table)
addCombinationToSheet(wb, metrics_c2, "Combination 8")


# Combination 9: b1 + f2 + c3
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c3 <- randomForest(Class ~ ., data = selected_data)
predicted_c3 <- predict(model_c3, testData)
predicted_c3 <- factor(predicted_c3, levels = c(0, 1))
metrics_c3 <- calculate_metrics(testData$Class, predicted_c3, NULL)
print("Combination 9: b1 + f2 + c3")
print(metrics_c3$confusion_matrix)
print(metrics_c3$performance_table)
addCombinationToSheet(wb, metrics_c3, "Combination 9")


# Combination 10: b1 + f2 + c4
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c4 <- svm(Class ~ ., data = selected_data)
predicted_c4 <- predict(model_c4, testData)
predicted_c4 <- factor(predicted_c4, levels = c(0, 1))
metrics_c4 <- calculate_metrics(testData$Class, predicted_c4, NULL)
print("Combination 10: b1 + f2 + c4")
print(metrics_c4$confusion_matrix)
print(metrics_c4$performance_table)
addCombinationToSheet(wb, metrics_c4, "Combination 10")


# Combination 11: b1 + f2 + c5
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c5 <- train(Class ~ ., data = selected_data, method = "knn")
predicted_c5 <- predict(model_c5, testData)
predicted_c5 <- factor(predicted_c5, levels = c(0, 1))
metrics_c5 <- calculate_metrics(testData$Class, predicted_c5, NULL)
print("Combination 11: b1 + f2 + c5")
print(metrics_c5$confusion_matrix)
print(metrics_c5$performance_table)
addCombinationToSheet(wb, metrics_c5, "Combination 11")


# Combination 12: b1 + f2 + c6
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c6 <- naiveBayes(Class ~ ., data = selected_data)
predicted_c6 <- predict(model_c6, testData)
probabilities_c6 <- predict(model_c6, testData, type = "raw")[, 2]
predicted_c6 <- factor(predicted_c6, levels = c(0, 1))
metrics_c6 <- calculate_metrics(testData$Class, predicted_c6, probabilities_c6)
print("Combination 12: b1 + f2 + c6")
print(metrics_c6$confusion_matrix)
print(metrics_c6$performance_table)
addCombinationToSheet(wb, metrics_c6, "Combination 12")


# Combination 13: b1 + f3 + c1
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c1 <- glm(Class ~ ., data = selected_data, family = "binomial")
predicted_c1 <- predict(model_c1, testData, type = "response")
predicted_c1_class <- factor(ifelse(predicted_c1 > 0.5, 1, 0), levels = c(0, 1))
metrics_c1 <- calculate_metrics(testData$Class, predicted_c1_class, predicted_c1)
print("Combination 13: b1 + f3 + c1")
print(metrics_c1$confusion_matrix)
print(metrics_c1$performance_table)
addCombinationToSheet(wb, metrics_c1, "Combination 13")


# Combination 14: b1 + f3 + c2
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c2 <- caret::train(Class ~ ., data = selected_data, method = "rpart")
predicted_c2 <- predict(model_c2, testData)
predicted_c2 <- factor(predicted_c2, levels = c(0, 1))
metrics_c2 <- calculate_metrics(testData$Class, predicted_c2, NULL)
print("Combination 14: b1 + f3 + c2")
print(metrics_c2$confusion_matrix)
print(metrics_c2$performance_table)
addCombinationToSheet(wb, metrics_c2, "Combination 14")


# Combination 15: b1 + f3 + c3
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c3 <- randomForest(Class ~ ., data = selected_data)
predicted_c3 <- predict(model_c3, testData)
predicted_c3 <- factor(predicted_c3, levels = c(0, 1))
metrics_c3 <- calculate_metrics(testData$Class, predicted_c3, NULL)
print("Combination 15: b1 + f3 + c3")
print(metrics_c3$confusion_matrix)
print(metrics_c3$performance_table)
addCombinationToSheet(wb, metrics_c3, "Combination 15")


# Combination 16: b1 + f3 + c4
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c4 <- svm(Class ~ ., data = selected_data)
predicted_c4 <- predict(model_c4, testData)
predicted_c4 <- factor(predicted_c4, levels = c(0, 1))
metrics_c4 <- calculate_metrics(testData$Class, predicted_c4, NULL)
print("Combination 16: b1 + f3 + c4")
print(metrics_c4$confusion_matrix)
print(metrics_c4$performance_table)
addCombinationToSheet(wb, metrics_c4, "Combination 16")


# Combination 17: b1 + f3 + c5
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c5 <- caret::train(Class ~ ., data = selected_data, method = "knn")
predicted_c5 <- predict(model_c5, testData)
predicted_c5 <- factor(predicted_c5, levels = c(0, 1))
metrics_c5 <- calculate_metrics(testData$Class, predicted_c5, NULL)
print("Combination 17: b1 + f3 + c5")
print(metrics_c5$confusion_matrix)
print(metrics_c5$performance_table)
addCombinationToSheet(wb, metrics_c5, "Combination 17")


# Combination 18: b1 + f3 + c6
balanced_data <- ovun.sample(Class ~ ., data = trainData, method = "under")$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c6 <- naiveBayes(Class ~ ., data = selected_data)
predicted_c6 <- predict(model_c6, testData)
probabilities_c6 <- predict(model_c6, testData, type = "raw")[, 2]
predicted_c6 <- factor(predicted_c6, levels = c(0, 1))
metrics_c6 <- calculate_metrics(testData$Class, predicted_c6, probabilities_c6)
print("Combination 18: b1 + f3 + c6")
print(metrics_c6$confusion_matrix)
print(metrics_c6$performance_table)
addCombinationToSheet(wb, metrics_c6, "Combination 18")


# Combination 19: b2 + f1 + c1
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_variance_threshold(balanced_data)
model_c1 <- glm(Class ~ ., data = selected_data, family = "binomial")
predicted_c1 <- predict(model_c1, testData, type = "response")
predicted_c1_class <- factor(ifelse(predicted_c1 > 0.5, 1, 0), levels = c(0, 1))
metrics_c1 <- calculate_metrics(testData$Class, predicted_c1_class, predicted_c1)
print("Combination 19: b2 + f1 + c1 using ROSE")
print(metrics_c1$confusion_matrix)
print(metrics_c1$performance_table)
addCombinationToSheet(wb, metrics_c1, "Combination 19")


# Combination 20: b2 (ROSE) + f1 + c2
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_variance_threshold(balanced_data)
model_c2 <- caret::train(Class ~ ., data = selected_data, method = "rpart")
predicted_c2 <- predict(model_c2, testData)
metrics_c2 <- calculate_metrics(testData$Class, predicted_c2, NULL)
print("Combination 20: b2 (ROSE) + f1 + c2")
print(metrics_c2$confusion_matrix)
print(metrics_c2$performance_table)
addCombinationToSheet(wb, metrics_c2, "Combination 20")


# Combination 21: b2 (ROSE) + f1 + c3
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_variance_threshold(balanced_data)
model_c3 <- randomForest(Class ~ ., data = selected_data)
predicted_c3 <- predict(model_c3, testData)
metrics_c3 <- calculate_metrics(testData$Class, predicted_c3, NULL)
print("Combination 21: b2 (ROSE) + f1 + c3")
print(metrics_c3$confusion_matrix)
print(metrics_c3$performance_table)
addCombinationToSheet(wb, metrics_c3, "Combination 21")


# Combination 22: b2 (ROSE) + f1 + c4
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_variance_threshold(balanced_data)
model_c4 <- svm(Class ~ ., data = selected_data)
predicted_c4 <- predict(model_c4, testData)
metrics_c4 <- calculate_metrics(testData$Class, predicted_c4, NULL)
print("Combination 22: b2 (ROSE) + f1 + c4")
print(metrics_c4$confusion_matrix)
print(metrics_c4$performance_table)
addCombinationToSheet(wb, metrics_c4, "Combination 22")


# Combination 23: b2 (ROSE) + f1 + c5
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_variance_threshold(balanced_data)
model_c5 <- caret::train(Class ~ ., data = selected_data, method = "knn")
predicted_c5 <- predict(model_c5, testData)
metrics_c5 <- calculate_metrics(testData$Class, predicted_c5, NULL)
print("Combination 23: b2 (ROSE) + f1 + c5")
print(metrics_c5$confusion_matrix)
print(metrics_c5$performance_table)
addCombinationToSheet(wb, metrics_c5, "Combination 23")


# Combination 24: b2 (ROSE) + f1 + c6
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_variance_threshold(balanced_data)
model_c6 <- naiveBayes(Class ~ ., data = selected_data)
predicted_c6 <- predict(model_c6, testData)
probabilities_c6 <- predict(model_c6, testData, type = "raw")[, 2]
metrics_c6 <- calculate_metrics(testData$Class, predicted_c6, probabilities_c6)
print("Combination 24: b2 (ROSE) + f1 + c6")
print(metrics_c6$confusion_matrix)
print(metrics_c6$performance_table)
addCombinationToSheet(wb, metrics_c6, "Combination 24")


# Combination 25: b2 (ROSE) + f2 + c1
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c1 <- glm(Class ~ ., data = selected_data, family = "binomial")
predicted_c1 <- predict(model_c1, testData, type = "response")
predicted_c1_class <- factor(ifelse(predicted_c1 > 0.5, 1, 0), levels = c(0, 1))
metrics_c1 <- calculate_metrics(testData$Class, predicted_c1_class, predicted_c1)
print("Combination 25: b2 (ROSE) + f2 + c1")
print(metrics_c1$confusion_matrix)
print(metrics_c1$performance_table)
addCombinationToSheet(wb, metrics_c1, "Combination 25")


# Combination 26: b2 (ROSE) + f2 + c2
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c2 <- caret::train(Class ~ ., data = selected_data, method = "rpart")
predicted_c2 <- predict(model_c2, testData)
metrics_c2 <- calculate_metrics(testData$Class, predicted_c2, NULL)
print("Combination 26: b2 (ROSE) + f2 + c2")
print(metrics_c2$confusion_matrix)
print(metrics_c2$performance_table)
addCombinationToSheet(wb, metrics_c2, "Combination 26")


# Combination 27: b2 (ROSE) + f2 + c3
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c3 <- randomForest(Class ~ ., data = selected_data)
predicted_c3 <- predict(model_c3, testData)
metrics_c3 <- calculate_metrics(testData$Class, predicted_c3, NULL)
print("Combination 27: b2 (ROSE) + f2 + c3")
print(metrics_c3$confusion_matrix)
print(metrics_c3$performance_table)
addCombinationToSheet(wb, metrics_c3, "Combination 27")


# Combination 28: b2 (ROSE) + f2 + c4
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c4 <- svm(Class ~ ., data = selected_data)
predicted_c4 <- predict(model_c4, testData)
metrics_c4 <- calculate_metrics(testData$Class, predicted_c4, NULL)
print("Combination 28: b2 (ROSE) + f2 + c4")
print(metrics_c4$confusion_matrix)
print(metrics_c4$performance_table)
addCombinationToSheet(wb, metrics_c4, "Combination 28")


# Combination 29: b2 (ROSE) + f2 + c5
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c5 <- caret::train(Class ~ ., data = selected_data, method = "knn")
predicted_c5 <- predict(model_c5, testData)
metrics_c5 <- calculate_metrics(testData$Class, predicted_c5, NULL)
print("Combination 29: b2 (ROSE) + f2 + c5")
print(metrics_c5$confusion_matrix)
print(metrics_c5$performance_table)
addCombinationToSheet(wb, metrics_c5, "Combination 29")


# Combination 30: b2 (ROSE) + f2 + c6
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_correlation_threshold(balanced_data)
model_c6 <- naiveBayes(Class ~ ., data = selected_data)
predicted_c6 <- predict(model_c6, testData)
probabilities_c6 <- predict(model_c6, testData, type = "raw")[, 2]
metrics_c6 <- calculate_metrics(testData$Class, predicted_c6, probabilities_c6)
print("Combination 30: b2 (ROSE) + f2 + c6")
print(metrics_c6$confusion_matrix)
print(metrics_c6$performance_table)
addCombinationToSheet(wb, metrics_c6, "Combination 30")


# Combination 31: b2 (ROSE) + f3 + c1
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c1 <- glm(Class ~ ., data = selected_data, family = "binomial")
predicted_c1 <- predict(model_c1, testData, type = "response")
predicted_c1_class <- factor(ifelse(predicted_c1 > 0.5, 1, 0), levels = c(0, 1))
metrics_c1 <- calculate_metrics(testData$Class, predicted_c1_class, predicted_c1)
print("Combination 31: b2 (ROSE) + f3 + c1")
print(metrics_c1$confusion_matrix)
print(metrics_c1$performance_table)
addCombinationToSheet(wb, metrics_c1, "Combination 31")


# Combination 32: b2 (ROSE) + f3 + c2
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c2 <- train(Class ~ ., data = selected_data, method = "rpart")
predicted_c2 <- predict(model_c2, testData)
metrics_c2 <- calculate_metrics(testData$Class, predicted_c2, NULL)
print("Combination 32: b2 (ROSE) + f3 + c2")
print(metrics_c2$confusion_matrix)
print(metrics_c2$performance_table)
addCombinationToSheet(wb, metrics_c2, "Combination 32")


# Combination 33: b2 (ROSE) + f3 + c3
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c3 <- randomForest(Class ~ ., data = selected_data)
predicted_c3 <- predict(model_c3, testData)
metrics_c3 <- calculate_metrics(testData$Class, predicted_c3, NULL)
print("Combination 33: b2 (ROSE) + f3 + c3")
print(metrics_c3$confusion_matrix)
print(metrics_c3$performance_table)
addCombinationToSheet(wb, metrics_c3, "Combination 33")


# Combination 34: b2 (ROSE) + f3 + c4
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c4 <- svm(Class ~ ., data = selected_data)
predicted_c4 <- predict(model_c4, testData)
metrics_c4 <- calculate_metrics(testData$Class, predicted_c4, NULL)
print("Combination 34: b2 (ROSE) + f3 + c4")
print(metrics_c4$confusion_matrix)
print(metrics_c4$performance_table)
addCombinationToSheet(wb, metrics_c4, "Combination 34")


# Combination 35: b2 (ROSE) + f3 + c5
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c5 <- train(Class ~ ., data = selected_data, method = "knn")
predicted_c5 <- predict(model_c5, testData)
metrics_c5 <- calculate_metrics(testData$Class, predicted_c5, NULL)
print("Combination 35: b2 (ROSE) + f3 + c5")
print(metrics_c5$confusion_matrix)
print(metrics_c5$performance_table)
addCombinationToSheet(wb, metrics_c5, "Combination 35")


# Combination 36: b2 (ROSE) + f3 + c6
balanced_data <- ROSE(Class ~ ., data = trainData, seed = 123)$data
selected_data <- apply_rfe(balanced_data[, -ncol(balanced_data)], balanced_data$Class)
model_c6 <- naiveBayes(Class ~ ., data = selected_data)
predicted_c6 <- predict(model_c6, testData)
probabilities_c6 <- predict(model_c6, testData, type = "raw")[, 2]
metrics_c6 <- calculate_metrics(testData$Class, predicted_c6, probabilities_c6)
print("Combination 36: b2 (ROSE) + f3 + c6")
print(metrics_c6$confusion_matrix)
print(metrics_c6$performance_table)
addCombinationToSheet(wb, metrics_c6, "Combination 36")

# Save the workbook
saveWorkbook(wb, "Confusion_Performance_Metrics.xlsx", overwrite = TRUE)