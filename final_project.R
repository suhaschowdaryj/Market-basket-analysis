#Statistical Machine learning, Math-574M Final project
#Instacart Market Basket Analysis
#Objective :To build a recommender system to predict the reordered items.
#Data set : https://www.instacart.com/datasets/grocery-shopping-2017
#The data set consists of over 3.4 million orders from over 200,000 customers distributed over6 datasets.
#Requirements : 16GB RAM or higher 

#libraries
library(xgboost)
library(data.table)
library(stringr)
library(dplyr)
library(tidyr)
library(knitr)
library(DT)
library(lattice)
library(caret)
library(adabag)
library(treemap)
library(ggplot2)
library(MLmetrics)
library(kableExtra)

# Load Data 
prior_orders <- fread('order_products__prior.csv')
train_orders <- fread('order_products__train.csv')
orders       <- fread('orders.csv')
products     <- fread('products.csv')
aisles       <- fread('aisles.csv')
departments  <- fread('departments.csv')

#Data preprocessing
products     <- products %>%
                mutate(product_name = as.factor(product_name))
orders       <- orders %>% 
                mutate(eval_set = as.factor(eval_set))
departments  <- departments %>% 
                mutate(department = as.factor(department))
aisles       <- aisles %>% 
                mutate(aisle = as.factor(aisle))

#Outlook of the data
kable(head(aisles,3))
kable(head(departments,3))
kable(head(prior_orders,3))
kable(head(train_orders,3))
kable(head(orders,3))
kable(head(products,3))

#merge products, aisles and departments by aisle_id,department_id
products       <- merge(products, aisles, by = "aisle_id")
products       <- merge(products, departments, by = "department_id")
train_orders   <- train_orders %>%
                  mutate(user_id = orders$user_id[match(train_orders$order_id,orders$user_id)])
product_orders <- orders %>% 
                  inner_join(prior_orders, by = "order_id")
rm(prior_orders)
gc()

###### Exploratory Data analysis ######

#Overview of data distibution 
aisle_map     <- products %>% 
                 group_by(department_id,aisle_id) %>% 
                 dplyr::summarize(n1=n())
aisle_map     <- aisle_map %>% 
                 left_join(departments,by="department_id")
aisle_map     <- aisle_map %>% 
                 left_join(aisles,by="aisle_id")

treemap(aisle_map,index=c("department","aisle"),vSize = "n1",title="",palette="Set3",border.col="#FFFFFF")
products      <- within(products,rm(department_id,aisle_id))
rm(departments,aisles,aisle_map)

#When people reorder again: number of days since reorder 
#It is found that majority of the people reorder again exactly after 1 week or 1 month.
dspo          <- orders$days_since_prior_order
dspo          <- as.data.frame(dspo)
ggplot(data = dspo, aes(dspo))+
  geom_histogram(stat="count",fill="blue")+
  xlab("Days since reorder")
rm(dspo)
gc()

#59% of the time,people order the same items again
reorders <- as.data.frame(train_orders$reordered)
ggplot(data = reorders, aes(reorders))+
  geom_histogram(stat="count",fill="blue")+
  xlab("People who reorder same item")+
  scale_x_continuous(breaks = round(seq(0,1, by = 1),1))
rm(reorders)
gc()

#Relation between the probability of reorder and time of last order:
#More likely people reorder the same product if they order on same day.
#People try out new things if there is more than 30 days gap.
temp    <- train_orders %>% 
           left_join(orders,by="order_id") %>% 
           group_by(days_since_prior_order) %>%
           dplyr ::summarize(mean_reorder = mean(reordered))

temp %>%
  ggplot(aes(x=days_since_prior_order,y=mean_reorder))+
  geom_bar(stat="identity",fill="blue")
rm(temp)
gc()

#feature engineering 

##User based features 
user_features <- orders %>% filter(eval_set == "prior") %>%
                 group_by(user_id) %>%
                 dplyr::summarise( user_number_of_orders = max(order_number),
                 average_timegap  = mean(days_since_prior_order, na.rm = T),
                 timegap_of_orders= sum(days_since_prior_order, na.rm = T))

product_temp  <- product_orders %>% group_by(user_id) %>%
                 dplyr::summarise(
                 number_of_purchases = n(),
                 up_unique_products  = n_distinct(product_id),
                 user_mean_dow       = round(mean(order_dow, na.rm = TRUE)),
                 repeated_orders_ratio = sum(reordered == 1) / sum(order_number > 1))

user_features <- user_features %>%
                 inner_join(product_temp)
user_features$capacity <- user_features$number_of_purchases / user_features$user_number_of_orders

product_temp  <- orders %>%
                 filter(eval_set != "prior") %>%
                 select(eval_set,user_id, order_id,
                 gap_between_orders = days_since_prior_order)

user_features <- user_features %>% inner_join(product_temp)
rm(product_temp)
gc()

# products based features
product_features <- product_orders %>%
                    arrange(user_id, order_number, product_id) %>%
                    group_by(user_id, product_id) %>%
                    mutate(time_gap = row_number())

product_features <- product_features %>% 
                    group_by(product_id) %>%
                    dplyr::summarise(
                    product_number_of_orders = n(),
                    reorders          = sum(reordered),
                    prod_mean_dow     = round(mean(order_dow, na.rm = TRUE)),
                    prod_recent_order = round(max(order_id,na.rm = TRUE)),
                    single_orders     = sum(time_gap == 1),
                    double_orders     = sum(time_gap == 2))


product_features$reorder_prob  <- product_features$double_orders / product_features$single_orders
product_features$reordered_num <- 1 + product_features$reorders / product_features$single_orders
product_features$reorder_ratio <- product_features$reorders / product_features$product_number_of_orders

product_features <- product_features %>% 
                    select(-reorders, -single_orders, -double_orders)
rm(products)
gc()

#features combining both user and products = user*products
final_data <- product_orders %>%
              group_by(user_id, product_id) %>% 
              dplyr::summarise(
              user_product_orders = n(),
              user_product_average = mean(add_to_cart_order),
              user_product_final_order = max(order_number),
              user_product_initial_order = min(order_number))
rm(product_orders, orders)
gc()

final_data <- final_data %>% 
              inner_join(product_features, by = "product_id") %>%
              inner_join(user_features, by = "user_id")

final_data$user_product_prob        <- final_data$user_product_orders / final_data$user_number_of_orders
final_data$user_product_timegap     <- final_data$user_number_of_orders - final_data$user_product_final_order
final_data$user_product_prob_reorder<- final_data$user_product_orders / (final_data$user_number_of_orders -
                                                                        final_data$user_product_initial_order + 1)

final_data <- final_data %>% 
              left_join(train_orders %>% 
              select(user_id, product_id, reordered), 
              by = c("user_id", "product_id"))
rm(train_orders, product_features, user_features)
gc()

#prepare train data
train_data            <- as.data.frame(final_data[final_data$eval_set == "train",])
train_data$product_id <- NULL
train_data$order_id   <- NULL
train_data$reordered[is.na(train_data$reordered)] <- 0
train_data$eval_set   <- NULL
train_data$user_id    <- NULL

#prepare test data
test_data             <- as.data.frame(final_data[final_data$eval_set == "test",])
test_data$eval_set    <- NULL
test_data$user_id     <- NULL
test_data$reordered   <- NULL

rm(final_data)
gc()

####### XgBoost model######## 

#train data
train_sample           <- train_data %>%
                          sample_frac(0.05)
train_sample$reordered <- as.factor(train_sample$reordered)
levels(train_sample$reordered) <- make.names(levels(as.factor(train_sample$reordered)))

#Model tuning 
#grid search for best model
xg_tune_grid         <-  expand.grid(
                      nrounds = 50,
                      eta = c(0.01, 0.03,0.1, 0.3),
                      max_depth = c( 4, 5, 6),
                      min_child_weight = c(6, 7, 8),
                      subsample = c(0.74, 0.76, 0.78),
                      colsample_bytree = c(0.94,0.95,0.96),
                      gamma = c(0,1 , 2 , 4)
                      )

# training control
xg_train_control      <- trainControl(
                      method = "cv",                      
                      number = 2,                         
                      verboseIter = TRUE,                
                      returnData = FALSE,                
                      returnResamp = "all",              
                      classProbs = TRUE,                  
                      summaryFunction = twoClassSummary,  
                      allowParallel = TRUE                
                      )

#XgBoosting : find best model
xgboost_model <- train(
             x = as.matrix(train_sample %>% select(-reordered)),
             y = train_sample$reordered,
             trControl = xg_train_control,
             tuneGrid = xg_tune_grid,
             method = "xgbTree"
             )

#training using best parameters found after grid search
input_parameters   <- list( 
  "objective"        = "reg:logistic",
  "eval_metric"      = "logloss",
  "subsample"        = 0.76,
  "eta"              = 0.1,
  "max_depth"        = 6,
  "min_child_weight" = 7,
  "lambda"           = 10,
  "gamma"            = 0.5,
  "colsample_bytree" = 0.95,
  "alpha"            = 2e-05)

train_x <- xgb.DMatrix(as.matrix(train_sample %>% select(-reordered)), 
                                  label = train_sample$reordered)

best_xgboost_model        <- xgboost(data = train_x, params = input_parameters, nrounds = 80)

feature_importance  <- xgb.importance(colnames(train_x), model = xgboost_model)
#plot top 5 features of XgBoosting
xgb.ggplot.importance(feature_importance,5)
rm(train_x, feature_importance)
gc()

# model prediction
test_x              <- xgb.DMatrix(as.matrix(test_data %>% select(-order_id, -product_id)))
test_data$reordered <- predict(best_xgboost_model, test_x)
test_data$reordered <- (test_data$reordered > 0.22) * 1

predicted_values    <- test_data %>%
                       filter(reordered == 1) %>%
                       group_by(order_id) %>%
                       summarise(
                       products = paste(product_id, collapse = " "))

nan_values          <- data.frame(
                       order_id = unique(test_data$order_id[!test_data$order_id %in% predicted_values$order_id]),
                       products = "None")

predicted_values    <- predicted_values %>% 
                       bind_rows(nan_values) %>%
                       arrange(order_id)
#Xgboost F1 score
Xg_f1_score <-  F1_Score(predicted_values,test_data$reordered, positive = NULL)
cat("Xgboost F1 score: ", Xg_f1_score)

##### adaboost model #####

#grid search input
ada_tune_grid <-  expand.grid(
                  iter = 30,
                  nu = c(0.03,0.1,1,0.3),
                  maxdepth = c(4,5,6),
                  cp = c(0.01, 0.02, 0.03),
                  bag.frac = c(0.4,0.5, 0.6 )
                   )

#training control
ada_train_control <- trainControl(
                     method = "repeatedcv",                      
                     number = 2,
                     repeats = 1,
                     verboseIter = TRUE,                
                     returnData = FALSE,                
                     returnResamp = "all",              
                     classProbs = TRUE                 
                     )

adaboost_model <- train(
                  x = as.matrix(train_sample %>% select(-reordered)),
                  y = train_sample$reordered,
                  trControl = ada_train_control,
                  tuneGrid = ada_tune_grid,
                  method = "ada" 
                  )
ada_train_x <- as.matrix(train_sample %>% select(-reordered), label = train_sample$reordered)

ada_input_parameters   <- list( 
  "loss"             = "exponential",
  "nu"               = 0.1,
  "maxdepth"         = 4,
  "min_child_weight" = 6,
  "cp"               = 0.02,
  "max.iter"         = 30,
  "bag.frac"         = 0.5
                      )

best_adaboost_model        <- adabag(data = ada_train_x, params = ada_input_parameters, nrounds = 50)
# model prediction
ada_test_x              <- as.matrix(test_data %>% select(-order_id, -product_id))
ada_test_data$reordered <- predict(best_adaboost_model, ada_test_x)
ada_test_data$reordered <- (ada_test_data$reordered > 0.22) * 1

ada_predicted_values    <- ada_test_data %>%
                       filter(reordered == 1) %>%
                       group_by(order_id) %>%
                       summarise(
                       products = paste(product_id, collapse = " "))

ada_nan_values          <- data.frame(
                       order_id = unique(ada_test_data$order_id[!ada_test_data$order_id %in% 
                                                                  ada_predicted_values$order_id]), products = "None")

ada_predicted_values    <- ada_predicted_values %>% 
                       bind_rows(ada_nan_values) %>% 
                       arrange(order_id)

ada_f1_score <-  F1_Score(ada_predicted_values,test_data$reordered, positive = NULL)
cat("adaboost F1 score: ", ada_f1_score)
