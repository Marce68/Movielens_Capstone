################################
# Importing data
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

if (sum(dir("~/R/Data/") == "ML10M.Rdata") == 0) {
  
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::",
                             "\t",
                             readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId",
                               "movieId",
                               "rating",
                               "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>%
  mutate(movieId = as.numeric(levels(movieId))[movieId],
         title = as.character(title),
         genres = as.character(genres))

movielens <- left_join(ratings,
                       movies,
                       by = "movieId")

save(movielens, file="~/R/Data/ML10M.Rdata")

}

################################
# Create edx set, validation set
################################
# Validation set will be 10% of MovieLens data
load(file = "~/R/Data/ML10M.RData")
paste("How many NA in movielens?", sum(is.na(movielens))) # Check if there are NA in the dataframe

options(digits=5)

set.seed(1, sample.kind="Rounding") # `set.seed(1)` for R version <= 3.5
test_index <- createDataPartition(y = movielens$rating,
                                  times = 1,
                                  p = 0.1,
                                  list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Clean Global Environment
rm(dl, ratings, movies, test_index, temp, movielens, removed)


################################
# Dataset exploration
################################
# Q1
dim(edx)
head(edx) %>% knitr::kable()

# Q2
edx %>%
  group_by(rating) %>%
  summarize(Num_Ratings = n()) %>%
  arrange(desc(rating)) %>%
  knitr::kable()

# Q3-Q4
data.frame(Movies = length(unique(edx$movieId)), Users = length(unique(edx$userId))) %>% knitr::kable()

# Q5
edx %>%
  group_by(genres) %>%
  summarize(Num_Ratings = length(rating)) %>%
  summarize(Drama = sum(Num_Ratings[str_detect(genres,"Drama")]),
            Comedy = sum(Num_Ratings[str_detect(genres,"Comedy")]),
            Thriller = sum(Num_Ratings[str_detect(genres,"Thriller")]),
            Romance = sum(Num_Ratings[str_detect(genres,"Romance")])) %>%
  knitr::kable()

# Q6
edx %>%
  group_by(movieId) %>%
  summarize(Title = first(title),
            Num_Ratings = length(rating)) %>%
  arrange(desc(Num_Ratings)) %>%
  slice(1:20) %>%
  knitr::kable()

# Q7
edx %>%
  group_by(rating) %>%
  summarize(Occurance = n()) %>%
  arrange(desc(Occurance)) %>%
  knitr::kable()

# Q8
edx %>%
  group_by(rating) %>%
  summarize(Occurance = n()) %>%
  arrange(desc(Occurance)) %>%
  summarize(Full_Star = sum(Occurance[(rating - round(rating)) == 0]),
            Half_Star = sum(Occurance[(rating - round(rating)) == 0.5])) %>%
  knitr::kable()


################################
# Loss Function
################################
RMSE <- function (true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
  }

###############################################
# Recommendation Systems
###############################################

#########################
# Just Average Model ####
mu_hat <- mean(edx$rating)
naive_rmse <- RMSE(validation$rating, mu_hat)

rmse_results <- data_frame(Method = "Full Set - Just the average",
                           RMSE = naive_rmse)

rmse_results %>% knitr::kable()

#########################
# Movie Effect Model ####
mu <- mean(edx$rating) 
b_i <- 
  edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

b_i %>% qplot(b_i,
              geom ="histogram",
              bins = 30,
              data = .,
              color = I("black"))

predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred


model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Full Set - Movie Effect Model",
                                     RMSE = model_1_rmse ))

rmse_results %>% knitr::kable()

###############################
# Movie & User Effect Model ###
# Let's give a look to the distribution of the average rating of each user,
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 50, color = "black")

b_u <-
  edx %>%
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

b_u %>% qplot(b_u,
              geom ="histogram",
              bins = 50,
              data = .,
              color = I("black"))

predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Full Set - Movie & User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

# Let's give a closer look at our model, we'll recommend movies with higher b_i
# So let's check which movies have the highest b_i ...
movie_titles <- edx %>%
  select(movieId, title) %>%
  distinct()

# Top 10 best movies
b_i %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i) %>%
  slice(1:10) %>%
  knitr::kable()

# ... and which one will have the lowest b_i
# Top 10 worst movies
b_i %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  select(title, b_i) %>%
  slice(1:10) %>%
  knitr::kable()

# There is something weird I don't know any movie among the best or the worst
# When calculating b_i or b_u we are grouping and averaging without considering 
# the sample size, that is how many ratings the movie had or how many ratings the user
# did.

# How many ratings had the best movies?
edx %>%
  count(movieId) %>%
  left_join(b_i) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i, n) %>%
  slice(1:10) %>%
  knitr::kable()

# How many ratings had the worst movies?
edx %>%
  count(movieId) %>%
  left_join(b_i) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  select(title, b_i, n) %>%
  slice(1:10) %>%
  knitr::kable()

##################
# Regularization
#####################################
# Let's use a more manageable dataset taking ramdomly 300,000 samples
#######################################################################################################
edx_subset <- sample(1:nrow(edx), 300000, replace = FALSE)
edx_1 <- edx[edx_subset, ]
# edx_1 <- edx

tst_idx <- createDataPartition(y = edx_1$rating,
                               times = 1,
                               p = 0.2,
                               list = FALSE)
trn_set <- edx_1[-tst_idx, ]
tst_set <- edx_1[tst_idx, ]

tst_set <- tst_set %>% 
  semi_join(trn_set, by = "movieId") %>%
  semi_join(trn_set, by = "userId")

movie_titles <- edx_1 %>% 
  select(movieId, title) %>%
  distinct()

# Just the average
mu_hat <- mean(trn_set$rating)
naive_rmse <- RMSE(tst_set$rating, mu_hat)
rmse_results <- data_frame(method = "Small Set - Just the average", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

# Movie Effect
mu <- mean(trn_set$rating) 

movie_avgs <- trn_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + tst_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, tst_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Small Set - Movie Effect Model",
                                     RMSE = model_1_rmse ))

rmse_results %>% knitr::kable()

# Movie and User effect
user_avgs <- trn_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- tst_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, tst_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Small Set - Movie & User Effects Model",  
                                     RMSE = model_2_rmse ))

rmse_results %>% knitr::kable()

###############################################################################
# We have a more manageable dataset but not considering the number of rating yet, in fact ...
###############################################################################################
movie_titles <- edx_1 %>% 
  select(movieId, title) %>%
  distinct()

# Top 10 best movies
edx_1 %>%
  dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Bottom 10 worst movies
edx_1 %>%
  dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


# Let's try a possible value for regularization
lambda <- 3

movie_reg_avgs <- trn_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# Top 10 best movies after regularization
trn_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Top 10 worst movies after regularization
trn_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

predicted_ratings <- tst_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, tst_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Small Set - Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

# It' works but the trial value is probably not optimal so
# Let's use cross validation to choose a better lambda
lambdas <- seq(0, 10, 0.25)
mu <- mean(trn_set$rating)

reg_fun <- function(l) {
  
  b_i <- trn_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- trn_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    tst_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, tst_set$rating)) 
}

rmses <- sapply(lambdas, reg_fun)

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Small Set - Regularized Movie & User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# Let's use the same lambda over the full data set
mu <- mean(edx$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_4_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Full Set - Regularized Movie & User Effect Model",  
                                     RMSE = model_4_rmse ))
rmse_results %>% knitr::kable()


# Let's finally check what are the best and worst movies

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

# Top 10 best movies after regularization
edx %>%
  dplyr::count(movieId) %>% 
  left_join(b_i) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Top 10 worst movies after regularization
edx %>%
  dplyr::count(movieId) %>% 
  left_join(b_i) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()




