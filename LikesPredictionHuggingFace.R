#install and access important libraries
library(httr)
library(dplyr)
library(purrr)
library(tidyverse)
library(ggplot2)
library(corrplot)
library(scales)
library(plotly)
library(h2o)
library(tidyverse)
library(skimr)
library(recipes)
library(stringr)
library(tidyverse)
library(kableExtra)
library(dplyr)
library(DALEX)
library(DALEXtra)
library(scales)
library(h2o)

# Fetch data from the Hugging face model api 
api_key = "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"

fetch_and_process_page <- function(url, api_key, selected_columns) {
  response <- GET(url, add_headers("Authorization" = paste("Bearer", api_key)))
  data_list <- content(response)
  
  # Extract only the selected columns
  filtered_data <- lapply(data_list, function(item) {
    selected <- item[c("id", "downloads", "private", "createdAt", "pipeline_tag", "library_name", "tags","likes")]
    names(selected) <- c("model_id", "downloads", "private", "created_At", "pipeline_tag", "library_names", "tags","likes")
    return(selected)
  })
  
  df <- as.data.frame(do.call(rbind, filtered_data), stringsAsFactors = FALSE)
  df$likes <- as.numeric(df$likes)
  return(df)
}

# Define the columns you want to fetch
selected_columns <- c("model_id", "downloads", "private", "created_At", "pipeline_tag", "library_names", "tags","likes")

# Use the selected_columns in the fetch function
df_all <- as.data.frame(NULL)  # Initialize an empty data frame
url = "https://huggingface.co/api/models"
response <- GET("https://huggingface.co/api/models", add_headers("Authorization" = paste("Bearer", api_key)))

# Fetching additional pages using loop
counter <- 1
while (!is.null(headers(response)$link) && counter < 50) {
  link_header <- headers(response)$link
  link_match <- regmatches(link_header, regexpr('<(.+?)>', link_header))[[1]]
  next_page_url <- gsub("(<|>)", "", link_match)
  # Fetch and process the next page
  df_next_page <- fetch_and_process_page(next_page_url, api_key)
  # Append the data to the existing dataframe
  df_all <- rbind(df_all, df_next_page)
  # Update response for the next iteration
  response <- GET(next_page_url, add_headers("Authorization" = paste("Bearer", api_key)))
}

# Filter rows where any selected column has "NULL" value or likes=0
df_filtered <- subset(df_all, model_id != "NULL" & 
                        downloads != "NULL" & 
                        private != "NULL" & 
                        created_At != "NULL" & 
                        pipeline_tag != "NULL" & 
                        library_names != "NULL" & 
                        tags != "NULL" & 
                        likes != 0)

#Checking the number of 0 and NA values in the dataframe
total_rows = nrow(df_filtered)
total_rows
num_zeros <- sum(df_filtered$likes == 0)
num_zeros     

any_na = any(is.null(df_all))
any_na <- any(is.na(df_filtered$pipeline_tag))
any_na
na_count_vec <- sum(is.na(df_filtered$pipeline_tag))
print(na_count_vec)

# Un-nest the tags column which is present in list type 
combined_df <- unnest(df_filtered, cols = tags)
View(combined_df)

# group the data by model_id parameter
grouped_data <- combined_df %>%
  group_by(combined_df$model_id)

#Using pivot wider to obtain tags as separate predictors
df_tags <- combined_df %>%
  group_by(likes , downloads , model_id, pipeline_tag, created_At , tags) %>%
  summarise(n = n()) %>%
  pivot_wider(names_from = tags, values_from = n, values_fill = 0, names_glue = "{tags}")
View(df_tags)

# Taking count of different Tags column
tag_counts <- df_tags %>% count(tags) %>% arrange(desc(n))

# List of Top 20 tags as predictors with other columns
subset_df <- df_tags[, c("model_id" , "private", "downloads", "created_At" , "pipeline_tag", 
                         "transformers" , "endpoints_compatible" , "pytorch" , 
                         "autotrain_compatible","license:apache-2.0","bert",
                         "text-generation-inference","tensorboard",
                         "text-classification", 
                         "en", "jax", "generated_from_trainer",
                         "text-generation", "text2text-generation" ,"gpt2", "has_space" ,
                         "model-index" ,
                         "tf" , "automatic-speech-recognition",
                         "fill-mask","likes")]
new_df <- data.frame(subset_df)

# Again checking if there is any zero or NA values in dataframe
total_rows = nrow(new_df)
total_rows
num_zeros <- sum(new_df$likes == 0)
num_zeros

any_na <- any(is.na(new_df$pipeline_tag))
any_na

columns_to_flatten <- c("model_id", "downloads", "created_At", "pipeline_tag", "likes")

# Unnest specified columns
new_df_flattened <- new_df %>%
  unnest(cols = all_of(columns_to_flatten))

# Converting character columns to appropriate types
new_df_flattened <- new_df_flattened %>%
  mutate(across(where(is.character), as.character),
         across(where(is.numeric), as.numeric))
output_file <- "HuggingFaceDataset.csv"

# Save the final processed data in csv file
write.csv(new_df_flattened, output_file, row.names = FALSE)

# Initialize h2o
h2o.init(nthreads = -1)

# Select predictors and response variables
X <- new_df |> select(-"likes")
Y <- new_df |> select("likes")

# Summary statistics for X
x_train_tbl_skim = partition(skim(X))
names(x_train_tbl_skim)
x_train_tbl_skim$character
x_train_tbl_skim$numeric |> tibble()

# Extracting character variables
string_2_factor_names <- x_train_tbl_skim$character$skim_variable

# Unnesting X variables
X_unnested1 <- unnest(X, cols = model_id, downloads, created_At, pipeline_tag)
class(X_unnested1$model_id)
class(X_unnested1$downloads)

# Unnesting Y variables
Y_unnest <- unnest(Y, cols = likes)
class(Y_unnest$likes)

# Data preprocessing for X
rec_obj <- recipe(~ ., data = X_unnested1) |>
  step_string2factor(all_of(string_2_factor_names)) |>
  step_impute_median(all_numeric()) |> # missing values in numeric columns
  step_impute_mode(all_nominal()) |> # missing values in factor columns
  prep()
rec_obj
x_train_processed_tbl <- bake(rec_obj, X_unnested1)
x_train_processed_tbl

# Data preprocessing for Y
rec_obj_y <- recipe( ~ ., data = Y_unnest) |> prep(stringsAsFactors = FALSE)
y_train_processed_tbl <- bake(rec_obj_y, Y_unnest)
y_train_processed_tbl
class(y_train_processed_tbl)

# Convert data frames to H2OFrame
data_h2o <- as.h2o(
  bind_cols(y_train_processed_tbl, x_train_processed_tbl),
  destination_frame = "train.hex" 
)

# Splitting the data into train, validation, and test sets
splits <- h2o.splitFrame(data = data_h2o, seed = 1234, ratios = c(0.6, 0.2)) # 60/20/20 split
train_h2o <- splits[[1]]
valid_h2o <- splits[[2]] 
test_h2o <- splits[[3]] # Convert the data frame to an H2OFrame

#Training the model
y <- "likes" 
x <- setdiff(names(train_h2o), y) 

random_forest <- h2o.randomForest(model_id = "group2-randomForest.h2o",x = setdiff(names(train_h2o), y) , y = "likes", training_frame = train_h2o)



h2o.saveModel(object = random_forest, 
              path = getwd(), 
              force = TRUE)


