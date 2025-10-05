# Imports
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, regexp_replace, split, when, lower

# ML lib
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# Word2Vec imports
# from pyspark.sql.functions import udf # has conflicts
# from pyspark.ml.feature import Word2Vec
# from pyspark.ml.feature import VectorSlicer

# CS777 Final Project
# Christopher Liu
# 2/19/2025
# This project aims to use classification models to predict whether a job posting is fraudulent or legitimate
# dataset used https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

# Initialize spark session
spark = SparkSession.builder.appName("FakeJobDetector").getOrCreate()

# read in file
# script is run in pycharm with the following params:
# fake_job_postings.csv as argument 1

# in pycharm this is where the arguments are read in
file_path = sys.argv[1]

# use data frame for easier manipulation
df = spark.read.csv(file_path, header=True, inferSchema=True)

# for some reason needed to reread in data for idf, some part of the cleaning for not using tf-idf threw multiple errors
df_tf_idf = spark.read.csv(file_path, header=True, inferSchema=True)

# Display 10 rows
print("Sample Data:")
df.show(10, truncate=False)

# Summary for numeric columns
print("\nSummary Statistics for Numeric Columns:")
df.describe(["job_id", "telecommuting", "has_company_logo", "has_questions", "fraudulent"]).show()

# class distributions
print("\nClass Distribution (Fake versus Real postings):")
df.groupBy("fraudulent").count().show()

# count missing values in each column
print("\nMissing Values Per Column:")
missing_values = df.select([(count(when(col(c).isNull(), c)).alias(c)) for c in df.columns])
missing_values.show()

# counts within categorical columns
print("\nUnique Value Counts for Categorical Columns:")
categorical_cols = ["employment_type", "required_experience", "required_education", "industry", "function"]
for col_name in categorical_cols:
    df.groupBy(col_name).count().orderBy("count", ascending=False).show(5)  # show 5

# data cleaning
# Fix numeric columns (convert to integers and remove bad values)
numeric_cols = ["telecommuting", "has_company_logo", "has_questions", "fraudulent"]
for col_name in numeric_cols:
    df = df.withColumn(col_name, when(col(col_name).cast("int").isNotNull(), col(col_name).cast("int")).otherwise(0))

# Extract salary range
df = df.withColumn("min_salary", split(col("salary_range"), "-").getItem(0).cast("int"))
df = df.withColumn("max_salary", split(col("salary_range"), "-").getItem(1).cast("int"))

# Replace NULL values in categorical columns with "Unknown"
categorical_cols = ["employment_type", "required_experience", "required_education", "industry", "function"]
for col_name in categorical_cols:
    df = df.withColumn(col_name, when(col(col_name).isNull(), "Unknown").otherwise(col(col_name)))

# Drop original salary_range column (since we extracted min/max salary)
df = df.drop("salary_range")

# Show dataset
# print("\n Cleaned Data Sample:")
# df.show(10, truncate=False)

# Encode Categorical Columns using StringIndexer
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html
# helps change categorical values into numeric for ML use
categorical_cols = ["employment_type", "required_experience", "required_education", "industry", "function"]
indexed_cols = [col_name + "_index" for col_name in categorical_cols]

for col_name, indexed_col in zip(categorical_cols, indexed_cols):
    indexer = StringIndexer(inputCol=col_name, outputCol=indexed_col, handleInvalid="keep")
    df = indexer.fit(df).transform(df)

# Handle unknown Salary Values, cast as double for later (error thrown when not in double?)
df = df.withColumn("min_salary", when(col("min_salary").isNotNull(), col("min_salary")).otherwise(0).cast("double"))
df = df.withColumn("max_salary", when(col("max_salary").isNotNull(), col("max_salary")).otherwise(0).cast("double"))

# print("salary columns after split:")
# df.select("min_salary", "max_salary").show(5)

# Vectorize and Scale Salary
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html
vector_assembler = VectorAssembler(inputCols=["min_salary", "max_salary"],
                                   outputCol="salary_features",
                                   handleInvalid="keep")
# apply transformation
df = vector_assembler.transform(df)

# scale features
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StandardScaler.html#pyspark.ml.feature.StandardScaler.outputCol
scaler = StandardScaler(inputCol="salary_features",
                        outputCol="scaled_salary",
                        withStd=True,
                        withMean=True)
df = scaler.fit(df).transform(df)

# Select Final Columns for ML
final_cols = ["telecommuting", "has_company_logo", "has_questions", "fraudulent"] + indexed_cols + ["scaled_salary"]

# pass each column from final_cols to .select
df = df.select(*final_cols)

# Show data after processing and do not truncate to see full output
print("\n Feature Engineered Data Sample:")
df.show(10, truncate=False)

# high cardinality features throw errors(binning error, change bin size, not great fix, but it works)
# Identify High-Cardinality Features(categorical features that throw errors)
high_cardinality_cols = ["industry_index", "function_index"]

for col_name in high_cardinality_cols:
    category_counts = df.groupBy(col_name).count().orderBy("count", ascending=False)

    # Set higher threshold to aggressively reduce unique values
    rare_threshold = 100
    rare_categories = [row[col_name] for row in category_counts.filter(col("count") < rare_threshold).collect()]
    df = df.withColumn(col_name, when(col(col_name).isin(rare_categories), "Other").otherwise(col(col_name)))

# Index Categories
indexed_cols = []
for col_name in high_cardinality_cols:
    # apply string indexer on high cardinal features
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed", handleInvalid="keep")
    df = indexer.fit(df).transform(df)
    indexed_cols.append(col_name + "_indexed")

# drop high cardinal columns
df = df.drop(*high_cardinality_cols)

indexed_features = ["industry_index_indexed", "function_index_indexed"]
# Get max unique values across all indexed categorical features
max_unique_categories = max([df.select(col_name).distinct().count() for col_name in indexed_features])

# Find the max unique category count to set `maxBins`
maxBinsValue = max(400,  # Set a fixed large value to avoid issues, dataset has some weird features, value was chosen
                   # through viewing error message that said one bin was like 360 values or smth
                   df.select("employment_type_index").distinct().count(),
                   df.select("required_experience_index").distinct().count(),
                   df.select("required_education_index").distinct().count())

# Assemble Feature Vector
feature_cols = ["telecommuting", "has_company_logo", "has_questions", "employment_type_index",
                "required_experience_index", "required_education_index",
                "industry_index_indexed", "function_index_indexed"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df).select("features", "fraudulent")

# Ensure `fraudulent` is binary (fix multi-class issue, threw multiple errors that fraudulent wasn't binary?)
# had to cast a few times before and slowly got rid of redundant ones
df = df.withColumn("fraudulent", when(col("fraudulent") == 1, 1).otherwise(0))

# Train-Test Split (80% train 20% test)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=6944)

models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="fraudulent"),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="fraudulent", maxBins=maxBinsValue),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="fraudulent", numTrees=50,
                                            maxBins=maxBinsValue),
    "Naive Bayes": NaiveBayes(featuresCol="features", labelCol="fraudulent"),
    "Gradient Boosting": GBTClassifier(featuresCol="features", labelCol="fraudulent", maxBins=maxBinsValue)
}

# Class Distribution
class_counts = train_data.groupBy("fraudulent").count()
class_counts.show()

# Get Majority and Minority Class Sizes
majority_class = train_data.filter(col("fraudulent") == 0)
minority_class = train_data.filter(col("fraudulent") == 1)

majority_count = majority_class.count()
minority_count = minority_class.count()

print(f"Majority Class Count: {majority_count}")
print(f"Minority Class Count: {minority_count}")

# Compute Oversampling Ratio
oversampling_ratio = majority_count / minority_count
print(f"Applying Oversampling Ratio: {oversampling_ratio:.2f}")

# Apply Oversampling (Randomly Duplicate Minority Class Rows)
minority_oversampled = minority_class.sample(withReplacement=True, fraction=oversampling_ratio, seed=6944)

# Combine with Majority Class
train_data_balanced = majority_class.union(minority_oversampled)

# Verify New Class Distribution
train_data_balanced.groupBy("fraudulent").count().show()

# Evaluators AUC, F1, Precision, Recall, and Accuracy
evaluator_auc = BinaryClassificationEvaluator(labelCol="fraudulent",
                                              rawPredictionCol="probability",
                                              metricName="areaUnderROC")

evaluator_f1 = MulticlassClassificationEvaluator(labelCol="fraudulent",
                                                 predictionCol="prediction",
                                                 metricName="f1")

evaluator_precision = MulticlassClassificationEvaluator(labelCol="fraudulent",
                                                        predictionCol="prediction",
                                                        metricName="weightedPrecision")

evaluator_recall = MulticlassClassificationEvaluator(labelCol="fraudulent",
                                                     predictionCol="prediction",
                                                     metricName="weightedRecall")

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="fraudulent",
                                                       predictionCol="prediction",
                                                       metricName="accuracy")

print("\nUnbalanced Dataset Metrics")
# Training & Evaluation Loop
for name, model in models.items():
    print(f"\nTraining {name}...")
    trained_model = model.fit(train_data)  # ensure using unbalanced data for comparison
    predictions = trained_model.transform(test_data)

    # metrics
    auc = evaluator_auc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    accuracy = evaluator_accuracy.evaluate(predictions)

    # Print
    print(f"{name} AUC: {auc:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")
    print(f"{name} Precision: {precision:.4f}")
    print(f"{name} Recall: {recall:.4f}")
    print(f"{name} Accuracy: {accuracy:.4f}")


# Training & Evaluation Loop
print("\nBalanced Dataset")
for name, model in models.items():
    print(f"\n Training {name}...")
    trained_model = model.fit(train_data_balanced)  # ensure using balanced dataset
    predictions = trained_model.transform(test_data)

    # metrics
    auc = evaluator_auc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    accuracy = evaluator_accuracy.evaluate(predictions)

    # Print
    print(f"{name} AUC: {auc:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")
    print(f"{name} Precision: {precision:.4f}")
    print(f"{name} Recall: {recall:.4f}")
    print(f"{name} Accuracy: {accuracy:.4f}")

# TF-IDF
# using the description column to see if any improvements are made with that feature
# Handle missing values in the description column (replace NULLs with empty strings)
# https://spark.apache.org/docs/3.5.2/mllib-feature-extraction.html#tf-idf
# https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/
if "description" in df_tf_idf.columns:
    df_tf_idf = df_tf_idf.withColumn("description", when(col("description").isNull(), "").otherwise(col("description")))
else:
    print("Error: 'description' column is missing from df_tf_idf")  # error catching to see if desc is in df

# Clean Data - removed punctuation and made lower
df_tf_idf = df_tf_idf.withColumn("description", regexp_replace(col("description"), "[^a-zA-Z\s]", ""))

# Tokenize Job Descriptions
tokenizer = Tokenizer(inputCol="description", outputCol="words")
df_tf_idf = tokenizer.transform(df_tf_idf)

# Remove Stopwords (like "the", "is", "and", etc.)
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_tf_idf = stopwords_remover.transform(df_tf_idf)
df_tf_idf = df_tf_idf.select(*df_tf_idf.columns)  # for Word(2)vec, cleaned columns

# Convert Words into Term Frequency (TF)
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
df_tf_idf = hashing_tf.transform(df_tf_idf)

# Compute TF-IDF Scores
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
df_tf_idf = idf.fit(df_tf_idf).transform(df_tf_idf)

# Apply StringIndexer to categorical columns
categorical_cols = ["employment_type", "required_experience", "required_education"]
indexed_cols = [col_name + "_index" for col_name in categorical_cols]

for col_name, indexed_col in zip(categorical_cols, indexed_cols):
    indexer = StringIndexer(inputCol=col_name, outputCol=indexed_col, handleInvalid="keep")
    df_tf_idf = indexer.fit(df_tf_idf).transform(df_tf_idf)

# Convert binary categorical columns to integer
binary_cols = ["telecommuting", "has_company_logo", "has_questions"]
for col_name in binary_cols:
    df_tf_idf = df_tf_idf.withColumn(col_name, col(col_name).cast("int"))

# Assemble Features
feature_cols = ["telecommuting", "has_company_logo", "has_questions", "employment_type_index",
                "required_experience_index", "required_education_index", "tfidf_features"]

# Assemble features and transform df_tf_idf
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_tf_idf = assembler.transform(df_tf_idf).select("features", "fraudulent")

models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="fraudulent"),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="fraudulent", maxBins=500),  # bins at 500
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="fraudulent", numTrees=50, maxBins=500),
    "Naive Bayes": NaiveBayes(featuresCol="features", labelCol="fraudulent"),
    "Gradient Boosting": GBTClassifier(featuresCol="features", labelCol="fraudulent", maxBins=500)
}

# Evaluate Models with Timing
print("\nmodels with TF-IDF")
for name, model in models.items():
    print(f"\n Training {name}...")

    start_time = time.time()  # Start timer

    trained_model = model.fit(train_data_balanced)  # Train model on balanced dataset
    predictions = trained_model.transform(test_data)

    end_time = time.time()  # End timer
    training_time = end_time - start_time  # elapsed time

    # Compute metrics
    auc = evaluator_auc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    accuracy = evaluator_accuracy.evaluate(predictions)

    # Print
    print(f"{name} Training Time: {training_time:.2f} seconds")  # Print training time
    print(f"{name} AUC: {auc:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")
    print(f"{name} Precision: {precision:.4f}")
    print(f"{name} Recall: {recall:.4f}")
    print(f"{name} Accuracy: {accuracy:.4f}")

# Word2Vec
# implementation failed
spark.stop()
