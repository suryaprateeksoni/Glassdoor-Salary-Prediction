# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 02:13:48 2020

@author: Surya Prateek Soni
"""
################################################################################
# Working Directory
################################################################################
import os
os.getcwd()
os.chdir("C:\\Users\\Surya Prateek Soni\\Desktop")



###################################################################################################################################################
# WEB SCRAPPING
###################################################################################################################################################
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium import webdriver
import time
import pandas as pd

# Web Scrapping Code 
def get_jobs(keyword, num_jobs, verbose, path, slp_time):
    
    '''Gathers jobs as a dataframe, scraped from Glassdoor'''
    
    #Initializing the webdriver
    options = webdriver.ChromeOptions()
    
    #Uncomment the line below if you'd like to scrape without a new Chrome window every time.
    #options.add_argument('headless')
    
    #Change the path to where chromedriver is in your home folder.
    driver = webdriver.Chrome(executable_path=path, options=options)
    driver.set_window_size(1120, 1000)
    
    url = "https://www.glassdoor.com/Job/jobs.htm?suggestCount=0&suggestChosen=false&clickSource=searchBtn&typedKeyword="+keyword+"&sc.keyword="+keyword+"&locT=&locId=&jobType="
    #url = 'https://www.glassdoor.com/Job/jobs.htm?sc.keyword="' + keyword + '"&locT=C&locId=1147401&locKeyword=San%20Francisco,%20CA&jobType=all&fromAge=-1&minSalary=0&includeNoSalaryJobs=true&radius=100&cityId=-1&minRating=0.0&industryId=-1&sgocId=-1&seniorityType=all&companyId=-1&employerSizes=0&applicationType=0&remoteWorkType=0'
    driver.get(url)
    jobs = []

    while len(jobs) < num_jobs:  #If true, should be still looking for new jobs.

        #Let the page load. Change this number based on your internet speed.
        #Or, wait until the webpage is loaded, instead of hardcoding it.
        time.sleep(slp_time)

        #Test for the "Sign Up" prompt and get rid of it.
        try:
            driver.find_element_by_class_name("selected").click()
        except ElementClickInterceptedException:
            pass

        time.sleep(.1)

        try:
            driver.find_element_by_css_selector('[alt="Close"]').click() #clicking to the X.
            print(' x out worked')
        except NoSuchElementException:
            print(' x out failed')
            pass

        
        #Going through each job in this page
        job_buttons = driver.find_elements_by_class_name("jl")  #jl for Job Listing. These are the buttons we're going to click.
        for job_button in job_buttons:  

            print("Progress: {}".format("" + str(len(jobs)) + "/" + str(num_jobs)))
            if len(jobs) >= num_jobs:
                break

#            job_button.click()  #You might 
            driver.execute_script("arguments[0].click();", job_button)
            time.sleep(1)
            collected_successfully = False
            
            while not collected_successfully:
                try:
                    company_name = driver.find_element_by_xpath('.//div[@class="employerName"]').text
                    location = driver.find_element_by_xpath('.//div[@class="location"]').text
                    job_title = driver.find_element_by_xpath('.//div[contains(@class, "title")]').text
                    job_description = driver.find_element_by_xpath('.//div[@class="jobDescriptionContent desc"]').text
                    collected_successfully = True
                except:
                    time.sleep(5)

            try:
                salary_estimate = driver.find_element_by_xpath('.//span[@class="gray salary"]').text
            except NoSuchElementException:
                salary_estimate = -1 #You need to set a "not found value. It's important."
            
            try:
                rating = driver.find_element_by_xpath('.//span[@class="rating"]').text
            except NoSuchElementException:
                rating = -1 #You need to set a "not found value. It's important."

            #Printing for debugging
            if verbose:
                print("Job Title: {}".format(job_title))
                print("Salary Estimate: {}".format(salary_estimate))
                print("Job Description: {}".format(job_description[:500]))
                print("Rating: {}".format(rating))
                print("Company Name: {}".format(company_name))
                print("Location: {}".format(location))

            #Going to the Company tab...
            #clicking on this:
            #<div class="tab" data-tab-type="overview"><span>Company</span></div>
            try:
                driver.find_element_by_xpath('.//div[@class="tab" and @data-tab-type="overview"]').click()

                try:
                    #<div class="infoEntity">
                    #    <label>Headquarters</label>
                    #    <span class="value">San Francisco, CA</span>
                    #</div>
                    headquarters = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Headquarters"]//following-sibling::*').text
                except NoSuchElementException:
                    headquarters = -1

                try:
                    size = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Size"]//following-sibling::*').text
                except NoSuchElementException:
                    size = -1

                try:
                    founded = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Founded"]//following-sibling::*').text
                except NoSuchElementException:
                    founded = -1

                try:
                    type_of_ownership = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Type"]//following-sibling::*').text
                except NoSuchElementException:
                    type_of_ownership = -1

                try:
                    industry = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Industry"]//following-sibling::*').text
                except NoSuchElementException:
                    industry = -1

                try:
                    sector = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Sector"]//following-sibling::*').text
                except NoSuchElementException:
                    sector = -1

                try:
                    revenue = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Revenue"]//following-sibling::*').text
                except NoSuchElementException:
                    revenue = -1

                try:
                    competitors = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Competitors"]//following-sibling::*').text
                except NoSuchElementException:
                    competitors = -1

            except NoSuchElementException:  #Rarely, some job postings do not have the "Company" tab.
                headquarters = -1
                size = -1
                founded = -1
                type_of_ownership = -1
                industry = -1
                sector = -1
                revenue = -1
                competitors = -1

                
            if verbose:
                print("Headquarters: {}".format(headquarters))
                print("Size: {}".format(size))
                print("Founded: {}".format(founded))
                print("Type of Ownership: {}".format(type_of_ownership))
                print("Industry: {}".format(industry))
                print("Sector: {}".format(sector))
                print("Revenue: {}".format(revenue))
                print("Competitors: {}".format(competitors))
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            jobs.append({"Job Title" : job_title,
            "Salary Estimate" : salary_estimate,
            "Job Description" : job_description,
            "Rating" : rating,
            "Company Name" : company_name,
            "Location" : location,
            "Headquarters" : headquarters,
            "Size" : size,
            "Founded" : founded,
            "Type of ownership" : type_of_ownership,
            "Industry" : industry,
            "Sector" : sector,
            "Revenue" : revenue,
            "Competitors" : competitors})
            #add job to jobs
            
            
        #Clicking on the "next page" button
        try:
            driver.find_element_by_xpath('.//li[@class="next"]//a').click()
        except NoSuchElementException:
            print("Scraping terminated before reaching target number of jobs. Needed {}, got {}.".format(num_jobs, len(jobs)))
            break

    return pd.DataFrame(jobs)  #This line converts the dictionary object into a pandas DataFrame

#import glassdoor_scraper as gs
path = "C:/Users/Surya Prateek Soni/Desktop/chromedriver"
# Calling the function we created above "get_jobs". 1000 are number of jobs, 15 is waiting time before it starts 
import pandas as pd
data = get_jobs("data scientist", 500, False, path, 15)



################################################################################
# Data Cleaning 
################################################################################
import pandas as pd
data = pd.read_csv("glassdoor_jobs.csv")
data["hourly"] =  data["Salary Estimate"].apply(lambda x : 1 if  'per hour' in x.lower() else 0)  # Creating new column, if "per hour" is there in "Salary Estimate"
data["employer_provided"] =  data["Salary Estimate"].apply(lambda x : 1 if  'employer provided salary' in x.lower() else 0)  # Creating new column, if "employer provided salary" is there in "Salary Estimate"
# Removing those rows which have Salary Estimate = -1
data = data[data["Salary Estimate"] != "-1"]  # Removing -1
# Removing extra elements from Salary Estimate
salary_without_glassdoor = data["Salary Estimate"].apply(lambda x: x.split('(')[0])  # Removing "Glassdooor"
salary_minues_kd = salary_without_glassdoor.apply(lambda x: x.replace('K', "").replace("$", "")) # Removing Dollar sign and K
salary_remove_all = salary_minues_kd.apply(lambda x: x.lower().replace("per hour", "").replace("employer provided salary:", "")) # Removing Per Hour and "Employer Provided" in Salary Estimate column
# Getting the average Salary
data["min_salary"] = salary_remove_all.apply(lambda x: x.split("-")[0])  # Placing the minimum salary form the series calculated above 
data["max_salary"] = salary_remove_all.apply(lambda x: x.split("-")[1])  # Placing the minimum salary form the series calculated above 
data["average_salary"] = (data.min_salary.astype(int) + data.max_salary.astype(int))/2  

# Company Name Text Only 
data["company_txt"] = data.apply(lambda x :x["Company Name"] if x['Rating'] < 0 else  x['Company Name'][:-3], axis = 1)

# State Name 
data["job_state"] = data["Location"].apply(lambda x: x.split(",")[1])
# Fixing San Francisco
data["job_state"] = data.job_state.apply(lambda x: x.strip() if x.strip().lower() != "los angeles" else "CA")
# Counting the number of jobs in every state
data.job_state.value_counts()  
# Checking if the job location is the same as Headquarters
data["location_same_headquarters"] = data.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis= 1)

# Age of Company 
data["Company_Age"] = data.Founded.apply(lambda x: x if x < 1 else 2020-x)  

# Check the Job Description, keywords it has 
data["python_YN"] = data["Job Description"].apply(lambda x: 1 if "python" in x.lower() else 0)   
data.python_YN.value_counts()
data["R_YN"] = data["Job Description"].apply(lambda x: 1 if "rstudio" in x.lower() else 0)   
data.R_YN.value_counts()
data["spark_YN"] = data["Job Description"].apply(lambda x: 1 if "spark" in x.lower() else 0)   
data.spark_YN.value_counts()
data["aws_YN"] = data["Job Description"].apply(lambda x: 1 if "aws" in x.lower() else 0)   
data.aws_YN.value_counts()
data["excel_YN"] = data["Job Description"].apply(lambda x: 1 if "excel" in x.lower() else 0)   
data.excel_YN.value_counts()

# Removing the first columns 
data = data.drop(["Unnamed: 0"], axis = 1)


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Converting "too many catorgies" into "broader lesser catogories"
def title_simplifier(title):
    if "data scientist" in title.lower():
        return "data scientist"
    elif "data engineer" in title.lower():
        return "data engineer"
    elif "analyst" in title.lower():
        return "analyst"
    elif "machine learning" in title.lower():
        return "machine learning"
    elif "manager" in title.lower():
        return "manager"
    elif "director" in title.lower():
        return "director"
    else:
        return "na"
    
# Converting the prefixes of Senior and Junior into only 2 categories 
def seniority(title):
    if "sr" in title.lower() or "senior" in title.lower() or "lead" in title.lower() or "principal" in title.lower():
        return "senior"
    elif "jr" in title.lower() or "jr." in title.lower():
        return "junior"
    else:
        return "na"
    
data["job_simp"] = data["Job Title"].apply(title_simplifier)
data.job_simp.value_counts()
data["seniority"] = data["Job Title"].apply(seniority)
data.seniority.value_counts()

# Description length
data["description_len"] = data["Job Description"].apply(lambda x: len(x))
data["description_len"] 

# Competitor Count 
data["competitors_number"] = data["Competitors"].apply(lambda x: len(x.split(',')) if x != "-1" else 0)

# Hourly Wage to annual wages 
data["min_salary"] = data["min_salary"].astype(int)
data["min_salary"] = data.apply(lambda x: x.min_salary*2 if x.hourly ==1 else x.min_salary, axis =1)
data[data.hourly == 1][["hourly", "min_salary", "max_salary"]]

data["max_salary"] = data["max_salary"].astype(int)
data["max_salary"] = data.apply(lambda x: x.max_salary*2 if x.hourly ==1 else x.max_salary, axis =1)
data[data.hourly == 1][["hourly", "min_salary", "max_salary"]]

# Remove New Line from Job Title
data.company_txt
data["company_txt"] = data.company_txt.apply(lambda x: x.replace("\n", ""))
data.company_txt



################################################################################
# Exploratory Data Analysis 
################################################################################

data.describe()   # This gives all continous variables 

# Now create some histograms for important variables 
 
# Creating Box Plot
data["Rating"].hist()
data["average_salary"].hist()
data["Company_Age"].hist()

# Creating Box Plot 
data.boxplot(column = ["Company_Age", "average_salary", "Rating"])
data.boxplot(column = "Rating")
data[["Company_Age", "average_salary", "Rating", "description_len"]].corr()

# Creating a Heatmap
cmap = sns.diverging_palette(220,10,as_cmap= True)
sns.heatmap(data[["Company_Age", "average_salary", "Rating", "description_len", "competitors_number"]].corr(), vmax = 0.3, center = 0, square = True, linewidths = 0.5, cbar_kws={"shrink": 0.9}, cmap = cmap)

# Creating a Dataframe with all Categorical variables
data_categorical = data[['Location', 'Headquarters', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'company_txt', 'job_state', "location_same_headquarters", 'python_YN', 'R_YN',
       'excel_YN', 'spark_YN', 'aws_YN', 'job_simp', 'seniority']]
# Making Bar Charts for all variables
for i in data_categorical.columns:  #Range is number of categorical columns
    different_categories = data_categorical[i].value_counts()
    print("Graph for %s: total = %d" % (i, len(different_categories)))
    barplot = sns.barplot(x = different_categories.index, y = different_categories)
    plt.show()


for i in data_categorical[['Location', 'Headquarters', 'company_txt']].columns:  #Range is number of categorical columns
    different_categories = data_categorical[i].value_counts()[:20]
    print("Graph for %s: total = %d" % (i, len(different_categories)))
    barplot = sns.barplot(x = different_categories.index, y = different_categories)
    barplot.set_xticklabels(barplot.get_xticklabels(), rotation = 90)
    plt.show()

pd.set_option ("display.max_rows", None)
# Average Salaries across Job Types
pd.pivot_table(data, index = "job_simp", values = "average_salary")
# Average Salaries across Job Types and Seniority Level
pd.pivot_table(data, index = ["job_simp", "seniority"], values = "average_salary")
# Average Salaries across States, sorted 
pd.pivot_table(data, index = "job_state",values = "average_salary").sort_values("average_salary", ascending = False)
# Count of Jobs across States and Job Types
pd.pivot_table(data, index = ["job_state","job_simp"],  values = "average_salary", aggfunc = "count").sort_values("job_state", ascending = True)
# Checking the avrage salary of only Data Scientist actoss States
pd.pivot_table(data[data.job_simp == "data scientist"], index = "job_state",  values = "average_salary").sort_values("average_salary", ascending = False)


# More Pivots 
more_pivots = data[['Rating','Industry', 'Sector', 'Revenue', 'competitors_number','hourly', 'employer_provided', 'python_YN', 'R_YN', 'excel_YN', 'spark_YN', 'aws_YN',  'Type of ownership', "average_salary"]]
for i in more_pivots.iloc[:, 0:13].columns:
    print(i)
    print(pd.pivot_table(more_pivots, index = i, values = "average_salary").sort_values("average_salary", ascending = False))

# Check if Pythin has more demand 
pd.pivot_table(more_pivots, index = "Revenue", columns = "python_YN", values = "average_salary", aggfunc = "count")


# Counting the words used in the description 
from wordcloud import  WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

words = " ".join(data["Job Description"]) 

# Removing the StopWords
def puctuation_stop(text):
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered
    
# Running the above function punctuation_stop
words_filtered = puctuation_stop(words)

text = " ".join([ele for ele in words_filtered])

wc = WordCloud(background_color = "white", random_state = 1, stopwords = STOPWORDS, max_words = 2000, width = 800, height = 1500)
wc.generate(text)

plt.figure(figsize = [10,10])
plt.imshow(wc, interpolation = "bilinear")
plt.axis("off")
plt.show()


####################################################################################################################################
# MODEL BUILDING 
####################################################################################################################################
# Estimate salaries of various positions by building various models 
# Choose Relevant columns 
# Get Dummy data 
# Train Validation Test Split 
# Multiple Linear regression 
# Lasso regression - Used becasue data is sparse becasue of all dummy variables, it will help normlaize it 
# random forest - Tree based model will perform better because we have a lot of Dummy Variables. It will to be compared against linear model 
# tune these models using GridsearchCV
# test enasambles 

# Choose Relevant columns
data.columns
# Selecting the columns we think which would be relevant 
data_for_modeling = data[["average_salary", 'Rating','Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'competitors_number','hourly', 'employer_provided', 'job_state', 'location_same_headquarters', 'Company_Age', 'python_YN','excel_YN', 'spark_YN', 'aws_YN', 'job_simp', 'seniority','description_len']]  
# Get Dummy Data - Convert every category of Categorical data into different columns
data_dummies = pd.get_dummies(data_for_modeling)     
# Train Test split 
from sklearn.model_selection import train_test_split
x = data_dummies.drop("average_salary", axis = 1)   #Removing the Dependent Variable 
y = data_dummies.average_salary.values   # Without Values it is Pandas series, but for Modeling, it is recommended to use Arrays, so use "Values"
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)



# Multiple Linear Regression 
# We will do Linear Resgression in both Stats Model and Sklearn, and will compare them
# Part 1- Stats Model 0
import statsmodels.api as stats_model
# Adding a constant as a Column
x_stats_model = stats_model.add_constant(x)
model_stats_model = stats_model.OLS(y,x_stats_model)
model_stats_model.fit().summary()
# Above we saw, R swuared is 82%, then check the P values of all the variables. if it is less than 0.05, only then it is significant.
# There is a high level of Multicollinearity, Hence this model is considered as Baseline Model  


# Part 2- Linear Regression with Cross Validation
# Everything is the same as Stats Model, except we will Cross validate it. 
from sklearn.linear_model import LinearRegression  # Model used is LinearRegression
from sklearn.model_selection import cross_val_score
linreg = LinearRegression()   # Instantiate Linear Regression as Reg 
linreg.fit(x_train, y_train) 
linreg_cv_scores = cross_val_score(linreg,x_train, y_train, scoring = "neg_mean_absolute_error", cv = 3)    # Cross Validation 
linreg_cv_scores_mean = np.mean(linreg_cv_scores )   # Taking the Mean of 3 CV scores calculated above
linreg_cv_scores_mean 


#######################

# LASSO
# Linear Regression with Cross Validation AND Lasso Regression 
# Lasso is used becasue the matrix is too sparse, and Regularization will normalize it 
# Alpha = 0 means the same as OLS Multiple Linear Regression
# As ALpha increases, it increases the amount by which data is smooth
# Lets start wiotyh trying with Alpha = 1  
from sklearn.linear_model import Lasso  # Model used is Lasso 
from sklearn.model_selection import cross_val_score
lasso_linreg = Lasso()
lasso_linreg_cv_scores = cross_val_score(lasso_linreg, x_train, y_train, scoring = "neg_mean_absolute_error", cv = 3)     # Cross Validation - Calculating the score
lasso_linreg_cv_scores_mean = np.mean(lasso_linreg_cv_scores)   # Taking the Mean of 3 CV scores calculated above
lasso_linreg_cv_scores_mean 
# Above was with Alpha = 1, whicb is by default.
# Below is to test various Alpha values to identify the Maximum score 
# Creating Empty List
alpha = []
error = []
# Creating a loop- For every value of alpha, Calculate array of errors and take the mean of that Error.
# This will give us 1 Error for 1 Alpha value
for i in range(1,100):   # We need i betyween 0.1 and 10, but we cannot use Floating numbers in Range
    alpha.append(i/10)   # We are storing all values of Alpha
    lasso_linreg = Lasso(alpha = (i/10))   # Instanciating a Lasso Regression with Alpha = (i/100)
    error.append(np.mean(cross_val_score(lasso_linreg, x_train, y_train, scoring =  "neg_mean_absolute_error", cv = 3)))    
plt.plot(alpha, error)
# Above graph becomes tappers as it reaches higher values of Alpha. 
# Below we should change the values of Alpha - try smaller values of Alpha. Try dividing by 100
alpha = []  # Clearing out Alpha and Error
error = []
for i in range(1,100):   # We need i betyween 0.1 and 10, but we cannot use Floating numbers in Range
    alpha.append(i/100)   # We are storing all values of Alpha
    lasso_linreg = Lasso(alpha = (i/100))   # Instanciating a Lasso Regression with Alpha = (i/100)
    error.append(np.mean(cross_val_score(lasso_linreg, x_train, y_train, scoring =  "neg_mean_absolute_error", cv = 3)))    
plt.plot(alpha, error)
# Above is good, Score increases and then decreases - Exactly what we want
# Below is to calculate "Exact" Alpha at which Score is Maximum
err = tuple(zip(alpha, error))  # It gives Tuple of ALL Alpha and Score
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])  # It gives DataFrame of ALL Alpha and Score
df_err[df_err.error == max(df_err.error)]   # It gives Maximum Score and corresponding Alpha Value
# With Alpha default = 1, Score was high, but with Alpha = 0.13, Score is higher
# Below is the revised Linear Regression Lasso with the best alpha value
lasso_linreg = Lasso(alpha = 0.13)
lasso_linreg.fit(x_train, y_train)
np.mean(cross_val_score(lasso_linreg, x_train, y_train, scoring =  "neg_mean_absolute_error", cv = 3))    
# Above Cross validation score is the best

#######################


# Random forest 
# We expect RF to perform excellent here, becasue a lot of Decision Trees are needed because there are a lot of 0 and 1 values
# In RF, we do not have to worry about Multi-Collinearity 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()    # THis is NOT tuned, has all default values 
rf_cv_scores = cross_val_score(rf, x_train, y_train, scoring =  "neg_mean_absolute_error", cv = 3)
rf_cv_scores_mean = np.mean(rf_cv_scores)
rf_cv_scores_mean    # This is much better values than we got from Multiple Linear Model, even with lasso. 
# Above Random Forest Corss validation score is better than Lasso, which is further better than Linear Regression
# Tune Models using griDSearchCV - In GridSearchCV, put all parameters, it runs all the models and it gives out the model with best results. 
from sklearn.model_selection import GridSearchCV
# Below "n_estimators" is Number of trees, For Cristerian- Try both MSE, MAE to see which works better. For Max features, Auto = max Features, sqrt = Square root of Maximum Features, Log2 = Taking Log of Maximum Features
# Difference b/w Normal GridSearchCV and Randomized GridSearchCV - Do Normal for better results, Do Randimized is time is less
# Setting up the parameters to build Random Forest Model 
parameters = {'n_estimators' : range(10,300,10), "criterion": ("mse", "mae"), "max_features":("auto", "sqrt", "log2")}
# Performing the GridSearch
rf_GridSearch = GridSearchCV(rf, parameters, scoring = "neg_mean_absolute_error", cv = 3)
# Fitting the Random Forest model 
rf_GridSearch.fit(x_train, y_train)
# Above is fitting the data
# Below- best_score_ is the Mean cross-validated score of the best_estimator
rf_GridSearch.best_score_
# Above is the Cross validation score, which is better than RF model with default values, without any tuning or Estimators
# Below shows the best values of the parameters that we selected like n_estimators, criterion, max_features types (Auto)
rf_GridSearch.best_estimator_



# "Predicting" using Test Datasets 
# Test all the above models, by Predicting using Test dataset and check if we get similar results 
# Use x_test here as Input, and will compare with y_test later on
predicted_linreg        = linreg.predict(x_test)
predicted_lasso_linreg  = lasso_linreg.predict(x_test)
predicted_rf_GridSearch = rf_GridSearch.best_estimator_.predict(x_test)

# "Comparing" the above Predicted values with Test dataset (y_test)
# Comparing and calculating the error
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,predicted_linreg)
mean_absolute_error(y_test,predicted_lasso_linreg)
mean_absolute_error(y_test,predicted_rf_GridSearch)
# Random Forest has Minimum error



#
######################################
## Model Production and API 
## Productionize the Model using Flask app
## Pickle the Model - Abstract it, Make the Model accessible by other programs, without retraining it. 
#import pickle
#pickle_abstracted_model = {'model': rf_GridSearch.best_estimator_}   # Converting normal model into Abstracted Model 
#pickle.dump(pickle_abstracted_model, open('pickle_model_file' + ".p", "wb" ))   # Dumping the abstracted model into a file. File name is "pickle_model_file.p"
#
#file_name = "pickle_model_file.p"
#with open(file_name, 'rb') as pickled:
#    data = pickle.load(pickled)
#    model = data['model']
#
#
#model.predict()
#model.predict(np.array(list(x_test.iloc[1,:])).reshape(1,-1))[0]
#
#list(x_test.iloc[1,:])
##################################################
