# Zillow Regression Project

##  About the Project

### Project Goals

    My goal is to produce a better model so that Zillow can more accuracatley predict the property assessed values of single family homes that had a transaction in 2017. 

### Project Description 

    New/better model needed. Zillow already has a model to predict the property assessed values of single family homes, however, they aren't fully satisfied with the result and are looking for a more accurate model. I will be looking at drivers of home value such as squarefootage, and number of rooms and bathrooms to most accurately prdict the tax value. I will be focusing on single family homes in three Soutnern California counties that had a transaction in 2017. My recommendations will be the features that prodecuded the best model and my goal is to create a model that out preform's Zillow's current one. 

### Initial Questions

    1. Does squarefootage impact home value? 

    2. Does the number of bedrooms add more value?

    3. Does the number of bathrooms add more value?

    4. Why would a house cost more than another with the same physical features? 
        - Do home values vary drastically by county?

### Data Dictionary
    A list of the variables in the dataframe and their meaning. 

    | Variable       | Description                         |
    | -------------- | ----------------------------------- |
    | num_beds       | The number of bedrooms a house has  | 
    | num_baths      | The number of bathrooms a house has |
    | square_footage | The square feet of the house        |
    | tax_value      | The tax value of the house          |
    | year_built     | The year the house was built        |
    | county_code    | The county the house is in          |

### Steps to Reproduce 

        1. You will need to import everything listed in imports used
        2. You will need a env file with your username, password, and host giving you access to Codeup's SQL server (make sure you also have a .gitignore in your github repo to protect your env file!)
        3. Clone this repo containing the Zelco_Regression_Report as well as my wrangle.py, explore.py, and model.py.
        4. That should be all you need to do run the Zillow_Regression_Report!

### The Plan 

        - Wrangle
            - create aqucire.py
                - will need to use properties_2017, predictions_2017, and propertylandusetype tables in SQL database (joins)
                - acquire transactions in 2017 only
            - create prepare.py
            - (can be one wrangle.py)
        - Explore
            - 4 vizualizations of drivers
            - 2 statistical tests of drivers

        - Model
            - for mvp (features: square feet of the home,
                                 number of bedrooms, 
                                 number of bathrooms)
            - target : property's assessed value (taxvaluedollarcnt)
            - 3 best models
            - on best model provide chart of how it preformed on the test sample

        - Refine (Report)
            - state what states and counties the homes are located in (fips)
            - project overview, goals, conclusion (did reach goal?, key findings, recommendations, and next steps)
            - Make sure markdown is clear on it's own.
            - Make sure all code is commented out. 

        - Deliver
            - README.md
            - acquire.py
            - prepare.py
            - working notebook(s)
            - report notebook
            - presentaion of report notebook

## Conclusion:
I have created a model that predicts home values better than the baseline does by $21,590.45.

The features I used to create this model were: square footage, the number of bathrooms, and the number of bedrooms.

## With more time...
With more time I would use county as a feature to build new models with hopfully greater accuracy.