#!/usr/bin/env python
# coding: utf-8

# # Python Task 1

# ### Question 1: Car Matrix Generation
# Under the function named generate_car_matrix write a logic that takes the dataset-1.csv as a DataFrame. Return a new DataFrame that follows the following rules:
# 
# values from id_2 as columns
# 
# values from id_1 as index
# 
# dataframe should have values from car column diagonal values should be 0

# In[1]:


import numpy as np

import pandas as pd


# In[2]:


df = pd.read_csv("dataset-1.csv")
df


# In[42]:


df.head(20)


# In[4]:


df.info()


# In[5]:


null_counts = df.isnull().sum()

# Display the count of null values in each column
print(null_counts)


# In[13]:


def generate_car_matrix(df):
    # Pivot the DataFrame
    pivot_df = df.pivot(index='id_1', columns='id_2', values='car')

    # Keep only the rows and columns you want (in this case, a 9x9 submatrix)
    pivot_df = pivot_df.iloc[:9, :9]

    # Set diagonal elements to zero
    np.fill_diagonal(pivot_df.values, 0)

    return pivot_df

# Use the function to create the desired matrix
result_matrix = generate_car_matrix(df)

# Display the result
print(result_matrix)


# ### Question 2: Car Type Count Calculation
# Create a Python function named get_type_count that takes the dataset-1.csv as a DataFrame. Add a new categorical column car_type based on values of the column car:
# 
# low for values less than or equal to 15,
# 
# medium for values greater than 15 and less than or equal to 25,
# 
# high for values greater than 25.
# 
# Calculate the count of occurrences for each car_type category and return the result as a dictionary. Sort the dictionary alphabetically based on keys.

# In[15]:


def get_type_count(dataset):
    # Add a new categorical column 'car_type' based on the 'car' column
    dataset['car_type'] = pd.cut(dataset['car'],
                                 bins=[float('-inf'), 15, 25, float('inf')],
                                 labels=['low', 'medium', 'high'],
                                 right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = dataset['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

# dataset
df = pd.read_csv('dataset-1.csv')

# Call the function and print the result
result = get_type_count(df)
print(result)


# # Question 3: Bus Count Index Retrieval

# ### Create a Python function named get_bus_indexes that takes the dataset-1.csv as a DataFrame. The function should identify and return the indices as a list (sorted in ascending order) where the bus values are greater than twice the mean value of the bus column in the DataFrame.

# In[19]:


def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()
    
    # Find indices where 'bus' values are greater than twice the mean
    bus_indices = df[df['bus'] > 2 * bus_mean].index
    
    # Convert indices to a sorted list
    sorted_bus_indices = sorted(bus_indices)
    
    return sorted_bus_indices

result1=get_bus_indexes(df)
print(result1)


# # Question 4: Route Filtering

# ### Create a python function filter_routes that takes the dataset-1.csv as a DataFrame. The function should return the sorted list of values of column route for which the average of values of truck column is greater than 7.

# In[59]:


import pandas as pd

def filter_routes(df):
    # Read the CSV file into a DataFrame
    df = pd.read_csv("dataset-1.csv")

    # Calculate the average of the 'truck' column for each unique 'route'
    avg_truck_by_route = df.groupby('route')['truck'].mean()

    # Filter routes where the average truck value is greater than 7
    selected_routes = avg_truck_by_route[avg_truck_by_route > 7]

    # Return the sorted list of selected routes
    return sorted(selected_routes.index.tolist())


result = filter_routes(file_path)
print(result)


# # Question 5: Matrix Value Modification

# ### Create a Python function named multiply_matrix that takes the resulting DataFrame from Question 1, as input and modifies each value according to the following logic:
# 
# If a value in the DataFrame is greater than 20, multiply those values by 0.75,
# 
# If a value is 20 or less, multiply those values by 1.25.
# 
# The function should return the modified DataFrame which has values rounded to 1 decimal place.

# In[ ]:





# In[55]:


def modify_distances(pivot_df):
    modified_pivot_df = pivot_df.copy()  # Create a copy to avoid modifying the original DataFrame
    
    # Iterate over rows and columns
    for row_index, row in pivot_df.iterrows():
        for col_index, value in row.items():
            # Apply the specified conditions
            if value > 20:
                modified_pivot_df.at[row_index, col_index] = round(value * 0.75, 1)
            else:
                modified_pivot_df.at[row_index, col_index] = round(value * 1.25, 1)
    
    return modified_pivot_df


# In[56]:


result4=modify_distances(pivot_df)
print(result4)


# # Question 6: Time Check

# ### You are given a dataset, dataset-2.csv, containing columns id, id_2, and timestamp (startDay, startTime, endDay, endTime). The goal is to verify the completeness of the time data by checking whether the timestamps for each unique (id, id_2) pair cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).
# 
# Create a function that accepts dataset-2.csv as a DataFrame and returns a boolean series that indicates if each (id, id_2) pair has incorrect timestamps. The boolean series must have multi-index (id, id_2).

# In[61]:


df3 = pd.read_csv("dataset-2.csv")
df3


# In[ ]:


def check_time_completeness(df3):
    # Load the dataset from the CSV file
    df3 = pd.read_csv("dataset-2.csv")

    # Combine startDay and startTime into a single datetime column
    df3['start_datetime'] = pd.to_datetime(df3['startDay'] + ' ' + df3['startTime'])
    # Combine endDay and endTime into a single datetime column
    df3['end_datetime'] = pd.to_datetime(df3['endDay'] + ' ' + df3['endTime'])

    # Create a DataFrame with all possible time slots for a 7-day week
    all_time_slots = pd.date_range(start='00:00:00', end='23:59:59', freq='15T')
    all_days = pd.date_range(start='00:00:00', end='23:59:59', freq='D')[:7]
    all_combinations = pd.MultiIndex.from_product([all_days, all_time_slots], names=['day', 'time'])

    # Create a DataFrame to store the availability information
    availability_df3 = pd.DataFrame(index=all_combinations)

    # Iterate over unique (id, id_2) pairs
    result = []
    for (id_value, id_2_value), group_df3 in df3.groupby(['id', 'id_2']):
        # Check if the availability covers a full 24-hour period and spans all 7 days
        covered_time_slots = pd.date_range(start=group_df3['start_datetime'].min(), end=group_df3['end_datetime'].max(), freq='15T')
        is_complete = covered_time_slots.equals(all_time_slots)
        result.append((id_value, id_2_value, is_complete))

    # Create a DataFrame with the results
    result_df3 = pd.DataFrame(result, columns=['id', 'id_2', 'is_complete'])
    result_df3.set_index(['id', 'id_2'], inplace=True)

    return result_df['is_complete']

# Example usage:
df3 = 'dataset-2.csv'
result_series = check_time_completeness(df3)
print(result_series)


# In[ ]:




