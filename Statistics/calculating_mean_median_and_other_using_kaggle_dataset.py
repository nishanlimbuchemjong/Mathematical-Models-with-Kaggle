import pandas as pd

# loading the sales dataset
data = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')

# displaying the first few rows to understand the dataset structure
print(data.head())

# Output:
"""
   ORDERNUMBER  QUANTITYORDERED  PRICEEACH  ORDERLINENUMBER    SALES  ... COUNTRY TERRITORY  CONTACTLASTNAME  CONTACTFIRSTNAME  DEALSIZE
0        10107               30      95.70                2  2871.00  ...     USA       NaN               Yu              Kwai    
 Small
1        10121               34      81.35                5  2765.90  ...  France      EMEA          Henriot              Paul    
 Small
2        10134               41      94.74                2  3884.34  ...  France      EMEA         Da Cunha            Daniel    
Medium
3        10145               45      83.26                6  3746.70  ...     USA       NaN            Young             Julie    
Medium
4        10159               49     100.00               14  5205.27  ...     USA       NaN            Brown             Julie    
Medium

[5 rows x 25 columns]
"""

# checking the columns of the dataset
print(data.columns)
# Output:
"""
Index(['ORDERNUMBER', 'QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER',
       'SALES', 'ORDERDATE', 'STATUS', 'QTR_ID', 'MONTH_ID', 'YEAR_ID',
       'PRODUCTLINE', 'MSRP', 'PRODUCTCODE', 'CUSTOMERNAME', 'PHONE',
       'ADDRESSLINE1', 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE',
       'COUNTRY', 'TERRITORY', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME',
       'DEALSIZE'],
      dtype='object')

"""

# choosing the column to analyze
column_data = data['SALES']
print(column_data)
# Output:
"""
0       2871.00
1       2765.90
2       3884.34
3       3746.70
4       5205.27
         ...
2818    2244.40
2819    3978.51
2820    5417.57
2821    2116.16
2822    3079.44
Name: SALES, Length: 2823, dtype: float64
"""

# Calculating mean, median, mode, and variance
mean_value = column_data.mean()
median_value = column_data.median()
mode_value = column_data.mode()[0]  # mode have multiple values, we take the first one
variance_value = column_data.var()


# Printing out the results:
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")
print(f"Variance: {variance_value}")

# Output:
"""
Mean: 3553.889071909316
Median: 3184.8
Mode: 3003.0
Variance: 3392467.067743291

"""
