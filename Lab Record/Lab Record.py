
LAB_RECORDS/Program_01(b).py
@@ -0,0 +1,16 @@
 def temperature_converter():
 	print("Temperature converter")
 	temp = float(input("Enter temperature:"))
 	print("Convert to:")
 	print("1.Celcius to Fahrenheit")
 	print("2.Fahrenheit to Celcius")
 	choice = input("ENter choice(1/2):")
 	if choice == '1':
 		fahrenheit = (temp * 9/5) + 32
 		print(F"{temp}^C is {fahrenheit}^F")
 	elif choice == '2':
 		celcius = (temp - 32) * 5/9
 		print(f"{temp}^F is {celcius}^C")
 	else:
 		print("Invalid Input")
 temperature_converter()

LAB_RECORDS/Program_01(c).py
@@ -0,0 +1,16 @@
 def find_largest():
     print("Find the largest of three numbers")
     num1 = float(input("Enter first number:"))
     num2 = float(input("Enter second number:"))
     num3 = float(input("Enter third number:"))
     
     if num1>=num2 and num1>=num3:
         largest = num1
     elif num2>=num1 and num2>=num3:
         largest = num2
     else:
         largest = num3
         
     print(f"The largest number is {largest}")
     
 find_largest()

LAB_RECORDS/Program_02(a).py

@@ -0,0 +1,11 @@
 def factorial(n):
 
     if n == 0 or n == 1:
      return 1
     else:
         return n * factorial(n-1)
 
 #Test the function
 
 num = int(input("Enter a number to find its factorial: "))
 print (f"The factorial of {num} is {factorial(num)}"
       
LAB_RECORDS/Program_02(b).py 
@@ -0,0 +1,6 @@
 def word_count(sentence):
     words = sentence.split()
     return len(words)
 #Test the function
 sentence = input("Enter a sentence:")
 print(f"The number of words in the sentence is :
        {word_count(sentence)}")"

LAB_RECORDS/Program_03(a).py
@@ -0,0 +1,44 @@
 def list_operations():
     my_list = []
 
     while True:
         print("\nList Operations:")
         print("1. Insert an element")
         print("2. Delete an element")
         print("3. Find an element")
         print("4. Display list")
         print("5. Exit")
 
         choice = int(input("Enter your choice: "))
 
         if choice == 1:
             element = input("Enter element to insert: ")
             my_list.append(element)
             print(f"Element '{element}' inserted.")
 
         elif choice == 2:
             element = input("Enter element to delete: ")
             if element in my_list:
                 my_list.remove(element)
                 print(f"Element '{element}' deleted.")
             else:
                 print(f"Element '{element}' not found.")
 
         elif choice == 3:
             element = input("Enter element to find: ")
             if element in my_list:
                 print(f"Element '{element}' found.")
             else:
                 print(f"Element '{element}' not found.")
 
         elif choice == 4:
             print(f"Current list: {my_list}")
 
         elif choice == 5:
             print("Exiting program...")
             break
 
         else:
             print("Invalid choice, please try again.")
 
 list_operations()

 LAB_RECORDS/Program_03(b).py

@@ -0,0 +1,8 @@
 def merge_dictionaries():
     dict1 = eval(input("Enter the first dictionary: "))
     dict2 = eval(input("Enter the second dictionary: "))
 
     merged_dict = {**dict1, **dict2}
     print("\nMerged Dictionary:", merged_dict)
 
 merge_dictionaries()

LAB_RECORDS/Program_04.py

@@ -0,0 +1,11 @@
 import pandas as pd
 # Create a DataFrame from a dictionary
 data = {
     "Name": ["Ram", "Robert", "Rahim"],
     "Age": [25, 30, 35],
     "City": ["Ayodya", "Chennai", "Delhi"],
 }
 df = pd.DataFrame(data)
 print(df)
 print(df.shape)
 print(len(df))

 LAB_RECORDS/Program_06(a).py
 @@ -0,0 +1,27 @@
 import numpy as np
 
 # Creating a 1D NumPy array
 array_1d = np.array([1, 2, 3, 4, 5, 6])
 
 print("1D Array:")
 print(array_1d)
 
 # Reshaping the 1D array to a 2x3 2D array
 array_2d = array_1d.reshape(2, 3)
 
 print("\nReshaped to 2D Array (2x3):")
 print(array_2d)
 
 # Accessing elements using indexing
 print("\nElement at position (1, 2):", array_2d[1, 2])
 
 # Modifying an element
 array_2d[0, 1] = 10
 
 print("\nModified Array (After changing element at position (0,1) to 10):")
 print(array_2d)
 
 # Calculating the sum of the array elements
 array_sum = np.sum(array_2d)
 
 print("\nSum of all elements in the array:", array_sum)

 LAB_RECORDS/Program_06(b).py
 @@ -0,0 +1,25 @@
 import numpy as np
 
 # Creating two matrices
 matrix_A = np.array([[1, 2], [3, 4]])
 matrix_B = np.array([[5, 6], [7, 8]])
 
 # Matrix Addition
 matrix_sum = np.add(matrix_A, matrix_B)
 print("Matrix Addition (A + B):")
 print(matrix_sum)
 
 # Matrix Multiplication (Element-wise)
 matrix_product_elementwise = np.multiply(matrix_A, matrix_B)
 print("\nElement-wise Matrix Multiplication (A * B):")
 print(matrix_product_elementwise)
 
 # Matrix Dot Product (Matrix Multiplication)
 matrix_dot_product = np.dot(matrix_A, matrix_B)
 print("\nMatrix Dot Product (A . B):")
 print(matrix_dot_product)
 
 # Transposing a matrix
 matrix_transpose = np.transpose(matrix_A)
 print("\nTranspose of Matrix A:")
 print(matrix_transpose)

 LAB_RECORDS/Program_07(a(i)).py
 @@ -0,0 +1,15 @@
 import pandas as pd
 
 # Creating a DataFrame from a dictionary
 data = { 'Name': ['Alice', 'Bob', 'Charlie', 'David'],
 'Age': [25, 30, 35, 40],
 'Salary': [50000, 60000, 70000, 80000]}
 
 df= pd.DataFrame(data)
 print("DataFrame from Scratch:")
 print(df)
 
 # Adding a new column 'Bonus' (10% of Salary)
 df['Bonus'] = df['Salary'] * 0.10
 print("\nDataFrame after adding 'Bonus' column:")
 print(df)

 LAB_RECORDS/Program_07(a(ii)).py
 @@ -0,0 +1,12 @@
 import pandas as pd
 
 # Reading data from a CSV file
 df=pd.read_csv('lexperience.csv')
 
 # Display the first 6 rows
 print("First 6 rows of the DataFrame:")
 print(df.head(6))
 
 # Displaying column names and data types
 print("\nColumn names and data types:")
 print(df.info())

 LAB_RECORDS/Program_07(b).py
 @@ -0,0 +1,17 @@
 import pandas as pd
 
 # Loading a CSV file into a DataFrame
 df = pd.read_csv(r"CASHANTIR VU\Programs\Minors\IDA\Activities\Datasets\6Mednull_values.csv")
 
 print("Original DataFrame:")
 print(df)
 
 # Identifying missing values
 print("\nMissing values in the DataFrame:")
 print(df.isnull().sum())
 
 # Filling missing values in the 'Q1' column with the mean
 df['Q1'].fillna(df['Q1'].mean(), inplace=True)
 
 print("\nDataFrame after filling missing values in 'Q1' column:")
 print(df)

 LAB_RECORDS/Program_08.py
 @@ -0,0 +1,16 @@
 import pandas as pd
 df = pd.read_csv('1experience.csv')
 mean_value = df['YearsExperience'].mean()
 median_value = df['YearsExperience'].median()
 mode_value = df['YearsExperience'].mode()[0]
 min_value = df['YearsExperience'].min()
 max_value = df['YearsExperience'].max()
 variance_value = df['YearsExperience'].var()
 std_dev_value = df['YearsExperience'].std()
 print(f"Mean: {mean_value}")
 print(f"Median: {median_value}")
 print(f"Mode: {mode_value}")
 print(f"Minimum: {min_value}")
 print(f"Maximum: {max_value}")
 print(f"Variance: {variance_value}")
 print(f"Standard Deviation: {std_dev_value}")

 LAB_RECORDS/Program_09.py
 @@ -0,0 +1,24 @@
 import pandas as pd
 
 # Load the CSV file
 df = pd.read_csv('lexperience.csv')
 
 # Statistical summary of the data using describe()
 description = df.describe()
 
 # Calculating quantiles
 quantiles = df['YearsExperience'].quantile([0.25, 0.5, 0.75])
 
 # Calculating skewness and kurtosis
 skewness = df['YearsExperience'].skew()
 kurtosis = df['YearsExperience'].kurt()
 
 # Calculating value counts for unique values in the 'YearsExperience' column
 value_counts = df['YearsExperience'].value_counts()
 
 # Displaying the results
 print("Statistical Summary:\n", description)
 print("\nQuantiles:\n", quantiles)
 print("\nSkewness:", skewness)
 print("Kurtosis:", kurtosis)
 print("\nValue Counts:\n", value_counts)

 LAB_RECORDS/Program_10.py
 @@ -0,0 +1,36 @@
 import pandas as pd
 df = pd.read_csv('4laptops.csv')
 df
 
 # Onehot encoding without sci-kit learn library
 pd.get_dummies(df)
 from sklearn.preprocessing import OneHotEncoder
 
 # creating instance of one-hot-encoder
 enc = OneHotEncoder()
 enc_data = enc.fit_transform(df[['Category']])
 enc_data # gives an object as output
 
 # Identifying the list of categories, one-hot encoding is considering as columns
 enc.categories_
 enc_data.toarray() # converting object into array using toarray function
 
 #creating a dataframe without giving column names
 enc_df = pd.DataFrame(enc_data.toarray())
 enc_df
 
 # creating a dataframe with giving column names
 enc_df = pd.DataFrame(enc_data.toarray(), columns=['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation'])
 enc_df
 df1 = df.join(enc_df) # joining df and enc_df
 df1
 
 # Using label encoding
 from sklearn.preprocessing import LabelEncoder
 le = LabelEncoder()
 
 # Fit and transform the data
 df['RAM'] = le.fit_transform(df['RAM'])
 df
 
 le.classes_

 LAB_RECORDS/Program_11.py

 @@ -0,0 +1,11 @@
 import pandas as pd
 
 df = pd.read_csv(r"3Salary_Data.csv")
 df
 
 from sklearn.preprocessing import StandardScaler
 
 scaler = StandardScaler()
 scaled_data = scaler.fit_transform(df)
 scaled_df1 = pd.DataFrame(scaled_data, columns=df.columns)
 scaled_df1

 LAB_RECORDS/Program_12(a).py
@@ -0,0 +1,23 @@
 import pandas as pd
 import matplotlib.pyplot as plt
 
 df2 = pd.read_csv("4laptops.csv")
 
 plt.boxplot(df2["Screen_size_inches"], notch=True, vert=False, showmeans=True, 
             sym="*", patch_artist=True, widths=0.1)
 
 plt.xlabel('Screen Size (inches)')
 plt.title('Boxplot of Screen Size')
 plt.show()
 
 plt.boxplot(df2['Weight_kg'])
 
 plt.xlabel('Weight (kg)')
 plt.title('Boxplot of Laptop Weight')
 plt.show()
 
 import seaborn as sns
 sns.boxplot(x=df2['Weight_kg'])
 plt.show()
 sns.boxplot(x=df2['Weight_kg'], y=df2['RAM'])
 plt.show()

 LAB_RECORDS/Program_12(b).py
 @@ -0,0 +1,18 @@
 #Let's first load the dataset to understand its structure and contents before proceeding with identifying outliers using IQR.
 
 import pandas as pd
 
 #Load the dataset 
 data=pd.read_csv('4laptops.csv') 
 data
 
 q3=data['Weight_kg'].quantile(0.75)
 q1=data['Weight_kg'].quantile(0.25)
 
 iqr-q3-q1
 
 lower_bound = q1 - 1.5*iqr
 upper_bound=q3 + 1.5*iqr
 outliers = data[(data['Weight_kg'] <lower_bound) | (data['Weight_kg'] > upper_bound)]
 print('lower bound, upper bound and iqr values are: ',lower_bound, upper_bound, iqr)
 print('No.of outliers are: ', outliers.shape[0])

 LAB_RECORDS/Program_13.py
 @ -0,0 +1,87 @@
 # Importing required libraries
 import pandas as pd
 
 # Creating the employee_details DataFrame
 employee_details = pd.DataFrame({
     'EmployeeID': [101, 102, 103, 104, 105],
     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
     'Department': ['HR', 'Engineering', 'Engineering', 'HR', 'Marketing']
 })
 
 # Creating the employee_salaries DataFrame
 employee_salaries = pd.DataFrame({
     'EmployeeID': [101, 102, 103, 104, 105],
     'Salary': [50000, 70000, 80000, 55000, 60000]
 })
 
 # Creating the sales_region_1 DataFrame
 sales_region_1 = pd.DataFrame({
     'Date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
     'Region': ['North'] * 5,
     'Sales': [250, 300, 200, 400, 350]
 })
 
 # Creating the sales_region_2 DataFrame
 sales_region_2 = pd.DataFrame({
     'Date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
     'Region': ['South'] * 5,
     'Sales': [300, 320, 280, 360, 310]
 })
 
 # Display the datasets
 print("Employee Details:")
 print(employee_details)
 
 print("\nEmployee Salaries:")
 print(employee_salaries)
 
 print("\nSales Region 1:")
 print(sales_region_1)
 
 print("\nSales Region 2:")
 print(sales_region_2)
 
 # Grouping by department and calculating average salary
 avg_salary_per_dept = employee_details.merge(employee_salaries, on='EmployeeID') \
     .groupby('Department')['Salary'].mean()
 
 print("\nAverage Salary per Department:")
 print(avg_salary_per_dept)
 
 # Merging two DataFrames on the EmployeeID column
 merged_data = pd.merge(employee_details, employee_salaries, on='EmployeeID', how='inner')
 
 print("\nMerged Employee Data:")
 print(merged_data)
 
 # Creating the stock_prices DataFrame with 'Date' as the index
 stock_prices = pd.DataFrame({
     'Date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
     'Price': [150, 155, 152, 158, 160]
 })
 stock_prices.set_index('Date', inplace=True)
 
 # Creating the market_volume DataFrame with 'Date' as the index
 market_volume = pd.DataFrame({
     'Date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
     'Volume': [1000, 1100, 1050, 1150, 1200]
 })
 market_volume.set_index('Date', inplace=True)
 
 print("\nStock Prices Data:")
 print(stock_prices)
 
 print("\nMarket Volume Data:")
 print(market_volume)
 
 # Joining market_volume to stock_prices based on their index
 joined_data = stock_prices.join(market_volume, how='inner')
 
 print("\nJoined Stock Prices and Market Volume Data:")
 print(joined_data)
 
 # Concatenating DataFrames vertically
 consolidated_sales = pd.concat([sales_region_1, sales_region_2], axis=0)
 
 print("\nConsolidated Sales Data:")
 print(consolidated_sales)

 LAB_RECORDS/Program_14(a).py
 @@ -0,0 +1,13 @@
 import pandas as p
 import seaborn as sns
 import matplotlib.pyplot as plt
 
 df=pd.read_csv(r"5ds_salaries.csv")
 df.columns
 df.skew(numeric_only=True)
 df.kurt(numeric_only=True)
 sns.histplot(df["salary"])
 plt.show()
 sns.histplot(df["salary"],kde=True)
 plt.show()
 sns.distplot(df["salary"])

 LAB_RECORDS/Program_14(b).py
 @@ -0,0 +1,7 @@
 import pandas as pd
 import seaborn as sns
 import matplotlib.pyplot as plt
 
 df2 = pd.read_csv(r"4laptops.csv")
 sns.pairplot(df2)
 plt.show()

 LAB_RECORDS/Program_15.py
 @@ -0,0 +1,5 @@
 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns
 df2 = pd.read_csv(r"4laptops.csv")
 sns.heatmap(df2.corr(numeric_only = True), annot = True)

 LAB_RECORDS/Program_16(a).py
 @@ -0,0 +1,11 @@
 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns
 df = pd.read_csv(r"3Salary_Data.csv")
 plt.plot(df['YearsExperience'], df['Salary'])
 plt.ylabel('YearsExperience')
 plt.xlabel('Index')
 plt.legend()
 plt.show()
 sns.lineplot(y = df['YearsExperience'], x = df['Salary'])
 plt.show()

 LAB_RECORDS/Program_16(b).py
 @@ -0,0 +1,11 @@
 import pandas as pd
 import seaborn as sns
 import matplotlib.pyplot as plt
 df = pd.read_csv(r"3Salary_Data.csv")
 plt.scatter(df['YearsExperience'], df['Salary'])
 plt.ylabel('YearsExperience')
 plt.xlabel('Index')
 plt.legend()
 plt.show()
 sns.scatterplot(x = df['Salary'], y = df['YearsExperience'])
 plt.show()

 LAB_RECORDS/Program_17.py
@@ -0,0 +1,7 @@
 import pandas as pd
 import seaborn as sns
 import matplotlib.pyplot as plt
 df4 = pd.read_csv(r"5ds_salaries.csv")
 plt.bar(df4['experience_level'], df4['salary_in_usd'], width = 0.5, edgecolor = 'white', linewidth = 0.4)
 plt.show()
 sns.barplot(x = df4['experience_level'], y = df4['salary_in_usd'])
