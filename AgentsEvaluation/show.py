import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Data configuration
data_str = """Store_Number,SKU_Coded,Product_Class_Code,Sold_Date,Qty_Sold,Total_Sale_Value,On_Promo
1320,6173050,22875,2021-11-01,1,4.99,0
1320,6174250,22875,2021-11-01,1,0.89,0
1320,6176200,22975,2021-11-01,2,99.98,0
1320,6176800,22800,2021-11-01,1,14.97,0
1320,6177250,22975,2021-11-01,1,6.89,0
1320,6177300,22800,2021-11-01,1,9.99,0
1320,6177350,22800,2021-11-01,2,16.98,0
1320,6177700,22875,2021-11-01,1,3.19,0
1320,6178000,22875,2021-11-01,2,6.38,0
1320,6178250,22800,2021-11-01,1,16.59,0
1320,6179250,24400,2021-11-01,1,14.99,0
1320,6179300,22800,2021-11-01,2,9.98,0
1320,6179400,24400,2021-11-01,2,29.98,0
1320,6179450,24400,2021-11-01,1,14.99,0
1320,6179500,24400,2021-11-01,1,14.99,0
1320,6179750,22800,2021-11-01,2,39.98,0
1320,6180550,22975,2021-11-01,1,15.99,0
1320,6182050,22975,2021-11-01,1,7.99,0
1320,6183750,22850,2021-11-01,3,38.97,0
1320,6184100,22975,2021-11-01,3,59.97,0
1320,6188550,22950,2021-11-01,2,15.98,0
1320,6190050,24425,2021-11-01,5,19.95,0
1320,6190150,24425,2021-11-01,1,8.99,0
1320,6190200,24425,2021-11-01,1,8.99,0
1320,6190250,24425,2021-11-01,1,7.99,0
1320,6190350,22950,2021-11-01,1,6.99,0
1320,6190400,22950,2021-11-01,1,6.99,0
1320,6193750,22875,2021-11-01,1,6.99,0
1320,6195350,24375,2021-11-01,1,16.99,0
1320,6195800,22850,2021-11-01,3,25.72,1
"""

# Load data
data = pd.read_csv(StringIO(data_str))

# Group by Sold_Date and SKU_Coded to sum the Total_Sale_Value
grouped_data = data.groupby(['Sold_Date', 'SKU_Coded'])['Total_Sale_Value'].sum().reset_index()

# Plotting
plt.figure(figsize=(12, 6))
for sku in grouped_data['SKU_Coded'].unique():
    sku_data = grouped_data[grouped_data['SKU_Coded'] == sku]
    plt.plot(sku_data['Sold_Date'], sku_data['Total_Sale_Value'], marker='o', label=sku)

plt.title('A bar chart of sales by product SKU. Put the product SKU on the x-axis and the sales on the y-axis.')
plt.xlabel('Date')
plt.ylabel('Total Sale Value')
plt.xticks(rotation=45)
plt.legend(title='SKU')
plt.tight_layout()
plt.show()