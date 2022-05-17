import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
# Add an overweight column to the data. To determine if a person is overweight, first calculate their BMI by 
# dividing their weight in kilograms by the square of their height in meters. If that value is > 25 then the 
# person is overweight. Use the value 0 for NOT overweight and the value 1 for overweight.
df_bmi = df['weight'] / ((df['height']/100) ** 2)
df['overweight'] = np.where(df_bmi > 25, 1, 0)
# df_bmi = (df['weight'] / ((df['height']/100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# Normalize data by making 0 always good and 1 always bad. 
# If the value of 'cholesterol' or 'gluc' is 1, make the value 0. 
# If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
print(df.head(8))

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    print(df_cat.head(8))

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. 
    # You will have to rename one of the columns for the catplot to work correctly.
    df_cat['total'] = 1
    print(df_cat.head(8))
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()
    print(df_cat.head(8))

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind='bar').fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data. 
    # Filter out the following patient segments that represent incorrect data:
    # - diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
    # - height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
    # - height is more than the 97.5th percentile 
    # - weight is less than the 2.5th percentile
    # - weight is more than the 97.5th percentile
    df_heat = df[
      (df['ap_lo'] <= df['ap_hi']) &
      (df['height'] >= df['height'].quantile(0.025)) &
      (df['height'] <= df['height'].quantile(0.975)) &
      (df['weight'] >= df['weight'].quantile(0.025)) &
      (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=1, square=True, center=0.08, cbar_kws={"shrink": .5})


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
