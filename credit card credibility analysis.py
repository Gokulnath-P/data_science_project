import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

df = pd.read_excel("credit_data.xlsx")

# Data Exploration
print(df.head())
print(df.info())
# Scatter plots
sns.scatterplot(x='AGE', y='LIMIT_BAL', hue='default', data=df)
plt.title('Scatter Plot: Age vs. Credit Limit by Default')
plt.show()

# Pair plot
sns.pairplot(df[['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1', 'default']], hue='default')
plt.title('Pair Plot')
plt.show()

# KDE plots
plt.figure(figsize=(5, 6))
sns.kdeplot(df[df['default'] == 'Y']['LIMIT_BAL'], label='Default', shade=True)
sns.kdeplot(df[df['default'] == 'N']['LIMIT_BAL'], label='No Default', shade=True)
plt.title('KDE Plot: Credit Limit Distribution by Default')
plt.xlabel('Credit Limit')
plt.ylabel('Density')
plt.legend()
plt.show()

# Kolmogorov-Smirnov Test
features_to_test = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']  # Select features for KS test

for feature in features_to_test:
    stat, p_value = ks_2samp(df[df['default'] == 'Y'][feature], df[df['default'] == 'N'][feature])
    print(f"KS Test for feature {feature}: Statistic={stat}, p-value={p_value}")

print("""KS Test Statistic:

The KS test statistic measures the maximum discrepancy (or maximum absolute difference) between the cumulative 
distribution functions (CDFs) of the two samples being compared. A larger KS test statistic indicates a greater 
difference between the distributions of the two samples. P-Value:

The p-value associated with the KS test indicates the probability of obtaining the observed KS test statistic if the 
two samples are drawn from the same underlying distribution (i.e., if there is no significant difference between the 
distributions). A smaller p-value suggests stronger evidence against the null hypothesis (i.e., the two distributions 
are the same), indicating that the observed difference is unlikely to be due to random chance.""")
print("""LIMIT_BAL:

The KS test statistic is 0.1819, indicating a moderate difference between the distributions of credit limits for 
credible and non-credible customers. The very small p-value (approximately 0) suggests strong evidence against the 
null hypothesis, indicating that the distributions of credit limits for the two groups are significantly different.""")
print("""The KS test statistic is 0.1516, indicating a relatively large difference between the distributions of the 
first payment amounts for credible and non-credible customers. The p-value (approximately 0) indicates strong 
evidence against the null hypothesis, suggesting a significant difference between the distributions.""")
