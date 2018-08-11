---
layout: post
current: post
cover:  assets/images/bar-plot-matplotlib/bar-cover.png
navigation: True
title: Customizing Bar Plots in Matplotlib
date: 2018-07-08 10:00:00
tags: [Data Visualization, Python, Matplotlib]
class: post-template
subclass: 'post tag-data-viz'
author: bmanohar16
---

Bar charts are good to visualize grouped data values with counts. In this post, we will see how to customize the default plot theme of matplotlib.pyplot to our personal aesthetics and design choices.

### Import libraries

```python
# Import libraries
import pandas as pd

from matplotlib import pyplot as plt
%matplotlib inline
```

### Custom Font in plots

```python
from matplotlib import rcParams

# Custom Font
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
rcParams['font.size'] = 14
```

### Load data into pandas dataframe

```python
# Read CSV into pandas
filepath = '/Users/bala/Dev/data/LendingClub/LoanStats_2017Q1.csv'
loan_df = pd.read_csv(filepath, low_memory=False)
```

### Loan Reason

```python
# Loan Reason Data Frame
title_cnt = loan_df.title.value_counts().reset_index()
title_cnt
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Debt consolidation</td>
      <td>54807</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Credit card refinancing</td>
      <td>21017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Home improvement</td>
      <td>7058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Other</td>
      <td>6130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Major purchase</td>
      <td>2167</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Medical expenses</td>
      <td>1353</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Business</td>
      <td>1146</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Car financing</td>
      <td>1121</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vacation</td>
      <td>780</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Moving and relocation</td>
      <td>704</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Home buying</td>
      <td>429</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Green loan</td>
      <td>67</td>
    </tr>
  </tbody>
</table>
</div>

### Basic Horizontal Bar Plot

```python
# Figure Size
fig, ax = plt.subplots(figsize=(10,7))

# Horizontal Bar Plot
ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1])

# Show Plot
plt.show()
```

![png](assets/images/bar-plot-matplotlib/default.png)

### Remove axes, ticks, add grids and show top counts first

```python
# Figure Size
fig, ax = plt.subplots(figsize=(10,7))

# Horizontal Bar Plot
ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='crimson')

# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

# Show top values 
ax.invert_yaxis()

# Show Plot
plt.show()
```

![png](assets/images/bar-plot-matplotlib/output_6_0.png)

## Final Plot

```python
# Figure Size
fig, ax = plt.subplots(figsize=(10,7))

# Horizontal Bar Plot
ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='crimson')

# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('LendingClub 2017 Q1 - Top Reasons for Requesting Loan',
             loc='left', pad=10)

# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+500, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')

# Add Text watermark
fig.text(0.9, 0.15, '@bmanohar16', fontsize=12, color='grey',
         ha='right', va='bottom', alpha=0.5)

# Save Plot as image
fig.savefig('Top Reasons for Requesting Loan.png', dpi=100,
            bbox_inches='tight')

# Show Plot
plt.show()
```

![png](assets/images/bar-plot-matplotlib/output_7_0.png)

from default theme...

![png](assets/images/bar-plot-matplotlib/default.png)

Thank you for reading.