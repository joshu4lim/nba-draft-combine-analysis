# Can Draft Combine Data Predict the Next NBA Superstar?
Joshua Lim - chaelim@umich.edu

## Introduction

The National Basketball Association or the NBA is a beloved league watched by millions. The NBA offers high-level basketball and superstars to provide sports entertainment for fans. Over decades of professional basketball history, many superstars have emerged where some were initially underrated and overlooked. For instance, Stephen Curry or Nikola Jokic. This project explores whether we can predict a player's performance in the NBA using data from the NBA Draft Combine. The dataset includes physical measurements, athletic drill results, college attended, and career statistics for players active as of June 2025.

### 📌 Project Question

**Can we predict a player's professional performance measured by Player Efficiency Rating (PER) using college name and Draft Combine physical and athletic data?**

This question is useful because PER is a well-known, single-number summary of a player's performance. If teams could reliably predict PER using only combine and college data, they could make more data-driven draft decisions — potentially reducing costly draft busts. For the average basketball fan, a working model can spotlight which rookies to watch, making it easier to follow their exciting growth into superstars.

### 📂 Dataset Overview

- **Source**: NBA Draft Combine data from the official website (https://www.nba.com/stats/draft/combine)
- **Tool**: nba_api - helps to fetch data from the NBA website (https://pypi.org/project/nba_api/)
- **Number of Rows**: 286  
- **Number of Relevant Columns**: 9 core features used in modeling

### 🔍 Relevant Columns

| Column Name              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `HEIGHT_WO_SHOES`        | Player's height in inches, measured without shoes.                         |
| `WEIGHT`                 | Player's weight in pounds.                                                 |
| `WINGSPAN`               | Arm span from fingertip to fingertip in inches.                            |
| `STANDING_REACH`         | Vertical reach with feet flat on the ground.                               |
| `LANE_AGILITY_TIME`      | Time to complete lane agility drill (lower is better).                     |
| `MAX_VERTICAL_LEAP`      | Highest vertical jump achieved (in inches).                                |
| `STANDING_VERTICAL_LEAP` | Vertical jump from standing position (in inches).                          |
| `THREE_QUARTER_SPRINT`   | Time to run three-quarters of the court (lower is better).                 |
| `SCHOOL_NAME`            | Name of the college the player attended (used to infer conference strength). |
| `Player Efficiency Rating (PER)`                    | **Target** variable: Player Efficiency Rating in the NBA (higher is better). We will have to create this from a player's career points, rebounds, etc.|

---


## Data Cleaning and Exploratory Data Analysis

### 🧹 Data Cleaning

To prepare the dataset for analysis and modeling, we cleaned the data in the following ways:

- **Narrowing Down Rows:** Since there are many data about NBA players, I first decide to narrow the dataset to currently active players (as of June 2025). Next, I narrow it down further to players who particpated in the Draft Combine, as some players opt out of it. In this step, I fetch both physical measurements and drill results of each player that participated in the Combine. Finally, I fetch name of college attended and career statistics of the pool of players I have. This step doesn't narrow down the rows since it all players have a career statistic, and players who didn't go to college can just be imputed with a NaN. This step narrowed down the dataset to 484 rows.

- **Narrowing Down Columns:** 
These are the columns I initially chose and their justifications:

| `PLAYER_ID`              | Unique identifier for each player.                                          |
| `full_name`              | Name of Player.                                                             |
| `POSITION`               | Basketball position of Player. Might be useful to categorization.           |
| `HEIGHT_WO_SHOES`        | Basketball is a height sport so, this is an important factor.               |
| `WEIGHT`                 | Basketball is contact heavy, so weight is important factor to physical interactions between players. |
| `WINGSPAN`               | Arm span helps players to reach farther for both defense and offense.       |
| `STANDING_REACH`         | Helps to reach the rim, easier to dunk if longer reach.                     |
| `HAND_LENGTH`            | Helps for better control over the ball.                                     |
| `HAND_WIDTH`             | Same as above.                                                              |
| `LANE_AGILITY_TIME`      | Basketball is a fast-paced sport, helps to measure explosiveness.           |
| `MODIFIED_LANE_AGILITY_TIME`  | Same as above.                                                         |
| `THREE_QUARTER_SPRINT`   | Same as above.                                                              |
| `MAX_VERTICAL_LEAP`      | Better leap = easier to score the ball at the rim.                          |
| `STANDING_VERTICAL_LEAP` | Same as above.                                                              |
| `BENCH_PRESS`            | Helps to infer player strength.                                             |
| `SCHOOL_NAME`            | Used to infer conference strength.                                          |
| `The following features below are used to create PER`                                                  |
| `PTS`                    | Player's career points                                                      |
| `REB`                    | Player's career rebounds                                                    |
| `AST`                    | Player's career assists                                                     |
| `STL`                    | Player's career steals                                                      |
| `FGA`                    | Player's career field goal attempts                                         |
| `FGM`                    | Player's career field goal made                                             |
| `FTA`                    | Player's career free throw attempts                                         |
| `FTM`                    | Player's career free throw made                                             |
| `TOV`                    | Player's career turnovers                                                   |
| `GP`                     | Player total game played                                                    |

- **Dropping Rows and Columns:** We want to drop instead of imputing because in the narrowed down dataset, there are some players who participated in just the physical measurements and some who participated in just the drills. We drop these players since we want to look at both measurements and drill results. We also drop columns: 'BENCH_PRESS', 'MODIFIED_LANE_AGILITY_TIME', 'HAND_LENGTH', and 'HAND_WIDTH' since they had many players who had missing values. Next, we drop any straggler players that have missing values in columns besides 'SCHOOL_NAME' (since it's allowed for a player to not attended a college). Finally for duplicate player rows, we keep the first occurrence of that 'PLAYER_ID' and drop the other overlaps. The dataset is now narrowed down to 286 rows and 23 columns.

- **Filling in Outliers:** We also change school names for certain players because the nba_api fetches the wrong school name. We fill in for Jalen Williams, Ryan Rollins, and Patrick Baldwin Jr.

- **Data Types:** Some numerical columns were stored as strings (e.g. heights or weights extracted from the web API). We converted these to numeric types using `pd.to_numeric(..., errors='coerce')`.

- **Creating PER Column:** Using career statistics (PTS, REB, etc.) and the PER formula, we create the PER column. 
**PER** = (**PTS** + **REB** + **AST** + **STL** + **BLK** − *(Missed FG)* − *(Missed FT)* − **TO**) / **GP**

Where Missed FG = FGA - FGM and Missed FT = FTA - FTM

(https://en.wikipedia.org/wiki/Efficiency_(basketball)#EFF)

The final dataset is 286 x 24

#### 📌 Cleaned Data Preview

Here’s a preview of the cleaned dataset:

<div style="display: flex; justify-content: center;">
  <div style="overflow-x: auto; max-width: 90%;">
  
  <table>
    <thead>
      <tr>
        <th>PLAYER_ID</th><th>SCHOOL_NAME</th><th>STANDING_VERTICAL_LEAP</th><th>MAX_VERTICAL_LEAP</th><th>LANE_AGILITY_TIME</th><th>THREE_QUARTER_SPRINT</th><th>full_name</th><th>POSITION</th><th>HEIGHT_WO_SHOES</th><th>WEIGHT</th><th>WINGSPAN</th><th>STANDING_REACH</th><th>PTS</th><th>REB</th><th>AST</th><th>STL</th><th>BLK</th><th>FGA</th><th>FGM</th><th>FTA</th><th>FTM</th><th>TOV</th><th>GP</th><th>PER</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>203500</td><td>Pittsburgh</td><td>28.5</td><td>33</td><td>11.85</td><td>3.4</td><td>Steven Adams</td><td>C</td><td>82.75</td><td>254.5</td><td>88.5</td><td>109.5</td><td>6743</td><td>6115</td><td>1145</td><td>646</td><td>703</td><td>4786</td><td>2804</td><td>2129</td><td>1134</td><td>1083</td><td>764</td><td>14.7801</td></tr>
      <tr><td>1628389</td><td>Kentucky</td><td>33.5</td><td>38.5</td><td>11.94</td><td>3.24</td><td>Bam Adebayo</td><td>PF-C</td><td>80.75</td><td>242.6</td><td>86.75</td><td>108</td><td>8923</td><td>5024</td><td>2044</td><td>607</td><td>489</td><td>6389</td><td>3428</td><td>2598</td><td>1965</td><td>1218</td><td>567</td><td>21.649</td></tr>
      <tr><td>1630534</td><td>Kansas</td><td>32</td><td>41.5</td><td>10.88</td><td>3.13</td><td>Ochai Agbaji</td><td>SG</td><td>76.5</td><td>214.4</td><td>82</td><td>103.5</td><td>2044</td><td>795</td><td>331</td><td>168</td><td>133</td><td>1786</td><td>787</td><td>229</td><td>164</td><td>223</td><td>279</td><td>7.82796</td></tr>
      <tr><td>1641725</td><td>Creighton</td><td>29.5</td><td>34</td><td>11.6</td><td>3.35</td><td>Trey Alexander</td><td>SG</td><td>75.25</td><td>184.6</td><td>82</td><td>101.5</td><td>32</td><td>12</td><td>11</td><td>2</td><td>1</td><td>41</td><td>13</td><td>4</td><td>3</td><td>5</td><td>24</td><td>1</td></tr>
      <tr><td>1628960</td><td>Duke</td><td>32.5</td><td>40.5</td><td>10.31</td><td>3.15</td><td>Grayson Allen</td><td>SG</td><td>75</td><td>198</td><td>79.25</td><td>97</td><td>4250</td><td>1216</td><td>812</td><td>293</td><td>112</td><td>3140</td><td>1416</td><td>657</td><td>563</td><td>399</td><td>403</td><td>11.0819</td></tr>
    </tbody>
  </table>

  </div>
</div>


---

### 📊 Univariate Analysis

We first explored the **distribution of Player Height without Shoes** to understand the outcome variable.

<iframe src="plots/uni-height-plot.html" width="800" height="500" frameborder="0"></iframe>

There is almost a bell-shaped distribution with a slight right-skew for height without shoes, with most players falling between 76-80 inches, with a few outliers at each end. It suggest that height is relatively standardized across players, which may be a useful feature for making our predictive model.

---

### 📈 Bivariate Analysis

Next, we examined how **Height without Shoes** relates to **PER**.

<iframe src="plots/biv-height-by-per-plot.html" width="800" height="500" frameborder="0"></iframe>

There doesn't appears to be a **strong linear relationship** between height and PER, where most players cluster around 76-80 inches with a wide range of PER. Elite PER > 20 occurs in various heights, suggesting that height alone does not predict PER reliably. 

---

### 📊 Interesting Aggregates

We grouped players by **Position** and calculated the **median PER**.

| POSITION   |      PER |
|:-----------|---------:|
| PG-SG      | 17.5278  |
| PF-SF      | 16.756   |
| C-PF       | 14.4618  |
| SF-SG      | 13.2562  |
| PF-C       | 12.8379  |
| SG-PG      | 11.3666  |
| SF-PF      |  9.47372 |
| SG-SF      |  9.42649 |
| PF         |  9.19232 |
| C          |  9.17965 |
| PG         |  9.00727 |
| SG         |  8.5     |
| SF         |  8.24883 |

I used the median since mean is affected by outlier, which basketball has a lot of in terms of star players or bench players or rookies who barely get minutes to play. Interestingly, players with hybrid positions seemed to have the highest PER medians compared to tradition one position players, suggesting that the NBA is transitioning into a more flexible playstyle.

---

## Framing a Prediction Problem

### ✅ Prediction Problem Type

This is a **regression** problem. We are predicting a continuous numerical variable — **Player Efficiency Rating (PER)** — using a set of physical and drill measurements from the NBA Draft Combine, as well as the player’s college program.

---

### 🎯 Response Variable

The **response variable** (i.e. target variable) we are predicting is:

> **PER (Player Efficiency Rating)** — a summary statistic representing a player's overall NBA performance.

We chose PER over other stats because:
- It is a widely-used, standardized, and interpretable measure of overall NBA performance.
- It is continuous and allows us to differentiate subtle performance differences between players.
- It can be easily calculated from public career stats.

---

### 🔍 Type of Model: Predictive

We are building a **predictive model**, not an inferential one. That is, our goal is to **predict a player’s future NBA performance** (measured by PER) based on data that would be available at the time of the NBA Draft — specifically:

- **Physical and athletic combine data** (height, sprint times, vertical leap, etc.)
- **College attended**, which we map to conference strength

All of these features are available before a player’s professional career begins, making this a valid prediction task that **does not leak information from the future**.

---

### 🧪 Evaluation Metric

We evaluate our model using **Mean Absolute Error (MAE)**. This metric was chosen because:

- **PER is a continuous variable**, so MAE is a natural choice for regression tasks.
- MAE measures the average absolute magnitude of prediction errors, making it easy to interpret on the same scale as the target variable (PER).
- Unlike MSE, MAE has a equal weighting of absolute errors, making it more robust to outliers. This is valuable in our case, since PER can vary widely and a few extreme players (i.e., MVP-caliber superstar athletes, bench players who get garbage minutes) shouldn't disproportionately influence model performance.

Alternative metrics like MSE were considered, but MAE was ultimately chosen because it offers a clear, interpretable measure of average absolute prediction error without being overly sensitive to extreme values, aligning with our goal of producing stable and reliable PER predictions across the full range of players.

## Baseline Model

### 🔧 Baseline Model

For our baseline model, we used a **Linear Regression** algorithm with two quantitative features from the NBA Draft Combine:

- `HEIGHT_WO_SHOES`: Player’s height in inches (measured without shoes)  
- `LANE_AGILITY_TIME`: Time to complete the lane agility drill (in seconds)

These two features were chosen as simple physical indicators of player performance.

### 🧮 Feature Breakdown

- **Quantitative Features**: 2  
- **Ordinal Features**: 0  
- **Nominal Features**: 0  

Since both features are quantitative, no encoding was needed. A `StandardScaler` was used to normalize them before feeding into the regression model. This was implemented using an `sklearn` Pipeline.

### ⚙️ Model Pipeline

1. `StandardScaler` – to scale input features  
2. `LinearRegression` – basic regression model

### 📈 Model Performance

- **Mean Absolute Error (MAE)**: **~3.64**

This means the model’s predictions for PER are off by about 3.64 on average.

### ✅ Evaluation

While this model is straightforward, it only considers two features and assumes a linear relationship. It does not capture nonlinear interactions or leverage other potentially valuable information such as other drill results, physical measurements, or school attended.

For example, when predicting a terrible 5 feet player with a 20 second lane agility time, we get a PER of 13.84, which is actually above average PER.

Thus, while the baseline is useful as a starting point, it is not a good model on its own. Further feature engineering and advanced models are likely required for better predictive accuracy.

## Final Model

### 🚀 Final Model: Feature Justification and Performance

#### 🔍 Feature Engineering and Why It Matters

To improve upon our baseline model, we introduced several additional features and transformations:

- **Physical Measurements** (`HEIGHT_WO_SHOES`, `WEIGHT`, `WINGSPAN`, `STANDING_REACH`):  
  These features represent a player's physical profile, which is an important aspect of player potential performance. Taller, longer players tend to have advantages in rebounding, shot blocking, and overall presence on the court.

- **Athletic Drill Results** (`LANE_AGILITY_TIME`, `MAX_VERTICAL_LEAP`, `STANDING_VERTICAL_LEAP`, `THREE_QUARTER_SPRINT`):  
  These are direct measurements of a player’s explosiveness, speed, and vertical ability — traits that strongly correlate with athletic success in the NBA.

- **College Program (`SCHOOL_NAME`) mapped to NCAA Conference**:  
  We engineered a new categorical feature by mapping each player’s school to their NCAA conference. This helps capture the **strength of competition** faced in college. Players from power conferences may have better preparation for the NBA, resulting to better performance.

- **Standardization & Aggregation**:  
  Instead of using each physical or drill stat independently, we standardized each group and created **summary features** (mean z-score). This helps reduce multicollinearity and highlights the player’s relative strengths in those areas.

---

#### 🧠 Modeling Algorithm: Gradient Boosting

We used a **Gradient Boosting Regressor**, which builds an ensemble of shallow decision trees in a sequential fashion. Each new tree corrects the errors made by the previous trees, allowing the model to learn **non-linear relationships** in the data.

This algorithm is good for tabular datasets like ours, especially when features may interact in non-obvious ways (i.e., a tall player with a fast sprint time may outperform shorter but equally fast players).

---

#### 🎛️ Hyperparameter Tuning

To avoid overfitting or underfitting, we used `GridSearchCV` to tune the following key hyperparameters:

| Hyperparameter   | Why It Matters                                                                 |
|------------------|--------------------------------------------------------------------------------|
| `n_estimators`   | Number of trees (more trees = better fit, but can overfit)                     |
| `learning_rate`  | Controls how much each tree corrects previous errors (small = more conservative learning) |
| `max_depth`      | Maximum depth of each tree (controls tree complexity and overfitting risk)     |

We used `GridSearchCV` with 5-fold cross-validation on the training set to identify the best combination of these parameters.

**Best Parameters Found**:  
`n_estimators=100`, `learning_rate=0.1`, `max_depth=3`

---

#### 📊 Performance Comparison

| Model           | Features Used                         | MAE (Mean Absolute Error) |
|-----------------|----------------------------------------|----------------------------|
| Baseline Model  | HEIGHT_WO_SHOES, LANE_AGILITY_TIME    | **3.64**                   |
| Final Model     | + 7 physical/drill features, conference encoding | **3.61**          |

While the numerical improvement in MAE is modest, the final model **uses a more complete picture of a player's profile** and is likely to generalize better to unseen data due to its ensemble nature and better treatment of nonlinearities. However, the slight improvement suggests that Draft Combine statistics and school Conference soley do not determine a player's PER, and requires more complex features or models for a better prediction. 

---

#### 📌 Summary

- We enhanced the model with features that are **available at draft time**.
- Gradient Boosting was chosen to model complex interactions and nonlinear trends.
- GridSearchCV was used to tune key hyperparameters.
- The final model slighlty improves on baseline MAE, indicating that a player's Draft Combine peformance and school Conference aren't the primary deciders of how a player will perform in the NBA.
- One challenge that might have dampened the model's prediction accuracy could be how there were many NaN school names (74 rows), which could have added noise. 
