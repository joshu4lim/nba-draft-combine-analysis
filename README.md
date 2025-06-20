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

- **Narrowing Down Rows:** Since there are many data about NBA players, I first decide to narrow the dataset to currently active players (as of June 2025). Next, I narrow it down further to players who particpated in the Draft Combine, as some players opt out of it. In this step, I fetch both physical measurements and drill results of each player that participated in the Combine. Finally, I fetch name of college attended and career statistics of the pool of players I have. This step doesn't narrow down the rows since it all players have a career statistic, and players who didn't went to college can just be imputed with a NaN. This step narrowed down the dataset to 484 rows.

- **Narrowing Down Columns:** 
These are the columns I initially chose and their justifications.

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
| `The following features below are used to create PER` |                                                |
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
$$
\text{EFF} = \frac{\text{PTS} + \text{REB} + \text{AST} + \text{STL} + \text{BLK} - (\text{Missed FG}) - (\text{Missed FT}) - \text{TO}}{\text{GP}}
$$
Where Missed FG = FGA - FGM and Missed FT = FTA - FTM
(https://en.wikipedia.org/wiki/Efficiency_(basketball)#EFF)

The final dataset is 286 x 24

#### 📌 Cleaned Data Preview

Here’s a preview of the cleaned dataset:

|   PLAYER_ID | SCHOOL_NAME   |   STANDING_VERTICAL_LEAP |   MAX_VERTICAL_LEAP |   LANE_AGILITY_TIME |   THREE_QUARTER_SPRINT | full_name      | POSITION   |   HEIGHT_WO_SHOES |   WEIGHT |   WINGSPAN |   STANDING_REACH |   PTS |   REB |   AST |   STL |   BLK |   FGA |   FGM |   FTA |   FTM |   TOV |   GP |      PER |
|------------:|:--------------|-------------------------:|--------------------:|--------------------:|-----------------------:|:---------------|:-----------|------------------:|---------:|-----------:|-----------------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|---------:|
|      203500 | Pittsburgh    |                     28.5 |                33   |               11.85 |                   3.4  | Steven Adams   | C          |             82.75 |    254.5 |      88.5  |            109.5 |  6743 |  6115 |  1145 |   646 |   703 |  4786 |  2804 |  2129 |  1134 |  1083 |  764 | 14.7801  |
|     1628389 | Kentucky      |                     33.5 |                38.5 |               11.94 |                   3.24 | Bam Adebayo    | PF-C       |             80.75 |    242.6 |      86.75 |            108   |  8923 |  5024 |  2044 |   607 |   489 |  6389 |  3428 |  2598 |  1965 |  1218 |  567 | 21.649   |
|     1630534 | Kansas        |                     32   |                41.5 |               10.88 |                   3.13 | Ochai Agbaji   | SG         |             76.5  |    214.4 |      82    |            103.5 |  2044 |   795 |   331 |   168 |   133 |  1786 |   787 |   229 |   164 |   223 |  279 |  7.82796 |
|     1641725 | Creighton     |                     29.5 |                34   |               11.6  |                   3.35 | Trey Alexander | SG         |             75.25 |    184.6 |      82    |            101.5 |    32 |    12 |    11 |     2 |     1 |    41 |    13 |     4 |     3 |     5 |   24 |  1       |
|     1628960 | Duke          |                     32.5 |                40.5 |               10.31 |                   3.15 | Grayson Allen  | SG         |             75    |    198   |      79.25 |             97   |  4250 |  1216 |   812 |   293 |   112 |  3140 |  1416 |   657 |   563 |   399 |  403 | 11.0819  |

---

### 📊 Univariate Analysis

We first explored the **distribution of Player Height without Shoes** to understand the outcome variable.

<iframe src="plots/uni-height-plot.html" width="800" height="500" frameborder="0"></iframe>

**Interpretation:**  
There is almost a bell-shaped distribution with a slight right-skew for height without shoes, with most players falling between 76-80 inches, with a few outliers at each end. It suggest that height is relatively standardized across players, which may be a useful feature for making our predictive model.

---

### 📈 Bivariate Analysis

Next, we examined how **WINGSPAN** relates to **PER**.

<iframe src="plots/biv-height-by-per-plot.html" width="800" height="500" frameborder="0"></iframe>

**Interpretation:**  
There appears to be a **mild positive relationship** between wingspan and PER. Players with longer wingspans tend to achieve slightly higher PERs, which makes sense since reach can be advantageous in defense, rebounding, and shot-blocking.

---

### 📊 Interesting Aggregates

We grouped players by **college conference** and calculated their **average PER**.

| Conference     | Avg PER |
|----------------|---------|
| SEC            | 15.2    |
| Big Ten        | 16.1    |
| ACC            | 16.7    |
| Big 12         | 14.8    |
| Pac-12         | 13.4    |
| West Coast     | 16.9    |

**Interpretation:**  
Players from conferences like the **ACC** and **West Coast** show slightly higher average PERs, suggesting that some conferences may better prepare players for professional success. This motivated our decision to engineer a new categorical feature for **conference strength** in our pipeline.

---

## Framing a Prediction Problem
## Baseline Model
## Final Model