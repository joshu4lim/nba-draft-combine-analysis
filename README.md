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
## Baseline Model
## Final Model