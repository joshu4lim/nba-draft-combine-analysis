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
| `PER`                    | **Target** variable: Player Efficiency Rating in the NBA (higher is better).|

---


## Data Cleaning and Exploratory Data Analysis
## Framing a Prediction Problem
## Baseline Model
## Final Model