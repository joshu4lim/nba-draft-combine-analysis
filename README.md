# Predicting NBA Player's Performance Based on Draft Combine Data

## Introduction

This project explores player performance in the NBA using pre-draft scouting and athleticism data. The dataset includes physical measurements, athletic drill results, and college information for players entering the NBA Draft.

### 📌 Project Question

**Can we predict a player's professional performance (as measured by PER – Player Efficiency Rating) using pre-draft physical, athletic, and educational background data?**

This question is valuable because PER is a well-known, single-number summary of a player's in-game performance. If teams could reliably predict PER using only combine and college data, they could make more data-driven draft decisions — potentially reducing costly draft busts.

### 📂 Dataset Overview

- **Source**: NBA Draft Combine Data (historical records combined with player outcomes)
- **Number of Rows**: `###` *(replace with actual number of rows from `final_df.shape[0]`)*  
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

This dataset is especially compelling for basketball analysts, front offices, and fans who want to better understand which early indicators best translate to success at the professional level.


## Data Cleaning and Exploratory Data Analysis
## Framing a Prediction Problem
## Baseline Model
## Final Model