# Predicting the Direction of the S&P 500 Index for Option Premium Trades

This project aims to create a machine learning model to predict the direction of the S&P 500 Index (SPX) after economic indicator announcements. The goal is to help option premium sellers identify opportunities to open short positions in the opposite direction of the predicted SPX movement, which can be closed after the Implied Volatility (IV) drops to lock in profits.

## Problem Definition

The SPX often experiences significant spikes in IV before economic indicator announcements, leading to overpricing in the options chain. After the indicator value is released, the IV usually drops sharply, causing the extrinsic value of options to decrease. By correctly predicting the direction of the SPX, it is possible to capitalize on these market movements.

## Research Questions

- How can feature engineering improve model prediction accuracy?
- How do findings from the financial domain inform the design of the prediction model?
- What is the best algorithm for predicting short-term SPX price trends?
- What features and economic indicators influence the direction of the SPX?

## Deliverables

The final deliverables of this project will include:

- A binary classifier for the SPX direction at the minute level using live stock market data and economic indicator data from the Trading Economics API.
- An MLOps system to continuously train, maintain, deploy, and monitor the machine learning model on AWS.
- A real-time trading system to trade the signals generated using the TradeStation API.

## Technologies

The project team will work with the following technologies:

- AWS
- MLflow
- DVCFileSystem
- Docker
- Trading Economics API
- TradeStation API

## Expected Learning Outcomes

Through this project, the team member expects to:

- Learn how to handle streaming data and make real-time predictions while working with APIs and MLOps tools.
- Gain experience with deploying fault-tolerant machine learning systems on the cloud, specifically exploring the available machine learning capabilities of AWS.

## Team

The team consists of one member:

Noel Namai

## System Design

### Data

The following data sources will be used in this project:

- SPX open, close, high, low, and volume.
- SPX Simple Moving Average (SMA) (9, 21, 50, 200), On-balance Volume (OBV), Moving Average Convergence Divergence (MACD), and Relative Strength Index (RSI).
- SPX option IV rank and open interest.