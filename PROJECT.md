# Advisor Simulation Project

## Goal

Build a synthetic institutional-quality dataset for financial advisor sales and assets behavior.

Dataset grain:

advisor_id × month × mstar_category

Later extensions:

* vehicle
* asset_type
* wholesaler touches

## Dataset Objectives

The dataset should:

* feel behaviorally realistic
* exhibit temporal persistence
* react to market regimes
* contain realistic advisor archetypes
* contain skewed AUM distributions
* support downstream AI experimentation

## Scale

* 1000 advisors
* 24 monthly periods
* 20–25 Morningstar categories

## Core Outputs

Columns:

* advisor_id
* month
* mstar_category
* gross_sales
* redemptions
* net_sales
* assets

## Behavioral Drivers

Flows should depend on:

* advisor archetype
* category preference
* market regime
* prior allocations
* randomness/noise

Assets should evolve using:
A_t = A_(t-1)*(1+r_t) + NetFlow_t

## Initial Scope

Version 1 should only include:

* advisors
* categories
* market regimes
* monthly flows
* monthly assets

No wholesalers or CRM yet.

## Technical Stack

* Python
* pandas
* numpy
* yaml
* parquet

## Required Modules

1. generate_advisors.py
2. generate_market.py
3. generate_allocations.py
4. simulate_monthly_flows.py
5. validation.ipynb
