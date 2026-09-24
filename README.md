# Loan Credit Risk Prediction Engine

## Project Overview
This project is built around the Kaggle Home Credit Default Risk Competition. 

Many people struggle to get loans due to insufficient or non-existent credit histories. Home Credit strives to broaden financial inclusion for the unbanked population by providing a safe and positive borrowing experience. The objective of this project is to build a machine learning classification model to predict an applicant's repayment capability, ensuring that clients capable of repayment are successfully identified and granted loans.

---

## Dataset Architecture
The dataset is split across multiple relational tables linked by unique client/loan identification keys:

* **application_train.csv / application_test.csv**: The main static tables containing applicant details (demographics, income, asset ownership, etc.). The training set includes the target label (`TARGET`: `0` for repayment, `1` for default).
* **bureau.csv**: Client's previous credits from other financial institutions reported to the credit bureau.
* **bureau_balance.csv**: Monthly balance history of previous credits reported to the credit bureau.
* **previous_application.csv**: Past application history for Home Credit loans by clients in the sample.
* **POS_CASH_balance.csv**: Monthly balance snapshots of previous point-of-sale or cash loans with Home Credit.
* **credit_card_balance.csv**: Monthly snapshots of previous credit cards held with Home Credit.
* **installments_payments.csv**: Repayment history for previously disbursed loans with Home Credit.
* **HomeCredit_columns_description.csv**: Metadata dictionary explaining all features across tables.

*Note: Due to its large size (over 2 GB total), the raw dataset is excluded from version control via `.gitignore`.*

---

## How to Download the Dataset
1. Visit the [Kaggle Home Credit Default Risk Competition Page](https://www.kaggle.com/competitions/home-credit-default-risk/data).
2. Download the compressed dataset archive.
3. Create a folder named `data/` in the root directory of this project.
4. Extract all downloaded CSV files directly into the `data/` folder.

---

## Project Setup Instructions

### 1. Clone the Repository
```bash
git clone [https://github.com/jobann/Loan-Credit-Risk-Engine.git](https://github.com/jobann/Loan-Credit-Risk-Engine.git)
cd Loan-Credit-Risk-Engine
```
### 2. Activate Your Virtual Environment
Before installing dependencies, ensure your virtual environment is active.

**For macOS/Linux:**

```bash
source .venv/bin/activate
```
**For Windows:**

```bash
.venv\Scripts\activate
```
### 3. Install Requirements
This project relies on a standard data science stack (Pandas, Scikit-Learn, LightGBM, etc.). Run the following command to install all required libraries:

```bash
pip install -r requirements.txt
```