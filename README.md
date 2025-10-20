
# Insurance Fraud Detection System

## 1. Overview

This project is a complete, end-to-end machine learning system for detecting fraudulent insurance claims. It includes a powerful prediction model and a user-friendly web interface built with Streamlit.

The system takes a CSV file of insurance claims as input, analyzes each claim, and assigns it to one of three investigation tiers based on its predicted probability of being fraudulent.

## 2. Features

- **Advanced Fraud Prediction:** Utilizes a LightGBM gradient boosting model trained with sophisticated feature engineering to achieve high accuracy.
- **Tiered Investigation System:** Classifies claims into "Urgent," "Review," or "Auto-Approve" tiers to optimize operational workflow.
- **Web-Based Interface:** An easy-to-use Streamlit application allows for simple file uploads and clear visualization of results.
- **Downloadable Results:** The scored output, including fraud probabilities and tiers, can be downloaded as a CSV file for further analysis.

## 3. Setup and Installation

To set up the environment for this project, you first need to install the required Python libraries.

1.  Navigate to the project directory in your terminal.
2.  Run the following command to install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 4. How to Run the Application

Once the setup is complete, you can launch the web application.

1.  Make sure you are in the project's root directory in your terminal.
2.  Run the following command:
    ```bash
    streamlit run app.py
    ```
3.  The application should automatically open in a new tab in your default web browser.

## 5. How to Use the Application

1.  Once the application is running, you will see a title and an "Upload a CSV file" section.
2.  Click the "Browse files" button and select the CSV file containing the claims you want to analyze. You can use the included `test_claims.csv` or `challenge_dataset.csv` to test the system.
3.  The application will automatically process the file and display the results.

## 6. Input CSV File Format

For the system to work correctly, the uploaded CSV file **must** contain the following columns. The order of columns does not matter, but the names must be exactly as shown below.

```
- Month
- WeekOfMonth
- DayOfWeek
- Make
- AccidentArea
- DayOfWeekClaimed
- MonthClaimed
- WeekOfMonthClaimed
- Sex
- MaritalStatus
- Age
- Fault
- PolicyType
- VehicleCategory
- VehiclePrice
- FraudFound_P  (Note: This column is required in the file, but is not used for prediction. It is kept for data consistency.)
- PolicyNumber
- RepNumber
- Deductible
- DriverRating
- Days_Policy_Accident
- Days_Policy_Claim
- PastNumberOfClaims
- AgeOfVehicle
- AgeOfPolicyHolder
- PoliceReportFiled
- WitnessPresent
- AgentType
- NumberOfSuppliments
- AddressChange_Claim
- NumberOfCars
- Year
- BasePolicy
```

## 7. Understanding the Output

The application will display a results table with two new columns:

-   **Fraud Probability:** A score from 0.0 to 1.0 indicating the model's predicted likelihood that the claim is fraudulent.
-   **Tier:** The assigned investigation tier based on the probability score.

The tiers are defined as follows:

-   **Tier 1: URGENT (Probability > 0.70)**
    -   **Meaning:** These claims are highly suspicious.
    -   **Action:** Assign to expert fraud investigators for immediate, in-depth review.

-   **Tier 2: REVIEW (Probability between 0.20 and 0.70)**
    -   **Meaning:** These claims are moderately suspicious and warrant a second look.
    -   **Action:** Assign to a standard review queue. This tier is designed to catch less obvious fraud cases.

-   **Tier 3: AUTO-APPROVE (Probability < 0.20)**
    -   **Meaning:** These claims are considered safe with high confidence.
    *   **Action:** These claims can be automatically approved, freeing up operational resources.
