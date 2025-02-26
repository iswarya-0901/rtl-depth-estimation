# rtl-depth-estimation
# Predicting Combinational Logic Depth Using Machine Learning

## Overview
Combinational logic depth is a crucial factor in digital circuit design, influencing the timing performance of integrated circuits. Traditional methods rely on synthesis tools, which can be time-consuming. This project aims to develop a machine learning model that predicts the combinational logic depth of RTL (Register Transfer Level) circuits without requiring full synthesis. By using a dataset of gate-level parameters, we train an ML model to estimate the logic depth efficiently.

## Problem Statement
Timing analysis in digital circuits is often performed after synthesis, leading to significant delays if violations are found. Our goal is to predict the combinational logic depth of signals early in the design phase using machine learning techniques. This will help designers identify timing issues before synthesis, speeding up the development process.

## Approach
1. **Dataset Preparation**
   - Create a dataset with sample RTL circuits.
   - Manually assign logic depth values based on gate count and fan-in.
   
2. **Feature Engineering**
   - Extract key parameters such as:
     - Number of logic gates
     - Fan-in (number of inputs per gate)
   
3. **Machine Learning Model**
   - Train a **Linear Regression** model to predict logic depth.
   - Use **Scikit-learn** for model training and evaluation.
   
4. **Model Evaluation**
   - Split the dataset into training and testing sets.
   - Evaluate accuracy using metrics like Mean Squared Error (MSE).

## Environment Setup
To set up the environment for this project, follow these steps:
1. **Install Python (3.8 or higher)**
   - Download from [Python Official Site](https://www.python.org/downloads/)
   - Verify installation using:
     ```sh
     python --version
     ```

2. **Install Required Dependencies**
   - Run the following command to install necessary Python libraries:
     ```sh
     pip install pandas scikit-learn numpy matplotlib
     ```

3. **Run the Machine Learning Model**
   ```sh
   python train_model.py
   ```
   This will train the model and predict the combinational logic depth for test data.

## Complexity Analysis
The complexity of the model is analyzed as follows:
- **Time Complexity:**
  - Data Preprocessing: **O(n)**
  - Model Training: **O(n^2)** for Linear Regression (depends on number of features and data points)
  - Prediction: **O(1)** for a single query
- **Space Complexity:**
  - **O(n)** for storing dataset
  - **O(m*n)** for storing model parameters (m = features, n = training examples)

## Implementation Details
### **Verilog Code Samples**
- **Complex ALU (`alu_complex.v`)**
  ```verilog
  module alu_complex(input [3:0] A, input [3:0] B, input [2:0] Op, output reg [3:0] Y);
      always @(*) begin
          case (Op)
              3'b000: Y = A & B;
              3'b001: Y = A | B;
              3'b010: Y = A + B;
              3'b011: Y = A - B;
              3'b100: Y = A * B;
              3'b101: Y = A / (B == 0 ? 1 : B);
              default: Y = 4'b0000;
          endcase
      end
  endmodule
  ```

### **Dataset (`logic_depth.csv`)**
| Num_Gates | Fan_In | Combinational_Depth |
|-----------|--------|---------------------|
| 2         | 2      | 1                   |
| 4         | 2      | 2                   |
| 6         | 3      | 3                   |
| 8         | 3      | 4                   |
| 10        | 4      | 5                   |
| 12        | 4      | 6                   |

## Files Included
- `and_gate.v` → Basic AND gate Verilog file
- `xor_gate.v` → XOR gate Verilog file
- `alu_complex.v` → Complex ALU module
- `logic_depth.csv` → Training dataset
- `train_model.py` → Python script for ML training
- `README.md` → Project documentation
- `report.md` → Final project report

## Results
The trained model provides an estimation of combinational logic depth based on the number of gates and fan-in. This approach allows for a faster prediction of timing constraints compared to full synthesis tools.

## Future Improvements
- Collect a larger dataset from real-world circuits.
- Experiment with more advanced ML models such as Decision Trees or Neural Networks.
- Automate feature extraction using parsing tools like PyVerilog.


