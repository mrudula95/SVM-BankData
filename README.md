#Support_Vector_Machine

This program is implementation of SVM in Python. Analysis has been performed with linear and RBF kernel using Bank Dataset. The code in divided into different sections for better understanding of the implementation. We have used the following conventions:
X -> Train data
Y -> Train labels
x_1 -> Test data
y_1 -> test labels

To run the code, you first need to install pip by running sudo apt-get install pip and then install the following packages as pip install <package-name>
1. pandas
2. numpy
3. pylab
4. sklearn
5. ggplot
6. csv
7. scipy

The original dataset consists of two csv files:
1. Bank.csv(train data)
2. Bank-full.csv(Test data)

The file that needs to be run is bank.py

i)  In Spliting the data section, we split and store the data and labels into seperate csv files for our convenience.

ii) In preprocessing the data section, necessary modifications are made to the data.

iii)In running the SVM classifier section, we first perform feature selection. We use c= 1 and gamma = auto. After cross-validation we can     alter the c and gamma values for accuracy. The precision and recall block is needed only when running for test data.

iv) In cross-validation with grid search section, we have intialised values for C and gamma. This might take some time to execute. This section of the code needs to be commented if you wish to run without cross-validation.
