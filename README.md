'''

    Pre-process the data, including cleaning and encoding, and determine if all the features are necessary for predicting the compressive strength. Address missing attribute values by considering whether to replace them with an average value or remove the feature altogether.
    Build a regressor using sci-kit learn functionalities, including the rational choice of a regression algorithm, data splitting, training, testing, and analysis of the hyper-parameter choices. Justify the choice of algorithm and parameters with an analysis of their effect.
    Identify the single feature that is most important to obtain a good prediction and create an interactive graph to summarize the relative importance of different variables.

To produce:

    Algorithm, including pre-processing, training, testing, and analysis of the predictivity of the algorithm.
    An interactive graph that summarizes the relative importance of different variables.
    A report detailing the choices made and the reasons behind them, when necessary.

The dataset contains 8 quantitative input variables and 1 quantitative output variable, which is the concrete compressive strength. The input variables include Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Age, while the output variable is the concrete compressive strength measured in MPa[1].

Pseudocode/Planning:

    Load the dataset from the Concrete_database.csv file.
    Perform data pre-processing, including cleaning and encoding, and handle missing attribute values.
    Split the data into training and testing sets.
    Choose a regression algorithm and build the regressor using sci-kit learn functionalities.
    Train the model on the training data and evaluate its performance on the testing data.
    Identify the most important feature for predicting the compressive strength.
    Create an interactive graph to visualize the relative importance of different variables.
    Provide comments and analysis of the choices made and their impact on the algorithm's predictivity.

# Step 1: Data Pre-processing
# Step 2: Data Splitting
# Step 3: Model Building
# Step 4: Identify the most important feature
# Step 5: Create an interactive graph
# Step 6: Report and Documentation
