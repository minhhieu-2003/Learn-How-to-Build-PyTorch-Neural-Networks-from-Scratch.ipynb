#importing required libraries
from torch import nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch import optim
import torch.utils.data as Data
from torch import Tensor
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    #loading data
    df = pd.read_csv("D:/codePython/project _python/src/data.csv")
   # print(df.head())  # Print the first few rows of the DataFrame

    data = df.drop(["customer_id", "phone_no", "year"], axis=1)

   # print(data.tail())  # Print the last few rows of the DataFrame
    #print(data.describe())  # Print the summary statistics of the DataFrame

    #checking for null values
   # print(data.isna().sum()) # Print the number of missing values in each column

    #dropping null values
    data = data.dropna(axis=0)
   # print(data.shape) # Print the shape of the DataFrame after dropping missing values

    #unique values in gender column
   # print(data["multi_screen"].unique())
    #print(data["mail_subscribed"].unique())

    #label encoding categorical features
    le = LabelEncoder()
    data["gender"] = le.fit_transform(data["gender"])
    data["multi_screen"] = le.fit_transform(data["multi_screen"])
    data["mail_subscribed"] = le.fit_transform(data["mail_subscribed"])
    print(data.head()) # Print the first few rows of the DataFrame after label encoding

    #distribution of the target column
    print(data.groupby("churn").size()) # Print the number of observations in each class

    #dropping categorical columns and keeping numerical columns only
    data_num = data.drop(["gender", "multi_screen", "mail_subscribed"], axis=1)
    cols = data_num.columns
    data_scaled = MinMaxScaler().fit_transform(data_num)

    #defining dependent and independent variables
    X = data.drop("churn", axis=1)
    Y = data["churn"].astype(int)

    # split a dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X.values, Y, test_size=0.2, random_state=42)

    #Hyperparameters for our network
    input_size = X.shape[1]
    hidden_sizes = [128, 64]
    output_size = 2

    # Build a feed-forward network
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.Softmax(dim=1)
    )
    print(model)

    # Define the loss
    criterion = nn.CrossEntropyLoss()

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #converting data into tensor
    X_train = Tensor(X_train)
    y_train = Tensor(np.array(y_train))
    BATCH_SIZE = 64

    torch_dataset = Data.TensorDataset(X_train, y_train)

    #loading data for the model
    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=0, pin_memory=True
    )

    epochs = 100 #100
    for e in range(epochs):
        running_loss = 255
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y.type(torch.LongTensor))
            
            # Training pass
            optimizer.zero_grad()
            
            output = model(b_x)
            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {e+1}/{epochs} - Training loss: {running_loss/len(loader)}")
    
    # testing with test data
    X_test = Tensor(X_test)
    y_test = Tensor(np.array(y_test))

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for testing
    with torch.no_grad():
        # Get predictions
        test_output = model(X_test)
        
        # Calculate the loss
        test_loss = criterion(test_output, y_test.type(torch.LongTensor))
        
        # Calculate accuracy
        _, predicted = torch.max(test_output, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / y_test.size(0)

    print(f"Test Loss: {test_loss.item()}")
    print(f"Test Accuracy: {accuracy * 100}%")