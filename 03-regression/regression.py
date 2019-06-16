import csv

# Your code for regression and R2 here

if __name__ == "__main__":
    # Here, we load the boston dataset
    boston = csv.reader(open('boston.csv'))  # The boston housing dataset in csv format
    # First line contains the header, short info for each variable
    header = boston.__next__()  # In python2, you might need boston.next() instead
    # Data will hold the 13 data variables, target is what we are trying to predict
    data, target = [], []
    for row in boston:
        # All but the last are the data points
        data.append([float(r) for r in row[:-1]])
        # The last is the median house value we are trying to predict
        target.append(float(row[-1])) 
    # Now, use the dataset with your regression functions to answer the exercise questions
    print("Names of the columns")
    print(header)
    print("First row of data ->variable to predict")
    print(data[0], " -> ", target[0])

    # Plot, regression here

    # Example of writing out the R2.txt file, with 0.0 guess for coefficient of correlation
    fout = open('results.txt', 'w')
    for i in range(13):
        column = [row[i] for row in data]  # get the column
        target  # target is always the median house value
        alpha, beta = 0.0, 0.0
        R2 = 0.0  # Fill with the real value from your code
        fout.write('%f,%f,%f\n' % (alpha, beta, R2))  # One line per variable
    fout.close()
