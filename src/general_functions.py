# Created by JJW Apr 22 2016
# This contains some general functions (like I/O functions)
# that are used by many of the codes
# for COS 424 final project


ignore_after_value = 98  # Ignore columns past this value
columns_to_remove = [53, 47, 46, 33, 1, 0]  # Ignore these columns (mostly just the name of the object listed several times)

line_to_ignore_unselected = 19750  # This particular line of data is bad


def read_in_data(path="../data/"):
    """Reads in the data and returns a feature set with classification

    Returns: tuple, first element is the feature set (2-d list) and
             second element is a list of the categories of the data

    Keyword Arguments:
        path {str} -- path to the two data files (default: {"../data/"})
    """

    # First read in the data for the candidates that were selected (category 1)
    with open(path + "selectedcandidates.txt", "r") as f:
        lines = f.readlines()
        data_selected = []  # Empty list to collect the data

        for line in lines:
            temp = line.split()
            temp = temp[:ignore_after_value]  # Cut off unused values at end
            for i in columns_to_remove:
                _ = temp.pop(i)  # Remove the columns we need to ignore
            data_selected.append(temp)

    # Convert all the data values from string to float
    for i in range(len(data_selected)):
        for j in range(len(data_selected[i])):
            data_selected[i][j] = float(data_selected[i][j])

    # Now read in the data for the unselected candidates (category 0)
    with open(path + "unselectedcandidates.txt", "r") as f:
        lines = f.readlines()
        data_unselected = []  # Empty list to collect the data

        for j in range(len(lines)):
            if j == line_to_ignore_unselected:  # Ignore the bad line
                continue
            temp = lines[j].split()
            temp = temp[:ignore_after_value]  # Cut off unused values at the end
            for i in columns_to_remove:
                _ = temp.pop(i)  # Remove the columns we need to ignore
            data_unselected.append(temp)

    # Convert all data values to float
    for i in range(len(data_unselected)):
        for j in range(len(data_unselected[i])):
            data_unselected[i][j] = float(data_unselected[i][j])

    # Create a list that quantifies the classification
    was_selected = [1] * len(data_selected)
    was_selected += [0] * len(data_unselected)

    # Return the data and the classification
    return (data_selected + data_unselected, was_selected)

