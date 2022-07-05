import pandas as pd

dataframe = pd.DataFrame({'Name':['Shobhit','vaibhav',
                                'vimal','Sourabh'],
                         
                        'Class':[11,12,10,9],
                        'Age':[18,20,21,17]})
 
# Checking created dataframe
display(dataframe)


new_data = dataframe.rename(columns = {'Name':'FirstName'})
 
# check new_data
display(new_data)


dataframe = pd.DataFrame({'Name':['Shobhit','vaibhav',
                                'vimal','Sourabh'],
                         
                        'Class':[11,12,10,9],
                        'Age':[18,20,21,17]})
 
# Checking created dataframe
display(dataframe)










