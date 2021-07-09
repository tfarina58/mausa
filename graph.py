import matplotlib.pyplot as plt
  
# x axis values
x = ['2%', '5%', '10%', '25%', '50%']
# corresponding y axis values
y = [0.2,0.4305,0.4436,0.4774,0.5194]
  
# plotting the points 
plt.plot(x, y)
  
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
  
# giving a title to my graph
plt.title('My first graph!')
  
# function to show the plot
plt.show()