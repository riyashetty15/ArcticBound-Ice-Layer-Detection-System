# Ice_Tracing
In the part2 of the project I have implemented the AirIce boundary using the Simple Bayes net and Viterbi. 

### Simple Approach:
In the simple I have implemented taking the Firstmax and Secondmax values and then taking the absolute of the difference between to be greater than 10 pixels. Then taking the minimum pixel as the first boundary i.e the Airice boundary and the maximum pixels the Icerock boundary.

### Viterbi Approach:
1. The transisition probablities which I have taken as a predefined list. The greater the difference in the number of rows from the current row the smaller the probabilities.
2. The Emission probablities I have made use of the formula which I have defined as a utility function, which provides us the with emission probabilities for the edge strengths.
3. The states are nothing but the values from the emission probablities in this case.
Then coming to the actual calculation of the Viterbi algorithm.

Here, I have taken the absolute difference between the transition probabilities and the row values in the edge strengths and I have checked if the difference is within the range for the values mentioned in the transition probablities. 
I have assumed 5 values in the list of transition probablities. These 5 values are the position of the rows i.e 5th value in the transition probablities indicates that the row value is 5 rows apart.

I have compared this value with the maximum which initially I have assumed to be zero which I are then assigning with the values calculated for the val. 
Once I have the maximum I are pushing it into the states by multiplying it with the emission values and then  taking the final maximum of the final value. 
I have used backtracking by starting from the last column value for each columnand then decrementing the value at each column.

Similarly, I have used the same approach for the IceRock boundary in which I have just added 10 pixels across the rows indexes from the Airice boundaries and have started checking or the state values from the +10 pixels till the end of the column. 
I have added this value to the final maximum value of the Viterbi through which I get the Icerock boundary.

For the human input values the similar approach is used but with a twist. For the given coordinates I make the whole row and column of the given coordinates as zero but just keeping the provided coordinates as 1. 
Except this the whole implementation is same as that of the above Viterbi functions which I implemented.

To support the code I have added the images of the output for each test file provided:
I have added the supporting output images inside the images in the part2 folder of the code.
