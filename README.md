# Neural Network

Author: David Wolf\
last changed: 14.03.2024

## Description
This is a simple neural network that recognizes handwritten digits of the mnist dataset. The mnist dataset provides test and training csv files of digits, wich each consist of 28x28 pixels. The pixels values range between 0 and 255 representing a different shade of white.<br>
The neural network consist of three layers. An input layer, an hidden layer and an output layer. The input layer takes all 784 color values as an input. The output layer calculates a probability for each number 0 to 9.<br>

## The Math

### Forward Propagation
The input Vector $\vec{i}$ represents all 784 pixel values between 0 and 255.

$\vec{i} = \begin{pmatrix}{}i_{1}\\.\\.\\.\\i_{784}\\\end{pmatrix}$

Of course not only one digit gets fed into the network. That's why there are $m$ input Vectors which thogether form the input matrix $I$.<br>

$I = \begin{pmatrix}{}i_{1_1}&.&.&i_{1_m}\\.&.&&.\\.&&.&.\\.&&&.\\i_{784_1}&.&.&i_{784_m}\\\end{pmatrix}\hspace{1cm}\dim(I) = 784 \times m$

$I$ gets passed into the first layer and multiplied by the first matrix $W_{1}$ of weights. $W_{1}$ represents a matrix of weights initialized with float numbers between $-0.5$ and $0.5$. Additionally a bias $\vec{b_{1}}$ is added.

$uL_{1} = W_{1} * I + \vec{b_{1}}$

$\dim(uL_{1})= 10 \times m \hspace{0.2cm}\dim(W_{1}) = 10 \times 784 \hspace{0.2cm}\dim(b_{1}) = 10 \times 1$

The unactivated Matrix $uL_{1}$ now needs to get passed through an activation function. The activation function used is called Rectefied Linear Uni (ReLu). ReLu simply converts all the nagative numbers into 0.

$reLu(x) = max\{x,0\}$

$aL_{1} = reLu(uL_{1})$

The input as now been converted into a matrix of $10 \times m $ values represented by the activated Matrix $aL_{1}$. 

To calculate the values of the output layer another weights matrix $W_{2}$ needs to be mulitplied with $aL_{1}$. Additionally a bias vector $\vec{b_{2}}$ gets added.

$uL_{2} = W_{2} * aL_{1} + \vec{b_{2}}$
$\dim(uL_{2})= 10 \times m \hspace{0.2cm} 
\dim(W_{1}) = 10 \times 10 \hspace{0.2cm} 
\dim(b_{1}) = 10 \times 1$

The unactivated Matrix $uL_{2}$ now has to be passed through another activation function. To calucalte probabilities which represent each number 0 to 9 an activation function called softmax is being used. It takes the exponential values of each element and divides it by the sum those exponantial values for each input vector.

$softmax(\vec{x}) = \frac{e^{x_{i}}}{ \sum e^{x_{i}}}$

$aL_{2} = softmax(uL_{2})$

Now $aL_{2}$ represents a matrix with $m$, $10 \times 1$ output vectors each containing a probability for the numbers 0 to 9.

### Backwards Propagation
Backwards Propagation is used to evaluate the impact each weight and bias has on the result of the forward propagation.

To be able to evalute how good the predication was in comparison to the optimal solution we need to calculate a cost for each dataset. The cost matrix $C_{2}$ which contains all the costs of the datasets is calculated by subtracting a matrix with optimal solutions $O$ from the output matrix $aL_{2}$.

$C_{2} = aL_{2} - O$

The cost matrix $C_{2}$ can now be used to evaluate how each weight and bias contributed to the result by applying the chain rule. As we use $m$ datasets we want to estimate an average via dividing by $m$.

$\frac{1}{m}\frac{\partial C_{2}}{\partial W_{2}} = \frac{1}{m} C_{2} \times aL_{1}^T $

$\frac{1}{m}\frac{\partial C_{2}}{\partial b_{2}} = \frac{1}{m}\sum^mC_{2}$

To evaluate the impact of $W_{1}$ and $b_{1}$ we can calculate a seperate cost function for the hidden layer which is independet of the weights and biases of the output layer.

$C_{1} = W_{2} \times C_{2} \times \frac{\partial ReLu(uL_{1})}{\partial uL_{1}}$

Now the same priciple as before can be applied to the second cost matrix to calclate the impcat of $W_{1}$ and $b_{1}$.

$\frac{1}{m}\frac{\partial C_{1}}{\partial W_{1}} = \frac{1}{m} C_{1} \times I^T$

$\frac{1}{m}\frac{\partial C_{1}}{\partial b_{1}} = \frac{1}{m}\sum^mC_{1}$

### Gradient Descend

For our neural network to be able to improve and "learn" we can let it try and minimize our cost matrix. An alogrithm which is able to find a local minimum is called gradient descend. It works by following the steepest descend at a given point inside our cost matrix. The size of the steps it takes is defined by a leraning rate $\alpha$.

The application of this alogorith is quite simple, we just need to update our weights and biases according to their impact each iteration we run the model, muliplied by the learing rate $\alpha$.

$W_{1} = W_{1} - \alpha * \frac{1}{m}\frac{\partial C_{2}}{\partial W_{1}}$

$b_{1} = b_{1} - \alpha * \frac{1}{m}\frac{\partial C_{2}}{\partial b_{1}}$

$W_{2} = W_{2} - \alpha * \frac{2}{m}\frac{\partial C_{1}}{\partial W_{2}}$

$b_{2} = b_{2} - \alpha * \frac{1}{m}\frac{\partial C_{1}}{\partial b_{2}}$