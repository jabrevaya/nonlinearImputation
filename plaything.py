import numpy


a1 = numpy.array([[ 1.01616032 , 0.4978586 , -2.02994064],
 [ 1.02146486 , 0.50641231, -2.04129067],
 [ 1.02286377 , 0.51438091, -2.06230951]]
)
b1 = numpy.array([[0.11118329, 0.1007696 , 0.13243589],
 [0.16861349, 0.13922124, 0.22269824],
 [0.17412253, 0.11149018, 0.17946228]])

a2 = numpy.array([[ 0.99580941 , 0.50647117, -2.00719317],
 [ 0.98913863 , 0.51653633, -2.0094086 ],
 [ 0.99055675,  0.50972586, -2.01160979]])

b2= numpy.array([[0.05170575, 0.04769405, 0.06916971],
 [0.0766289,  0.06802272, 0.09872459],
 [0.07625339, 0.05352257, 0.07834162]])

a3 = numpy.array([[ 0.99577122 , 0.49893608, -1.99781411],
 [ 0.99849489 , 0.49658705, -1.99816648],
 [ 0.99847379 , 0.50080108, -2.00268935]])

b3 = numpy.array([[0.02644199, 0.02567425, 0.03915527],
 [0.03788588, 0.03568636, 0.0545546 ],
 [0.03773806, 0.02823357, 0.04709023]])

true = [1,0.5,-2, 1.2, -0.7, 0.3]

def evaluatorMSE(a, b, true):
    MSE_vector = (a - true)**2 + b**2
    MSE_average = numpy.mean(MSE_vector, axis=1)
    MSE_averageNormalized = MSE_average / MSE_average[0]
    MSE_normalized = MSE_vector / MSE_vector[0]
    print('Average MSE: \n', MSE_average)
    print('\nNormalized Average MSE: \n', MSE_averageNormalized)
    print('\nNormalized MSE: \n', MSE_normalized)

###################
# CLEANING TABLES #
###################

def tableCleaner(fname, true):
    """Enter the .npy filename, define 'true' before running containing the true
    values in order AS A LIST!!!!
    """
    with open(fname, 'rb') as f:
        res1 = numpy.load(f)
    #    res2 = numpy.load(f)
    print("The true values are", true)
    print("Number of repetitions before cleaning: ", len(res1))
    resnew = numpy.stack([arr for arr in res1 if numpy.max(numpy.abs(arr))<100])
    a = numpy.mean(resnew, axis=0)
    b= numpy.std(resnew, axis=0)
    true = numpy.stack([true] * resnew.shape[1])
    print("Number of repetitions after cleaning:", len(resnew))
    print('\n\nMeans:\n', a)
    print('Standard deviations:\n', b)
    print('\n\nMSE report\n')
    evaluatorMSE(a, b, true)

for n in [250, 1000, 4000, 10000]:
    print('n=', n)
    tableCleaner('LinearNW5'+str(n), true)
    print('\n\n\n')
