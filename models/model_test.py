# Temporary file
import sys

sys.path.append('../')

import models

# model = models.regression_model()
# model.summary()

models.train()

data = [
    [7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4],
    [7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8],
    [7.8,0.76,0.04,2.3,0.092,15.0,54.0,0.997,3.26,0.65,9.8],
    [11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8]
]

pred = []

for d in data:
    pred.append( models.predict_quality(d) )

print( 'Pred :', pred )
print( 'True :', [5,5,5,6])


# model = models.load_model()
# model.summary()