"""
Demo
"""

"""
To run the demo:

```
python eq_odds.py <path_to_model_predictions.csv>
```

`<path_to_model_predictions.csv>` should contain the following columns for the VALIDATION set:

- `prediction` (a score between 0 and 1)
- `label` (ground truth - either 0 or 1)
- `group` (group assignment - either 0 or 1)

Try the following experiments, which were performed in the paper:
```
python eq_odds.py data/income.csv
python eq_odds.py data/health.csv
python eq_odds.py data/criminal_recidivism.csv
```
"""
import pandas as pd
import sys

def equalized_odds(dataset):
  # Load the validation set scores from csvs
  data_filename = dataset
  test_and_val_data = pd.read_csv(data_filename)#sys.argv[1])

  # Randomly split the data into two sets - one for computing the fairness constants
  order = np.random.permutation(len(test_and_val_data)) #randomizes the list of indices
  val_indices = order[0::2] #get even index elements (the elements themselves are the original indices), i.e. starting from 0 with a step of 2
  test_indices = order[1::2] #get odd numbered index elements, i.e. starting from 1 with a step of 2
  val_data = test_and_val_data.iloc[val_indices]
  test_data = test_and_val_data.iloc[test_indices]

  # Create model objects - one for each group, validation and test
  group_0_val_data = val_data[val_data['group'] == 0]
  group_1_val_data = val_data[val_data['group'] == 1]
  group_0_test_data = test_data[test_data['group'] == 0]
  group_1_test_data = test_data[test_data['group'] == 1]

  group_0_val_model = Model(group_0_val_data['prediction'].to_numpy(), group_0_val_data['label'].to_numpy())
  group_1_val_model = Model(group_1_val_data['prediction'].to_numpy(), group_1_val_data['label'].to_numpy())
  group_0_test_model = Model(group_0_test_data['prediction'].to_numpy(), group_0_test_data['label'].to_numpy())
  group_1_test_model = Model(group_1_test_data['prediction'].to_numpy(), group_1_test_data['label'].to_numpy())

  # Find mixing rates for equalized odds models
  _, _, mix_rates = Model.eq_odds(group_0_val_model, group_1_val_model)

  # Apply the mixing rates to the test models
  eq_odds_group_0_test_model, eq_odds_group_1_test_model = Model.eq_odds(group_0_test_model,
                                                                          group_1_test_model,
                                                                          mix_rates)

  # Print results on test model
  print('Original group 0 model:\n%s\n' % repr(group_0_test_model))
  print('Original group 1 model:\n%s\n' % repr(group_1_test_model))
  print('Equalized odds group 0 model:\n%s\n' % repr(eq_odds_group_0_test_model))
  print('Equalized odds group 1 model:\n%s\n' % repr(eq_odds_group_1_test_model))

  #Plotting the convex hulls
  from scipy.spatial import ConvexHull
  import matplotlib.pyplot as plt
  from collections import OrderedDict

  #(group_0_test_model.tnr(), group_0_test_model.fnr())
  points = np.array([(0,0), (group_0_test_model.fpr(), group_0_test_model.tpr()), (1-group_0_test_model.fpr(), 1-group_0_test_model.tpr()) , (1,1)])
  hull = ConvexHull(points)
  mylabel = 'group 0'
  for simplex in hull.simplices:
      plt.plot(points[simplex,0], points[simplex,1], 'k-', label=mylabel)
      mylabel = "_nolegend_"

  points = np.array([(0,0), (group_1_test_model.fpr(), group_1_test_model.tpr()), (1-group_1_test_model.fpr(), 1-group_1_test_model.tpr()), (1,1)])
  hull = ConvexHull(points)
  mylabel = 'group 1'
  for simplex in hull.simplices:
      plt.plot(points[simplex,0], points[simplex,1], 'r-', label=mylabel)
      mylabel = "_nolegend_"
      
  plt.plot(group_0_test_model.fpr(), group_0_test_model.tpr(), 'k+', label="Original Group 0")
  plt.plot(group_1_test_model.fpr(), group_1_test_model.tpr(), 'r+', label="Original Group 1")
  plt.plot(eq_odds_group_0_test_model.fpr(), eq_odds_group_0_test_model.tpr(), 'k*', label="Eq Odds Group 0")
  plt.plot(eq_odds_group_1_test_model.fpr(), eq_odds_group_1_test_model.tpr(), 'r*', label="Eq Odds Group 1")

  plt.plot(0, 0, 'g+')

  plt.ylabel('Y = 1')
  plt.xlabel('Y = 0')
  plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
  plt.show()

  return group_0_test_model.fpr(), group_0_test_model.tpr(), group_1_test_model.fpr(), group_1_test_model.tpr(), eq_odd_group_0_test_model.fpr(), eq_odds_group_0_test_model.tpr(), eq_odds_group_1_test_model.fpr(), eq_odds_group_1_test_model.tpr()
