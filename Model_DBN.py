#from keras.utils import to_categorical
import numpy as np
from Evaluation import evaluation
from dbn.tensorflow import SupervisedDBNClassification


def Model_DBN(Train_Data, Train_Target, Test_Data, Test_Target, soln = None):
    sol = [soln, soln]
    classifier = SupervisedDBNClassification(hidden_layers_structure=[sol[0], sol[1]],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=1,
                                             n_iter_backprop=2,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    pred = np.zeros((Test_Target.shape[0], Test_Target.shape[1]))
    for i in range(Test_Target.shape[1]):
        print(i)
        classifier.fit(Train_Data, Train_Target[:,0])
        pred[:,i] = classifier.predict(Test_Data)
    Eval = evaluation(pred, Test_Target)
    return Eval, pred
