from Model_RNN import Model_RNN
from Model_TCNN import Model_TCNN


def Model_TCNN_RNN(Train_Data, Train_Target, Test_Data, Test_Target):
    Eval, pred_tcn = Model_TCNN(Train_Data, Train_Target, Test_Data, Test_Target)
    Pred, pred_rnn = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
    Predict = (pred_tcn+pred_rnn)/2
    return Eval, Predict