from Model_RNN import Model_MRNN_AM
from Model_TCNN import Model_MTCNN_AM


def Model_MDDHN_AM(Train_Data, Train_Target, Test_Data, Test_Target):
    Eval, pred_tcn = Model_MTCNN_AM(Train_Data, Train_Target, Test_Data, Test_Target)
    Pred, pred_rnn = Model_MRNN_AM(Train_Data, Train_Target, Test_Data, Test_Target)
    Predict = (pred_tcn+pred_rnn)/2
    return Eval, Predict