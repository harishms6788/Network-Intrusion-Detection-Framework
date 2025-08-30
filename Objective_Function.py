import numpy as np
from Global_Vars import Global_Vars

def objfun_feat(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            feat_col = sol[:round(len(sol) - 3)].astype(np.uint8)
            weight = sol[round(len(sol) - 3):]
            feat1 = Feat[0][:, :len(feat_col) - 20]
            feat2 = Feat[1][:, 10:len(feat_col) - 10]
            feat3 = Feat[2][:, :len(feat_col) - 20]
            feat = feat1 * weight[0] + feat2 * weight[1] + feat3 * weight[2]
            varience = np.var(feat)
            Fitn[i] = 1 / varience
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        feat_col = sol[:round(len(sol) - 3)].astype(np.uint8)
        weight = sol[round(len(sol) - 3):]
        feat1 = Feat[0][:, :len(feat_col) - 20]
        feat2 = Feat[1][:, 10:len(feat_col) - 10]
        feat3 = Feat[2][:, :len(feat_col) - 20]
        feat = feat1 * weight[0] + feat2 * weight[1] + feat3 * weight[2]
        varience = np.var(feat)
        Fitn = 1 / varience
        return Fitn
